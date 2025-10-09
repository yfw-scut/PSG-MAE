import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE

from dataset.dataset import (
    DownstreamSleepDataset,
    AddGaussianNoise,
    RandomChannelDrop,
    RandomTimeStretch,
    Compose
)
from models.PSG_Encoder import PSG_Encoder
from models.SleepStageClassifier import SleepStageClassifier

LABEL_DICT = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "R"
}

def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", annot_kws={"size": 12})
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_f1_curve(f1_list, val_interval, save_path):
    plt.figure()
    epochs = np.arange(1, len(f1_list) + 1) * val_interval
    plt.plot(epochs, f1_list, marker='o')
    plt.title("Validation Macro F1 Score", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Macro F1", fontsize=16)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_loss_curve(loss_list, save_path):
    plt.figure()
    epochs = np.arange(1, len(loss_list) + 1)
    plt.plot(epochs, loss_list, marker='o', color='orange')
    plt.title("Training Loss Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Avg Loss", fontsize=16)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def print_per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accs = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(accs):
        print(f"  Class {LABEL_DICT[i]} Accuracy: {acc:.4f}")

def train_one_fold(fold, train_subjects, test_subjects, args):
    print(f"\n=== Fold {fold + 1} ===")

    val_count = int(len(train_subjects) * 0.2)
    val_subjects = train_subjects[:val_count]
    train_subjects = train_subjects[val_count:]

    if args.augment:
        train_transform = Compose([
            AddGaussianNoise(std=0.02, prob=0.5),
            RandomChannelDrop(drop_prob=0.3),
            RandomTimeStretch(stretch_range=(0.9, 1.1), prob=0.3)
        ])
    else:
        train_transform = None

    train_dataset = DownstreamSleepDataset(args.h5_path, train_subjects, transform=train_transform)
    val_dataset = DownstreamSleepDataset(args.h5_path, val_subjects)
    test_dataset = DownstreamSleepDataset(args.h5_path, test_subjects)

    print(f"[Dataset] Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 构造模型结构
    encoder = PSG_Encoder()
    classifier = SleepStageClassifier()
    model = nn.Sequential(encoder, classifier)

    # 加载 state_dict 权重
    state_dict = torch.load(args.model_weight, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)

    if args.freeze:
        for param in model[0].parameters():
            param.requires_grad = False

    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

    best_val_f1 = 0
    best_model_path = os.path.join(args.save_dir, f"fold_{fold+1}_best_model.pth")
    val_f1_list = []
    train_loss_list = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"[Fold {fold+1}] Epoch {epoch}/{args.epochs}")
        for x, y, _ in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_loss)
        print(f"[Fold {fold+1}] Epoch {epoch} Done | Avg Loss: {avg_loss:.4f}")

        if epoch % args.val_interval == 0:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x, y, _ in tqdm(val_loader, desc=f"[Fold {fold+1}] Validating Epoch {epoch}"):
                    x = x.to(device)
                    out = model(x)
                    pred = out.argmax(dim=1).cpu().numpy()
                    preds.extend(pred)
                    targets.extend(y.numpy())

            f1 = f1_score(targets, preds, average='macro')
            acc = accuracy_score(targets, preds)
            print(f"[Fold {fold+1}] Val Acc: {acc:.4f}, Val MF1: {f1:.4f}")
            print(f"[Fold {fold+1}] Per-class Val Accuracy:")
            print_per_class_accuracy(targets, preds)

            val_f1_list.append(f1)
            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save(model.state_dict(), best_model_path)
                print(f"[Fold {fold+1}] Saved best model at epoch {epoch}, val MF1: {f1:.4f}")

    plot_f1_curve(val_f1_list, args.val_interval, os.path.join(args.save_dir, f"fold_{fold+1}_val_f1_curve.png"))
    plot_loss_curve(train_loss_list, os.path.join(args.save_dir, f"fold_{fold+1}_train_loss_curve.png"))

    # === Test ===
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    preds, targets, features, subj_ids = [], [], [], []

    with torch.no_grad():
        for x, y, subj in test_loader:
            x = x.to(device)
            out = model(x)
            feats = model[0](x)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(y.numpy())
            features.extend(feats)
            subj_ids.extend(subj)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    report = classification_report(targets, preds, target_names=[LABEL_DICT[i] for i in range(5)], digits=4)
    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, [LABEL_DICT[i] for i in range(5)], os.path.join(args.save_dir, f"fold_{fold+1}_confusion_matrix.png"))

    with open(os.path.join(args.save_dir, f"fold_{fold+1}_report.txt"), 'w') as f:
        f.write(report)

    print(f"[Fold {fold+1}] Test Acc: {acc:.4f}, Test MF1: {f1:.4f}")
    print(f"[Fold {fold+1}] Per-class Test Accuracy:")
    print_per_class_accuracy(targets, preds)

    try:
        tsne_dir = os.path.join(args.save_dir, f"fold_{fold+1}_tsne")
        os.makedirs(tsne_dir, exist_ok=True)

        subject_to_feats = {}
        subject_to_labels = {}

        for feat, label, subj in zip(features, targets, subj_ids):
            if subj not in subject_to_feats:
                subject_to_feats[subj] = []
                subject_to_labels[subj] = []
            subject_to_feats[subj].append(feat)
            subject_to_labels[subj].append(label)

        for subj in subject_to_feats:
            feats = np.stack(subject_to_feats[subj])
            labels = np.array(subject_to_labels[subj])

            if len(np.unique(labels)) < 2:
                print(f"[TSNE] Skip {subj} due to single class.")
                continue

            tsne_result = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feats)
            plt.figure(figsize=(8, 6))
            for label in np.unique(labels):
                idx = labels == label
                plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=LABEL_DICT[label], alpha=0.6, s=10)
            plt.legend()
            plt.title(f"t-SNE Subject {subj} (Fold {fold+1})", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(tsne_dir, f"{subj}.png"))
            plt.close()
    except Exception as e:
        print(f"[Fold {fold+1}] TSNE failed: {e}")

    return acc, f1, report, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--model_weight', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    all_subjects = sorted([s.replace(".h5", "") for s in os.listdir(args.h5_path) if s.endswith(".h5")])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_acc, all_f1 = [], []
    total_cm = np.zeros((5, 5), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_subjects)):
        train_subjects = [all_subjects[i] for i in train_idx]
        test_subjects = [all_subjects[i] for i in test_idx]

        acc, f1, report, cm = train_one_fold(fold, train_subjects, test_subjects, args)
        all_acc.append(acc)
        all_f1.append(f1)
        total_cm += cm

    print("=== Summary ===")
    avg_acc = np.mean(all_acc)
    avg_f1 = np.mean(all_f1)
    print(f"Avg Acc: {avg_acc:.4f}, Avg MF1: {avg_f1:.4f}")

    with open(os.path.join(args.save_dir, "final_result.txt"), 'w') as f:
        f.write("=== 5-Fold Cross Validation Summary ===\n")
        f.write(f"Average Accuracy: {avg_acc:.4f}\n")
        f.write(f"Average Macro F1: {avg_f1:.4f}\n\n")

        report_dict = classification_report(
            y_true=np.repeat(np.arange(5), total_cm.sum(axis=1)),
            y_pred=np.concatenate([np.repeat(np.arange(5), row) for row in total_cm]),
            target_names=[LABEL_DICT[i] for i in range(5)],
            digits=4,
            output_dict=True
        )
        f.write("Per-Class Metrics (from aggregated confusion matrix):\n")
        for label in LABEL_DICT.values():
            metrics = report_dict[label]
            f.write(f"{label}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}\n")

    plot_confusion_matrix(total_cm, [LABEL_DICT[i] for i in range(5)], os.path.join(args.save_dir, "final_confusion_matrix.png"))

if __name__ == '__main__':
    main()
