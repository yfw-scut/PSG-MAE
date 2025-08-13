import os
import argparse
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.PSG_Encoder import PSG_Encoder
from model.PSG_Decoder import PSG_Decoder
from data.PSG_dataset import PSG_dataset
from model.utils import reset_weights, train_one_epoch, validation

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    if batch[0] is None:
        return None
    if isinstance(batch[0], tuple) and len(batch[0]) == 3:
        data = [item[0] for item in batch]
        mask1 = [item[1] for item in batch]
        mask2 = [item[2] for item in batch]
        return (
            torch.stack(data),
            torch.stack(mask1),
            torch.stack(mask2)
        )
    else:
        return torch.stack(batch)

def main(args):
    set_seed(42)
    os.makedirs("./weights", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    h5_directory = args.h5_dir
    if not os.path.isdir(h5_directory):
        raise FileNotFoundError(f"HDF5 directory not found: {h5_directory}")

    subject_list = [
        os.path.splitext(f)[0] for f in os.listdir(h5_directory)
        if f.endswith('.h5') and os.path.isfile(os.path.join(h5_directory, f))
    ]
    if not subject_list:
        raise RuntimeError(f"No HDF5 files found in directory: {h5_directory}")

    print(f"Found {len(subject_list)} subjects in the directory")

    random.shuffle(subject_list)
    total_subjects = len(subject_list)
    train_end = int(total_subjects * args.train_split)
    train_subjects = subject_list[:train_end]
    val_subjects = subject_list[train_end:]

    print(f"Training subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}")

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 12])
    print(f'Using {nw} dataloader workers per process')

    encoder = PSG_Encoder().to(device)
    decoder = PSG_Decoder().to(device)
    encoder.apply(reset_weights)
    decoder.apply(reset_weights)

    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")

    print("Creating training dataset...")
    train_dataset = PSG_dataset(
        h5_directory=h5_directory,
        subject_list=train_subjects,
        wt_mask=True
    )

    print("Creating validation dataset...")
    val_dataset = PSG_dataset(
        h5_directory=h5_directory,
        subject_list=val_subjects,
        wt_mask=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=collate_fn,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=min(nw, 4),
        collate_fn=collate_fn,
        persistent_workers=(nw > 0),
        prefetch_factor=1 if nw > 0 else None,
        drop_last=False
    )

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    best_epoch = 0
    val_epoch = args.val_start

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 30} Epoch {epoch}/{args.epochs} {'=' * 30}")
        train_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )
        print(f"Training loss: {train_loss:.4f}")

        if epoch >= val_epoch and (epoch % args.val_freq == 0):
            print("\nStarting validation...")
            val_loss = validation(
                encoder=encoder,
                decoder=decoder,
                data_loader=val_loader,
                device=device,
                epoch=epoch
            )
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, f"./weights/best_checkpoint.pth")

        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), f"./weights/encoder_epoch_{epoch}.pth")
            torch.save(decoder.state_dict(), f"./weights/decoder_epoch_{epoch}.pth")

    print(f"\nTraining complete. Best epoch: {best_epoch} with loss {best_val_loss:.4f}")
    torch.save(encoder.state_dict(), "./weights/final_encoder.pth")
    torch.save(decoder.state_dict(), "./weights/final_decoder.pth")
    torch.save({
        'epoch': args.epochs,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "./weights/final_checkpoint.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PSG Masked Autoencoder Pretraining')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_start', type=int, default=5)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--h5_dir', type=str, default='/data02/latest_pretrain_hdf5', help='Path to HDF5 dataset')

    args = parser.parse_args()

    print("\nStarting training with parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    main(args)
    print("\nTraining completed successfully!")
