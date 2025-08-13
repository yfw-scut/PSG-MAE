import torch
import torch.nn.functional as F


def loss_recon(psg_true, psg_pred):
    return F.mse_loss(psg_true, psg_pred, reduction='mean')


def loss_recon_val(psg_true, psg_pred):

    assert psg_true.shape == psg_pred.shape, "Input shapes must match"


    loss_per_channel = F.mse_loss(psg_true, psg_pred, reduction='none') 
    loss_per_channel = loss_per_channel.mean(dim=(0, 2))


    overall_loss = F.mse_loss(psg_true, psg_pred, reduction='mean')


    return loss_per_channel.detach().cpu().numpy().tolist(), overall_loss


def loss_cls(y_pred, y_true):
    ce_loss = torch.nn.CrossEntropyLoss()
    return ce_loss(y_pred, y_true)


def sim_loss(psg_true, psg_pred, alpha=0.01, beta=10.0, weights=None):
    cos_loss = 0.0
    mse_loss = 0.0
    batch_size, channel_no, time_step = psg_true.shape

    if weights is None:
        weights = torch.ones(channel_no, device=psg_true.device)
    else:
        weights = weights.to(psg_true.device)
    weights = weights / weights.sum() 

    for i in range(channel_no):

        cosine_sim = F.cosine_similarity(psg_true[:, i, :], psg_pred[:, i, :], dim=1)
        channel_cos_loss = 1 - cosine_sim.mean()
        cos_loss += weights[i] * channel_cos_loss


        channel_mse_loss = F.mse_loss(psg_true[:, i, :], psg_pred[:, i, :])
        mse_loss += weights[i] * channel_mse_loss

    loss = alpha * mse_loss + beta * cos_loss
    return loss


def triplet_loss(anchor, positive, negative, margin=0.5):
    """
    anchor:   [B, D]
    positive: [B, D]
    negative: [B * N, D]
    """
    pos_distance = F.pairwise_distance(anchor, positive)
    neg_distance = F.pairwise_distance(anchor, negative)
    loss = torch.relu(pos_distance - neg_distance + margin)
    return loss.mean()


def triplet_loss_batched(anchor, positive, negatives, margin=0.5):
    """
    更高效的批处理三元组损失计算。
    anchor: [B, D]
    positive: [B, D]
    negatives: [B, N, D]
    """
    B, N, D = negatives.shape


    anchor_exp = anchor.unsqueeze(1).expand(-1, N, -1)
    positive_exp = positive.unsqueeze(1).expand(-1, N, -1)


    anchor_flat = anchor_exp.reshape(-1, D)
    positive_flat = positive_exp.reshape(-1, D)
    negatives_flat = negatives.reshape(-1, D)


    pos_dist = F.pairwise_distance(anchor_flat, positive_flat)
    neg_dist = F.pairwise_distance(anchor_flat, negatives_flat)

    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def loss_cont(psg_mask1, psg_mask2, margin=0.5):
    """
    Patch-level triplet loss, optimized version.
    输入: [B, 5, 3000]
    """
    assert psg_mask1.shape[1:] == (5, 3000), "Expected shape [B, 5, 3000]"
    B, C, L = psg_mask1.shape
    patch_number = 10
    patch_size = L // patch_number

    total_loss = 0.0

    for i in range(patch_number):

        anchor = psg_mask1[:, :, i*patch_size:(i+1)*patch_size]    
        positive = psg_mask2[:, :, i*patch_size:(i+1)*patch_size]  


        negatives = torch.stack([
            psg_mask1[:, :, j*patch_size:(j+1)*patch_size]
            for j in range(patch_number) if j != i
        ], dim=1)  


        anchor_flat = anchor.reshape(B, -1)
        positive_flat = positive.reshape(B, -1)
        negatives_flat = negatives.reshape(B, patch_number - 1, -1)

        loss = triplet_loss_batched(anchor_flat, positive_flat, negatives_flat, margin)
        total_loss += loss

    return total_loss / patch_number
