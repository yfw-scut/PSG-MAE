import sys
import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model.losses import sim_loss, loss_cont, loss_recon_val

def train_one_epoch(encoder, decoder, optimizer, data_loader, device, epoch):
    loss_file_path = "./loss.txt"
    
    encoder.train()
    decoder.train()

    for param in encoder.parameters():
        param.requires_grad = True
    for param in decoder.parameters():
        param.requires_grad = True

    accu_loss = torch.zeros(1).to(device)   
    data_loader = tqdm(data_loader, file=sys.stdout, dynamic_ncols=True)

    with open(loss_file_path, "a") as f:
        for step, data in enumerate(data_loader):
            if data is None:
                continue

            psg_data, mask1, mask2 = data
            psg_data = psg_data.to(device, non_blocking=True)
            mask1 = mask1.to(device, non_blocking=True)
            mask2 = mask2.to(device, non_blocking=True)

            optimizer.zero_grad()

            input_2 = psg_data * mask1
            input_3 = psg_data * mask2

            encoded_2 = encoder(input_2)
            output_2 = decoder(encoded_2) * mask1

            encoded_3 = encoder(input_3)
            output_3 = decoder(encoded_3) * mask2

            loss = sim_loss(input_2, output_2) + sim_loss(input_3, output_3) + loss_cont(output_2, output_3)
            loss.backward()

            accu_loss += loss.detach()
            data_loader.set_description(
                f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()

        avg_loss = accu_loss.item() / (step + 1) if step > 0 else 0.0
        f.write(f"{avg_loss}\n")
        f.flush()

    return avg_loss

def validation(encoder, decoder, data_loader, device, epoch):
    encoder.eval()
    decoder.eval()
    
    accu_loss = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout, dynamic_ncols=True)

    psg_save_folder = "/data01/results/val"
    output_save_folder = "/data01/results/recon"
    ch_loss_file = "./val.txt"

    os.makedirs(psg_save_folder, exist_ok=True)
    os.makedirs(output_save_folder, exist_ok=True)

    channel_names = ['EEG', 'EEG2', 'EOG(L)', 'EMG', 'AIR FLOW']
    total_ch_losses = torch.zeros(len(channel_names)).to(device)

    with torch.no_grad():
        with open(ch_loss_file, "a") as f:
            for step, data in enumerate(data_loader):
                if data is None:
                    continue

                
                psg_data = data.to(device, non_blocking=True)

                encoded = encoder(psg_data)
                output = decoder(encoded)

                psg_npy = psg_data.cpu().numpy()
                output_npy = output.cpu().numpy()

                np.save(os.path.join(psg_save_folder, f"epoch{epoch}_val_step{step}.npy"), psg_npy)
                np.save(os.path.join(output_save_folder, f"epoch{epoch}_output_step{step}.npy"), output_npy)

                ch_losses, loss = loss_recon_val(psg_data, output)
                total_ch_losses += torch.tensor(ch_losses).to(device)
                accu_loss += loss

                data_loader.set_description(
                    f"[validation] loss: {accu_loss.item() / (step + 1):.4f}"
                )

            n_steps = max(len(data_loader), 1)
            avg_ch_losses = total_ch_losses / n_steps
            val_loss = accu_loss.item() / n_steps

            ch_loss_str = ", ".join([
                f"{name}: {loss.item():.6f}" for name, loss in zip(channel_names, avg_ch_losses)
            ])
            f.write(f"Epoch {epoch}, Total Loss: {val_loss:.6f}, Channel Losses: {ch_loss_str}\n")
            f.flush()

    return val_loss

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
