import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Conv1DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channel)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.maxpool(x)
        return x

class Transformer_Block(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(Transformer_Block, self).__init__()
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
    
    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)

        return x.permute(0, 2, 1)

class PSG_Encoder(nn.Module):
    def __init__(self, patch_size=300, in_channel=5, d_model=256, 
                 nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(PSG_Encoder, self).__init__()

        conv_out_len = (patch_size - 3) // 3 + 1  
        conv_out_len = conv_out_len // 2  
        
        self.embedding = Conv1DBlock(in_channel, d_model, kernel_size=3, stride=3)
        self.transformer_block = Transformer_Block(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.patch_size = patch_size
        self.conv_out_len = conv_out_len
    
    def forward(self, x):

        batch_size, channels, seq_len = x.size()
        

        patches = x.unfold(2, self.patch_size, self.patch_size)
        

        patches = patches.permute(0, 2, 1, 3).contiguous()
        num_patches = patches.size(1)
        patches = patches.view(-1, channels, self.patch_size)

        x_emb = self.embedding(patches)  
        x_trans = self.transformer_block(x_emb) 
        
        x_trans = x_trans.view(batch_size, num_patches, -1, self.conv_out_len)
        x_trans = x_trans.permute(0, 2, 1, 3).contiguous()
        output = x_trans.view(batch_size, -1, num_patches * self.conv_out_len)
        
        return output