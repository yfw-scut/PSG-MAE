import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, input_channels=256, output_channels=5, input_seq_len=50, output_seq_len=300):
        super(MLPDecoder, self).__init__()

        self.fc1 = nn.Linear(input_channels, output_channels)
        self.fc2 = nn.Linear(input_seq_len, output_seq_len)

    def forward(self, x):

        x = self.fc1(x.permute(0, 2, 1))  
        x = self.fc2(x.permute(0, 2, 1))  
        return x

class PSG_Decoder(nn.Module):
    def __init__(self, patch_size=50, patch_number=10):
        super(PSG_Decoder, self).__init__()
        self.patch_size = patch_size
        self.patch_number = patch_number
        self.mlp_decoder = MLPDecoder(input_channels=256, output_channels=5, input_seq_len=patch_size, output_seq_len=300)

    def forward(self, x):

        patches = [self.mlp_decoder(x[:, :, i*self.patch_size:(i+1)*self.patch_size]) 
                   for i in range(self.patch_number)]

        output = torch.cat(patches, dim=2)

        return output