import torch
import torch.nn as nn

class Classification_Head(nn.Module):
    def __init__(self, num_classes=5, kernel_sizes=[3, 5, 7]):
        super(Classification_Head, self).__init__()

        self.cnn_filters = {
            3: 32,   
            5: 64,  
            7: 128   
        }


        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=self.cnn_filters[k], kernel_size=k, stride=1, padding=k // 2),
                nn.BatchNorm1d(self.cnn_filters[k]),
                nn.ReLU()
            ) for k in kernel_sizes
        ])


        self.conv_reduce = nn.Conv1d(in_channels=sum(self.cnn_filters[k] for k in kernel_sizes), out_channels=64, kernel_size=1)


        self.global_pool = nn.AdaptiveAvgPool1d(1)  

        self.fc = nn.Sequential(
            nn.Linear(64, 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        conv_outs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1) 


        x = self.conv_reduce(x)  # [batch_size, 64, seq_len]


        x = self.global_pool(x)  # [batch_size, 64, 1]


        x = x.flatten(1)  # Flatten, except batch dimension

   
        x = self.fc(x)  # [batch_size, num_classes]

        return x