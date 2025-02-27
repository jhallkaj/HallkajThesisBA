import torch.nn as nn

class MordredLinearAutoEncoder(nn.Module):
    def __init__(self, input = 932, dropout = 0.2, leakyRelu = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(leakyRelu),
            #nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(leakyRelu),
            #nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyRelu),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(leakyRelu),
            nn.Dropout(dropout),

            nn.Linear(64, 32) # the bottleneck layer 32 features.
            )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(leakyRelu),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyRelu),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(leakyRelu),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(leakyRelu),

            nn.Linear(512, input)

        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder(encoded)
        return x

class FirstMordredLinearAutoEncoder(nn.Module):
    def __init__(self, input = 932, hidden = [256, 64, 32], dropout = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            

            nn.Linear(hidden[0],hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden[1], hidden[2])

            )

        self.decoder = nn.Sequential(
            nn.Linear(hidden[2], hidden[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden[1]),
            nn.Linear(hidden[1], hidden[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden[0]),
            nn.Linear(hidden[0], input)
            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder(encoded)
        return x