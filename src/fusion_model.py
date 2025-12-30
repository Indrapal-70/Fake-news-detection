import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, dropout=0.2):
        super(FusionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 1) # Binary classification (Real/Fake)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_features, text_features):
        # Concatenate features
        combined_features = torch.cat((image_features, text_features), dim=1)
        
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        output = self.sigmoid(x)
        
        return output
