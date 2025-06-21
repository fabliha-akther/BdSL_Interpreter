import torch
import torch.nn as nn
import torch.nn.functional as F

# ANN Model
class SignLanguageANN(nn.Module):
    def __init__(self, input_size=63, hidden_sizes=[512, 256, 128], output_size=65):
        super(SignLanguageANN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)  # Additional layer

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)  # Leaky ReLU instead of ReLU
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

# CNN Model
class SignLanguageCNN(nn.Module):
    def __init__(self, output_size=65):
        super(SignLanguageCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 14, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# CNN2 Model

class SignLanguageCNN2(nn.Module):
    def __init__(self, output_size=65):
        super(SignLanguageCNN2, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.6)
        self.activation = nn.LeakyReLU(0.01)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 63)
            x = self.activation(self.bn1(self.conv1(dummy_input)))
            x = self.activation(self.bn2(self.conv2(x)))
            x = self.activation(self.bn3(self.conv3(x)))
            x = self.activation(self.bn4(self.conv4(x)))
            x = self.pool(x)
            self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

