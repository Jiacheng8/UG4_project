from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # 输入 x 的形状是 [batch_size, num_channels, sequence_length]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # 输出 logits 的形状是 [batch_size, num_classes, sequence_length]
        return x
