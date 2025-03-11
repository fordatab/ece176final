import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# Channel-wise Fully Connected Module
#########################################
class ChannelWiseFC(nn.Module):
    def __init__(self, num_channels, spatial_size):
        """
        For an input of shape (B, C, H, W) where H == W == spatial_size,
        this module applies an independent fully-connected layer per channel.
        """
        super(ChannelWiseFC, self).__init__()
        self.num_channels = num_channels
        self.spatial_size = spatial_size
        # Each channel gets its own (spatial_size*spatial_size x spatial_size*spatial_size) weight matrix.
        self.fc = nn.Parameter(torch.Tensor(num_channels, spatial_size * spatial_size, spatial_size * spatial_size))
        self.bias = nn.Parameter(torch.Tensor(num_channels, spatial_size * spatial_size))
        self.reset_parameters()
        
    def reset_parameters(self):
        for i in range(self.num_channels):
            nn.init.xavier_uniform_(self.fc[i])
            nn.init.zeros_(self.bias[i])
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1)  # shape: (B, C, H*W)
        # For each channel, perform a linear transformation (batch multiplication)
        out = torch.bmm(x_flat, self.fc) + self.bias.unsqueeze(0)  # (B, C, H*W)
        out = out.view(B, C, H, W)
        return out

#########################################
# Encoder: Inspired by AlexNet’s conv layers
#########################################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # A simplified version of AlexNet's first layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNormalization2d()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # (B, 64, ~55, ~55)
        x = self.pool2(self.relu2(self.conv2(x)))  # (B, 192, ~27, ~27)
        x = self.relu3(self.conv3(x))              # (B, 384, ~27, ~27)
        x = self.relu4(self.conv4(x))              # (B, 256, ~27, ~27)
        x = self.pool5(self.relu5(self.conv5(x)))    # Expected to be (B, 256, 6, 6) for 227x227 input
        return x

#########################################
# Decoder: Upsampling with Transposed Convs
#########################################
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Series of transposed convolutions (up-convolutions)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 6x6 -> 12x12
        self.relu1   = nn.ReLU(inplace=True)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 12x12 -> 24x24
        self.relu2   = nn.ReLU(inplace=True)
        
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 24x24 -> 48x48
        self.relu3   = nn.ReLU(inplace=True)
        
        # Final up-conv to generate a 3-channel output (using tanh to bound outputs)
        self.upconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)     # 48x48 -> 96x96
        self.relu4   = nn.ReLU(inplace=True)
        
        # (For higher resolution, more layers or interpolation may be applied)
        self.upconv5 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)     # 96 -> 192

        
    def forward(self, x):
        x = self.relu1(self.upconv1(x))
        x = self.relu2(self.upconv2(x))
        x = self.relu3(self.upconv3(x))
        x = torch.tanh(self.upconv4(x))
        x = F.interpolate(x, size=(227, 227), mode='bilinear', align_corners=False)

        return x

#########################################
# Context Encoder: Combines Encoder, Channel-wise FC, and Decoder
#########################################
class ContextEncoder(nn.Module):
    def __init__(self, use_channel_fc=True):
        super(ContextEncoder, self).__init__()
        self.encoder = Encoder()
        self.use_channel_fc = use_channel_fc
        # Assuming encoder output is (B, 256, 6, 6)
        if self.use_channel_fc:
            self.channel_fc = ChannelWiseFC(num_channels=256, spatial_size=6)
        self.decoder = Decoder()
        
    def forward(self, x):
        encoded = self.encoder(x)  # (B, 256, 6, 6)
        if self.use_channel_fc:
            encoded = self.channel_fc(encoded)
        decoded = self.decoder(encoded)
        return decoded

#########################################
# Example usage
#########################################
if __name__ == "__main__":
    # Create a model instance
    model = ContextEncoder()
    # Dummy input: batch of 4 images of size 227x227 with 3 channels
    input_tensor = torch.randn(4, 3, 227, 227)
    output = model(input_tensor)
    print("Output shape:", output.shape)
