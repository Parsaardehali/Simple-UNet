import torch
import torch.nn as nn

# Define a small U-Net model
class SmallUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        
        # Decoder
        self.dec1 = self.conv_block(32, 16)
        self.dec2 = self.conv_block(16, out_channels)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        
        # Decoder
        d1 = self.dec1(self.upsample(e2))
        d2 = self.dec2(d1)  # Remove the second upsample
        
        return d2

class LargerUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargerUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        
        # Bridge
        self.bridge = self.conv_block(128, 256)
        
        # Decoder
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        
        # Final convolution
        self.final_conv = nn.Conv2d(32, out_channels, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        e3 = self.enc3(self.maxpool(e2))
        
        # Bridge
        bridge = self.bridge(self.maxpool(e3))
        
        # Decoder
        d3 = self.dec3(torch.cat([self.upsample(bridge), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Final convolution
        out = self.final_conv(d1)
        
        return out

class LargestUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super(LargestUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64, num_groups)
        self.enc2 = self.conv_block(64, 128, num_groups)
        self.enc3 = self.conv_block(128, 256, num_groups)
        self.enc4 = self.conv_block(256, 512, num_groups)
        
        # Bridge
        self.bridge = self.conv_block(512, 1024, num_groups)
        
        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512, num_groups)
        self.dec3 = self.conv_block(512 + 256, 256, num_groups)
        self.dec2 = self.conv_block(256 + 128, 128, num_groups)
        self.dec1 = self.conv_block(128 + 64, 64, num_groups)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, 32), num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch, num_groups):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        
        # Bridge
        bridge = self.bridge(self.maxpool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(bridge), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Final convolution
        out = self.final_conv(d1)
        
        return out