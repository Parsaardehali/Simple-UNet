import torch.nn as nn

# Define a small U-Net model
class SmallUNet(nn.Module):
    """
    A small U-Net model for image segmentation tasks.
    
    Attributes:
        enc1 (nn.Sequential): The first encoding layer, which downsamples the input image.
        enc2 (nn.Sequential): The second encoding layer, which further downsamples the image.
        dec1 (nn.Sequential): The first decoding layer, which upsamples the feature map.
        dec2 (nn.Sequential): The second decoding layer, which produces the final output.
        maxpool (nn.MaxPool2d): A max pooling layer to reduce spatial dimensions.
        upsample (nn.Upsample): An upsampling layer to increase spatial dimensions.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initializes the SmallUNet model with given input and output channels.
        
        Args:
            in_channels (int): The number of channels in the input images.
            out_channels (int): The number of channels in the output segmentation masks.
        """
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
        """
        Creates a convolutional block with two convolutional layers followed by ReLU activations.
        
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        
        Returns:
            nn.Sequential: A sequential container of the convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the SmallUNet model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        
        # Decoder
        d1 = self.dec1(self.upsample(e2))
        d2 = self.dec2(d1)  # Remove the second upsample
        
        return d2