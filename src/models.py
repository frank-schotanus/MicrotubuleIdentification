"""
Neural network models for microtubule detection.

This module provides model architectures suitable for detecting point-like
features (microtubule centers) in cryo-EM images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    U-Net architecture for dense prediction tasks.
    
    This is a standard U-Net that outputs a heatmap or distance map
    of the same resolution as the input image. Suitable for detecting
    microtubule locations as spatial probability maps.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 init_features: int = 32):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels (1 for single heatmap)
            init_features: Number of features in first layer (doubled at each level)
        """
        super(UNet, self).__init__()
        
        features = init_features
        
        # Encoder (downsampling path)
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 
                                          kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 
                                          kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 
                                          kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 
                                          kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        # Final output layer
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output
        out = self.conv_out(dec1)
        
        return out
    
    @staticmethod
    def _block(in_channels, features, name):
        """
        Create a convolutional block (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU).
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


class SimpleConvNet(nn.Module):
    """
    Simpler fully-convolutional network for microtubule detection.
    
    This is a lighter alternative to U-Net, useful for faster experiments
    or when computational resources are limited.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_features: int = 32):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_features: Number of features in convolutional layers
        """
        super(SimpleConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, base_features, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(base_features)
        
        self.conv2 = nn.Conv2d(base_features, base_features * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(base_features * 2)
        
        self.conv3 = nn.Conv2d(base_features * 2, base_features * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_features * 4)
        
        self.conv4 = nn.Conv2d(base_features * 4, base_features * 2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_features * 2)
        
        self.conv5 = nn.Conv2d(base_features * 2, base_features, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(base_features)
        
        self.conv_out = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv_out(x)
        
        return x


def create_model(model_type: str = 'unet', **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('unet' or 'simple')
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Initialized model
    """
    if model_type == 'unet':
        return UNet(**kwargs)
    elif model_type == 'simple':
        return SimpleConvNet(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# TODO: Consider implementing more advanced architectures:
# - Attention U-Net for better feature localization
# - FPN (Feature Pyramid Network) for multi-scale detection
# - Custom architectures specifically designed for elongated structures
# - Pretrained backbones (ResNet, EfficientNet) as encoders
