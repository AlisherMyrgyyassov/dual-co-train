import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# from torchvision.ops import SqueezeExcitation

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

# Standard SE block
# class SqueezeExcitation(nn.Module):
#     def __init__(self, ch_in, reduction=16):
#         super(SqueezeExcitation, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
#         self.fc = nn.Conv2d(ch_in, ch_in, kernel_size=1, groups=ch_in // reduction)  # Depthwise Conv
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         se_weight = self.gap(x)
#         se_weight = self.fc(se_weight)
#         se_weight = self.sigmoid(se_weight)
#         return x * se_weight

# Linear SE block
class SqueezeExcitation(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(ch_in, ch_in // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(ch_in // reduction, ch_in, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        se_weight = self.gap(x).view(b, c)  # Squeeze
        se_weight = self.fc1(se_weight)
        se_weight = self.relu(se_weight)
        se_weight = self.fc2(se_weight)
        se_weight = self.sigmoid(se_weight).view(b, c, 1, 1)  # Excitation
        return x * se_weight

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, use_bn=False, se_reduction=None):
        """
        Parameters:
        - ch_in: Number of input channels.
        - ch_out: Number of output channels.
        - use_bn: Whether to use Batch Normalization.
        - se_reduction: Reduction ratio for the SE block. Set to None to disable SE.
        """
        super(conv_block, self).__init__()
        self.use_bn = use_bn
        self.se_reduction = se_reduction

        # First convolution layer
        layers = [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        if use_bn:
            layers.insert(1, nn.GroupNorm(num_groups=8, num_channels=ch_out))  # Add BatchNorm right after Conv

        # Second convolution layer
        layers += [
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        if use_bn:
            layers.insert(len(layers) - 1, nn.GroupNorm(num_groups=8, num_channels=ch_out))  # Add BatchNorm

        self.conv = nn.Sequential(*layers)

        # Add optional SE block
        if se_reduction is not None:
            self.se = SqueezeExcitation(ch_out, ch_out // se_reduction)
        else:
            self.se = None

    def forward(self, x):
        x = self.conv(x)
        if self.se is not None:
            x = self.se(x)  # Apply SE block if enabled
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, use_bn=False, se_reduction=None):
        """
        Parameters:
        - ch_in: Number of input channels.
        - ch_out: Number of output channels.
        - use_bn: Whether to use Batch Normalization.
        - se_reduction: Reduction ratio for the SE block. Set to None to disable SE.
        """
        super(up_conv, self).__init__()
        self.use_bn = use_bn
        self.se_reduction = se_reduction

        # Upsampling + Conv layer
        layers = [
            nn.Upsample(scale_factor=2),  # Upsample by a factor of 2
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        if use_bn:
            layers.insert(2, nn.GroupNorm(num_groups=8, num_channels=ch_out))  # Add BatchNorm after Conv

        self.up = nn.Sequential(*layers)

        # Add optional SE block
        if se_reduction is not None:
            self.se = SqueezeExcitation(ch_out, ch_out // se_reduction)
        else:
            self.se = None

    def forward(self, x):
        x = self.up(x)
        if self.se is not None:
            x = self.se(x)  # Apply SE block if enabled
        return x



# ultraunet_small
# previous+decreased dec/enc by 1
class UltraUNet_small(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, n_channels=24):
        super(UltraUNet_small, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder with selective SE and GroupNorm
        self.Conv1 = conv_block(img_ch, n_channels, use_bn=False, se_reduction=None)  # Input layer
        self.Conv2 = conv_block(n_channels, n_channels * 2, use_bn=False, se_reduction=None)
        self.Conv3 = conv_block(n_channels * 2, n_channels * 4, use_bn=True, se_reduction=16)
        self.Conv4 = conv_block(n_channels * 4, n_channels * 8, use_bn=True, se_reduction=16)  # Bottleneck

        # Decoder with selective SE and GroupNorm
        self.Up4 = up_conv(n_channels * 8, n_channels * 4, use_bn=False, se_reduction=None)
        self.Up_conv4 = conv_block(n_channels * 4, n_channels * 4, use_bn=False, se_reduction=16)

        self.Up3 = up_conv(n_channels * 4, n_channels * 2, use_bn=False, se_reduction=None)
        self.Up_conv3 = conv_block(n_channels * 2, n_channels * 2, use_bn=False, se_reduction=16)

        self.Up2 = up_conv(n_channels * 2, n_channels, use_bn=False, se_reduction=None)
        self.Up_conv2 = conv_block(n_channels, n_channels, use_bn=False, se_reduction=None)

        self.Conv_1x1 = nn.Conv2d(n_channels, output_ch, kernel_size=1, stride=1, padding=0)  # Final output layer

    def forward(self, x):
        # Encoder Path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # Bottleneck

        # Decoder Path
        d4 = self.Up4(x4)
        d4 = x3 + d4  # Skip connection
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = x2 + d3  # Skip connection
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = x1 + d2  # Skip connection
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)  # Output layer
        return d1

# ultraunet v2
class UltraUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, n_channels=24, use_bn=False, se_reduction=None):
        """
        img_ch: Number of input channels (e.g., 3 for RGB images).
        output_ch: Number of output channels (e.g., 1 for binary segmentation).
        n_channels: Number of channels in the first encoder block, scales up in deeper layers.
        use_bn: Whether to use Batch Normalization globally (can be overridden selectively).
        se_reduction: Reduction ratio for SE blocks, set to None to disable SE globally.
        """
        super(UltraUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder with selective SE and BatchNorm
        self.Conv1 = conv_block(img_ch, n_channels, use_bn=False, se_reduction=None)  # Input layer (no SE, no BatchNorm)
        self.Conv2 = conv_block(n_channels, n_channels * 2, use_bn=False, se_reduction=None)  # Shallow encoder
        self.Conv3 = conv_block(n_channels * 2, n_channels * 4, use_bn=True, se_reduction=None)  # Deeper encoder, BatchNorm for stability
        self.Conv4 = conv_block(n_channels * 4, n_channels * 8, use_bn=True, se_reduction=16)  # Deeper layer, SE for feature recalibration
        self.Conv5 = conv_block(n_channels * 8, n_channels * 16, use_bn=True, se_reduction=32)  # Bottleneck, BatchNorm + SE for robustness

        # Decoder with selective SE and BatchNorm
        self.Up5 = up_conv(n_channels * 16, n_channels * 8, use_bn=False, se_reduction=None)  # Up-convolution (no BatchNorm for speed)
        self.Up_conv5 = conv_block(n_channels * 8, n_channels * 8, use_bn=False, se_reduction=16)  # SE for skip connection refinement

        self.Up4 = up_conv(n_channels * 8, n_channels * 4, use_bn=False, se_reduction=None)  # Up-convolution (no BatchNorm for speed)
        self.Up_conv4 = conv_block(n_channels * 4, n_channels * 4, use_bn=False, se_reduction=None)  # SE for skip connection refinement

        self.Up3 = up_conv(n_channels * 4, n_channels * 2, use_bn=False, se_reduction=None)  # Up-convolution
        self.Up_conv3 = conv_block(n_channels * 2, n_channels * 2, use_bn=False, se_reduction=None)  # No SE for shallow layers

        self.Up2 = up_conv(n_channels * 2, n_channels, use_bn=False, se_reduction=None)  # Up-convolution
        self.Up_conv2 = conv_block(n_channels, n_channels, use_bn=False, se_reduction=None)  # Final reconstruction layers (no SE)

        self.Conv_1x1 = nn.Conv2d(n_channels, output_ch, kernel_size=1, stride=1, padding=0)  # Final output layer

    def forward(self, x):
        # Encoder Path
        x1 = self.Conv1(x)  # Input layer
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # Shallow encoder
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # Deeper encoder
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # Deeper encoder with SE
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  # Bottleneck with SE

        # Decoder Path
        d5 = self.Up5(x5)
        d5 = x4 + d5  # Skip connection
        d5 = self.Up_conv5(d5)  # Refining skip connection with SE

        d4 = self.Up4(d5)
        d4 = x3 + d4  # Skip connection
        d4 = self.Up_conv4(d4)  # Refining skip connection with SE

        d3 = self.Up3(d4)
        d3 = x2 + d3  # Skip connection
        d3 = self.Up_conv3(d3)  # No SE

        d2 = self.Up2(d3)
        d2 = x1 + d2  # Skip connection
        d2 = self.Up_conv2(d2)  # No SE

        d1 = self.Conv_1x1(d2)  # Output layer
        return d1



class UltraUNet_large(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, n_channels=24, use_bn=False, se_reduction=None):
        """
        img_ch: Number of input channels (e.g., 3 for RGB images).
        output_ch: Number of output channels (e.g., 1 for binary segmentation).
        n_channels: Number of channels in the first encoder block, scales up in deeper layers.
        use_bn: Whether to use Batch Normalization globally (can be overridden selectively).
        se_reduction: Reduction ratio for SE blocks, set to None to disable SE globally.
        """
        super(UltraUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder with selective SE and BatchNorm
        self.Conv1 = conv_block(img_ch, n_channels, use_bn=False, se_reduction=16)  # Input layer (no SE, no BatchNorm)
        self.Conv2 = conv_block(n_channels, n_channels * 2, use_bn=False, se_reduction=16)  # Shallow encoder
        self.Conv3 = conv_block(n_channels * 2, n_channels * 4, use_bn=True, se_reduction=16)  # Deeper encoder, BatchNorm for stability
        self.Conv4 = conv_block(n_channels * 4, n_channels * 8, use_bn=True, se_reduction=16)  # Deeper layer, SE for feature recalibration
        self.Conv5 = conv_block(n_channels * 8, n_channels * 16, use_bn=True, se_reduction=32)  # Bottleneck, BatchNorm + SE for robustness

        # Decoder with selective SE and BatchNorm
        self.Up5 = up_conv(n_channels * 16, n_channels * 8, use_bn=False, se_reduction=None)  # Up-convolution (no BatchNorm for speed)
        self.Up_conv5 = conv_block(n_channels * 8, n_channels * 8, use_bn=False, se_reduction=16)  # SE for skip connection refinement

        self.Up4 = up_conv(n_channels * 8, n_channels * 4, use_bn=False, se_reduction=None)  # Up-convolution (no BatchNorm for speed)
        self.Up_conv4 = conv_block(n_channels * 4, n_channels * 4, use_bn=False, se_reduction=16) 

        self.Up3 = up_conv(n_channels * 4, n_channels * 2, use_bn=False, se_reduction=None)  # Up-convolution
        self.Up_conv3 = conv_block(n_channels * 2, n_channels * 2, use_bn=False, se_reduction=16)  

        self.Up2 = up_conv(n_channels * 2, n_channels, use_bn=False, se_reduction=None)  # Up-convolution
        self.Up_conv2 = conv_block(n_channels, n_channels, use_bn=False, se_reduction=16) 

        self.Conv_1x1 = nn.Conv2d(n_channels, output_ch, kernel_size=1, stride=1, padding=0)  # Final output layer

    def forward(self, x):
        # Encoder Path
        x1 = self.Conv1(x)  # Input layer
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # Shallow encoder
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # Deeper encoder
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # Deeper encoder with SE
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  # Bottleneck with SE

        # Decoder Path
        d5 = self.Up5(x5)
        d5 = x4 + d5  # Skip connection
        d5 = self.Up_conv5(d5)  # Refining skip connection with SE

        d4 = self.Up4(d5)
        d4 = x3 + d4  # Skip connection
        d4 = self.Up_conv4(d4)  # Refining skip connection with SE

        d3 = self.Up3(d4)
        d3 = x2 + d3  # Skip connection
        d3 = self.Up_conv3(d3)  # No SE

        d2 = self.Up2(d3)
        d2 = x1 + d2  # Skip connection
        d2 = self.Up_conv2(d2)  # No SE

        d1 = self.Conv_1x1(d2)  # Output layer
        return d1



# ultra_unet v1
# class conv_block(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(conv_block,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )


#     def forward(self,x):
#         x = self.conv(x)
#         return x

# class up_conv(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(up_conv,self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
# 		    nn.BatchNorm2d(ch_out),
# 			nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         x = self.up(x)
#         return x


# class UltraUNet(nn.Module):
#     def __init__(self,img_ch=3,output_ch=1, n_channels = 24):
#         super(UltraUNet,self).__init__()

#         self.n_channels = n_channels 
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=self.n_channels)
#         self.Conv2 = conv_block(ch_in=self.n_channels, ch_out=self.n_channels*2)
#         self.Conv3 = conv_block(ch_in=self.n_channels*2, ch_out=self.n_channels*4)
#         self.Conv4 = conv_block(ch_in=n_channels*4, ch_out=n_channels*8)
#         self.Conv5 = conv_block(ch_in=n_channels*8, ch_out=n_channels*16)

#         self.Up5 = up_conv(ch_in=n_channels*16, ch_out=n_channels*8)
#         self.Up_conv5 = conv_block(ch_in=n_channels*16, ch_out=n_channels*8)

#         self.Up4 = up_conv(ch_in=n_channels*8, ch_out=n_channels*4)
#         self.Up_conv4 = conv_block(ch_in=n_channels*8, ch_out=n_channels*4)
        
#         self.Up3 = up_conv(ch_in=n_channels*4, ch_out=n_channels*2)
#         self.Up_conv3 = conv_block(ch_in=n_channels*4, ch_out=n_channels*2)
        
#         self.Up2 = up_conv(ch_in=n_channels*2, ch_out=n_channels)
#         self.Up_conv2 = conv_block(ch_in=n_channels*2, ch_out=n_channels)

#         self.Conv_1x1 = nn.Conv2d(n_channels, output_ch, kernel_size=1, stride=1, padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4,d5),dim=1)
        
#         d5 = self.Up_conv5(d5)
        
#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1