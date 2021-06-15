import torch
import torch.nn as nn
import torch.functional as F
import torchsnooper
from loss_func import Dice_loss

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                                kernel_size = kernel_size, stride = stride, padding = padding)
        self.batch_norm = nn.BatchNorm3d(num_features = out_channels)
        self.relu = nn.LeakyReLU(inplace = True)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x

class ConvTranspose(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0, output_padding = 0):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels,
                                                    kernel_size = kernel_size, stride = stride,
                                                    padding = padding, output_padding = output_padding)
        self.batch_norm = nn.BatchNorm3d(num_features = out_channels)
        self.relu = nn.LeakyReLU(inplace = True)
    def forward(self, x):
        x = self.conv3d_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        return x
        
        
class U_Net_3D(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size = 2)

        self.encoder_1_1 = ConvBlock(1, 32)
        self.encoder_1_2 = ConvBlock(32, 64)
        self.encoder_2_1 = ConvBlock(64, 64)
        self.encoder_2_2 = ConvBlock(64, 128)
        self.encoder_3_1 = ConvBlock(128, 128)
        self.encoder_3_2 = ConvBlock(128, 256)
        self.encoder_4_1 = ConvBlock(256, 256)
        self.encoder_4_2 = ConvBlock(256, 512)

        self.deconv_4 = ConvTranspose(512, 512)
        self.deconv_3 = ConvTranspose(256, 256)
        self.deconv_2 = ConvTranspose(128, 128)

        self.decoder_3_1 = ConvBlock(768, 256)
        self.decoder_3_2 = ConvBlock(256, 256)
        self.decoder_2_1 = ConvBlock(384, 128)
        self.decoder_2_2 = ConvBlock(128, 128)
        self.decoder_1_1 = ConvBlock(192, 64)
        self.decoder_1_2 = ConvBlock(64, 64)
        self.final = ConvBlock(64, 1)

    #@torchsnooper.snoop()
    def forward(self, x):
        encoder_1_out = self.encoder_1_2(self.encoder_1_1(x))     #[n 64 64 64 64]
        encoder_2_out = self.encoder_2_2(self.encoder_2_1(self.max_pool(encoder_1_out)))   #[n 128 32 32 32]
        encoder_3_out = self.encoder_3_2(self.encoder_3_1(self.max_pool(encoder_2_out)))   #[n 256 16 16 16]
        encoder_4_out = self.encoder_4_2(self.encoder_4_1(self.max_pool(encoder_3_out)))   #[n 512 8 8 8]

        decoder_4_out = self.deconv_4(encoder_4_out)    #[n 512 16 16 16]
        concatenated_data_3 = torch.cat((encoder_3_out, decoder_4_out), dim = 1)  #[n 768 16 16 16]
        decoder_3_out = self.decoder_3_2(self.decoder_3_1(concatenated_data_3))   #[n 256 16 16 16]

        decoder_3_out = self.deconv_3(decoder_3_out)    #[n 256 32 32 32]
        concatenated_data_2 = torch.cat((encoder_2_out, decoder_3_out), dim = 1)    #[n 384 32 32 32]
        decoder_2_out = self.decoder_2_2(self.decoder_2_1(concatenated_data_2))     #[n 128 32 32 32]

        decoder_2_out = self.deconv_2(decoder_2_out)    #[n 128 64 64 64]
        concatenated_data_1 = torch.cat((encoder_1_out, decoder_2_out), dim = 1)    #[n 192 64 64 64]
        decoder_1_out = self.decoder_1_2(self.decoder_1_1(concatenated_data_1))     #[n 64 64 64 64]

        final_data = self.final(decoder_1_out)  #[n 1 64 64 64]

        return final_data

class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size = 2)

        self.encoder_1_1 = ConvBlock(1, 16)
        self.encoder_1_2 = ConvBlock(16, 16)
        self.encoder_2_1 = ConvBlock(16, 32)
        self.encoder_2_2 = ConvBlock(32, 32)
        self.encoder_3_1 = ConvBlock(32, 64)
        self.encoder_3_2 = ConvBlock(64, 64)
        self.encoder_4_1 = ConvBlock(64, 128)
        self.encoder_4_2 = ConvBlock(128, 128)

        self.deconv_4 = ConvTranspose(128, 128)
        self.deconv_3 = ConvTranspose(64, 64)
        self.deconv_2 = ConvTranspose(32, 32)

        self.decoder_3_1 = ConvBlock(192, 64)
        self.decoder_3_2 = ConvBlock(64, 64)
        self.decoder_2_1 = ConvBlock(96, 32)
        self.decoder_2_2 = ConvBlock(32, 32)
        
        self.final = nn.Conv3d(in_channels = 48, out_channels = 1,kernel_size = 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    #@torchsnooper.snoop()
    def forward(self, x):
        encoder_1_out = self.encoder_1_2(self.encoder_1_1(x))
        encoder_2_out = self.encoder_2_2(self.encoder_2_1(self.max_pool(encoder_1_out)))  
        encoder_3_out = self.encoder_3_2(self.encoder_3_1(self.max_pool(encoder_2_out)))  
        encoder_4_out = self.encoder_4_2(self.encoder_4_1(self.max_pool(encoder_3_out)))   

        decoder_4_out = self.deconv_4(encoder_4_out)    
        concatenated_data_3 = torch.cat((encoder_3_out, decoder_4_out), dim = 1) 
        decoder_3_out = self.decoder_3_2(self.decoder_3_1(concatenated_data_3)) 

        decoder_3_out = self.deconv_3(decoder_3_out)  
        concatenated_data_2 = torch.cat((encoder_2_out, decoder_3_out), dim = 1)    
        decoder_2_out = self.decoder_2_2(self.decoder_2_1(concatenated_data_2))     

        decoder_2_out = self.deconv_2(decoder_2_out) 
        concatenated_data_1 = torch.cat((encoder_1_out, decoder_2_out), dim = 1)  

        final_data = self.final(concatenated_data_1) 

        return final_data