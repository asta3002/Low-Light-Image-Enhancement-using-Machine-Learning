import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CCNConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CCNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class DENConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DENConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        x = self.conv(x)
        x = self.relu(x)
        return x   
       
class Channel_Correlation_Network(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Channel_Correlation_Network, self).__init__()
        self.Block1 = CCNConvBlock(input_channels,64)
        self.Block2 = CCNConvBlock(64,64)
        self.Block3 = CCNConvBlock(64,64)
        self.Block4 = CCNConvBlock(64,64)
        self.Block5 = CCNConvBlock(64,64)
        self.Block6 = CCNConvBlock(128,64)
        self.Block7 = CCNConvBlock(128,64)
        self.Block8 = CCNConvBlock(128,64)
        self.Block9 = CCNConvBlock(128,64)
        self.Block10 = CCNConvBlock(64,64)
        self.Out_Layer = nn.Conv2d(64,output_channels,kernel_size=(3,3),stride=1,padding=1)
      
   
    def forward(self,x):
        Encoder_Output = []
        x = self.Block1.forward(x)
        Encoder_Output.append(x)
        x = self.Block2.forward(x)
        Encoder_Output.append(x)
        x = self.Block3.forward(x)
        Encoder_Output.append(x)
        x = self.Block4.forward(x)
        Encoder_Output.append(x)
        x = self.Block5.forward(x)
        x = torch.cat((x,Encoder_Output.pop()),dim=1)
        x = self.Block6.forward(x)
        x = torch.cat((x,Encoder_Output.pop()),dim=1)
        x = self.Block7.forward(x)
        x =torch.cat((x,Encoder_Output.pop()),dim=1)
        x = self.Block8.forward(x)
        x = torch.cat((x,Encoder_Output.pop()),dim=1)
        x = self.Block9.forward(x)
        x = self.Block10.forward(x)
        out = self.Out_Layer.forward(x)
        return out


class Detail_Enhancement_Network(nn.Module):
    def __init__(self,input_channels,device):
        super(Detail_Enhancement_Network, self).__init__()
        self.Block1 = DENConvBlock(input_channels,64)
        self.Block2 = DENConvBlock(64,64)
        self.Block3 = DENConvBlock(64,64)
        self.Block4 = DENConvBlock(64,64)
        self.Out_Layer = nn.Conv2d(64,3,kernel_size=(3,3),stride=1,padding=1)
        self.mean_filter = (torch.ones(3,1,3, 3) / 9.0).to(device)

    def fm(self,image_a):
        
        self.mean_filter = (torch.ones(3,1,3, 3) / 9.0).to(self.device)  # Dividing by 9 for normalization
        output_image  = F.conv2d(image_a, self.mean_filter,groups=3, padding=1)
       
        return output_image
    
    def forward(self,x):

          # Dividing by 9 for normalization
        
        I_det_fm  = F.conv2d(x, self.mean_filter,groups=3,padding=1)
        I_det = x  -I_det_fm
        I_det = self.Block1(I_det)
        I_det = self.Block2(I_det)
        I_det = self.Block3(I_det)
        I_det = self.Block4(I_det)
        I_det = self.Out_Layer(I_det)
        return I_det+x
    
class Multi_Scale_Feature_Channel_Shuffle(nn.Module):
    def __init__(self,in_channels):
        super(Multi_Scale_Feature_Channel_Shuffle, self).__init__() 
        
       # 1x1 conv branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=(1,1)),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        # 1x1 conv -> 3x3 conv branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels,4 , kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
            
        )
        
        # 1x1 conv -> 5x5 conv branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels,32 , kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,4 , kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(in_channels,32 , kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32 , kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,4 , kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
    def ChannelShuffle(self,x):
        _, _, _, num_channels = x.size()

        # Generate a random permutation of channel indices
        permutation = torch.randperm(num_channels, device=x.device)

        # Shuffle the channels using the permutation
        shuffled_x = x.permute(0, 1, 2, 3)[..., permutation]

        return shuffled_x

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        
        x  = torch.cat([branch1x1,branch3x3,branch5x5,branch7x7],dim=1)
        x = self.ChannelShuffle(x)
        
        return x
        
class Multi_Fusion_Network(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Multi_Fusion_Network, self).__init__() 
        self.Block1 = Multi_Scale_Feature_Channel_Shuffle(in_channels)
        self.Block2 = Multi_Scale_Feature_Channel_Shuffle(16)
        self.Block3 = Multi_Scale_Feature_Channel_Shuffle(16)
        self.Block4 = Multi_Scale_Feature_Channel_Shuffle(16)
        self.Block5 = Multi_Scale_Feature_Channel_Shuffle(16)
        self.Block6 = Multi_Scale_Feature_Channel_Shuffle(16)
        self.Block7 = Multi_Scale_Feature_Channel_Shuffle(16)
        self.Block8 = Multi_Scale_Feature_Channel_Shuffle(16)

        self.Output_Layer = nn.Conv2d(16,out_channels,kernel_size=(3,3),stride=1,padding=1)
    
    def forward(self,x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = self.Block6(x)
        x = self.Block7(x)
        x = self.Block8(x)

        x = self.Output_Layer(x)
        return x
