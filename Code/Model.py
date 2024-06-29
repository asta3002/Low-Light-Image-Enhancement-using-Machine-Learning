from Components import *
from Data_Utils import *
from Custom_Losses import *

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")



class Low_light_image_enhancement():
    def __init__(self,ccn_weight_r,ccn_weight_c,ccn_weight_cr,mfn_weight_r,mfn_weight_c,mfn_weight_p,mfn_weight_s,lr,device):

        self.CCN = Channel_Correlation_Network(2,2).to(device)
        self.MFN = Multi_Fusion_Network(2,2).to(device)
        self.DEN = Detail_Enhancement_Network(3,device).to(device)
        self.parameters = list(self.CCN.parameters()) + list(self.MFN.parameters()) + list(self.DEN.parameters())
        self.ccn_criterion = CCNLoss(ccn_weight_r,ccn_weight_c,ccn_weight_cr).to(device)
        self.optimizer = optim.Adam(self.parameters, lr=lr)
        self.device = device

        self.mfn_criterion = MFNLoss(mfn_weight_r,mfn_weight_c,mfn_weight_p,mfn_weight_s,device).to(device)

        self.den_criterion = DENLoss(device).to(device)


    def Train(self,epochs,train_data_loader,save_model):

        
        
        for epoch in tqdm(range(epochs)):
            MSE = 0.0
            PSNR=0.0
            SSIM=0.0

            for batch in train_data_loader:
                batch =batch.to(self.device)

                RG,GB,RB  = Extract_Channels(batch,self.device)

                RG_1 = self.CCN(RG)
                GB_1 = self.CCN(GB)
                RB_1 = self.CCN(RB)

                ccnl1 = self.ccn_criterion(RG,RG_1)
                ccnl2= self.ccn_criterion(RB,RB_1)
                ccnl3 = self.ccn_criterion(GB,GB_1)
                ccn_loss = ccnl1 + ccnl2 + ccnl3




                w_rb =self.MFN(RB_1)
                w_gb =self.MFN(GB_1)
                w_rg =self.MFN(RG_1)
                A = torch.matmul(w_rb,RB_1)
                B = torch.matmul(w_gb,GB_1)
                C = torch.matmul(w_rg,RG_1)
                R_fus = A[:,0,:,:].unsqueeze(1) + C[:,0,:,:].unsqueeze(1)
                G_fus = B[:,0,:,:].unsqueeze(1) + C[:,1,:,:].unsqueeze(1)
                B_fus = A[:,1,:,:].unsqueeze(1) + B[:,1,:,:].unsqueeze(1)
                I_fus = torch.cat([R_fus,G_fus,B_fus],dim=1)

                mfn_loss = self.mfn_criterion(I_fus,batch)


                I_enh = self.DEN(I_fus)
                den_loss = self.den_criterion(I_enh,batch)
                # Loss = ccn_loss+mfn_loss+den_loss
                self.optimizer.zero_grad()  # Clear previous gradients
                # Loss.backward()
                ccn_loss.backward(retain_graph=True)
                mfn_loss.backward(retain_graph=True)
                den_loss.backward()
                self.optimizer.step()
                I_enh = I_enh.cpu()
                batch = batch.cpu()
                RG=RG.cpu()
                GB=GB.cpu()
                RB=RB.cpu()
                mse = F.mse_loss(batch, I_enh).item()
                psnr_value = psnr(batch.detach().numpy(),I_enh.detach().numpy())
                ssim_value, _ = ssim(batch.detach().numpy(),I_enh.detach().numpy(), full=True,channel_axis=1,data_range=255.0)
                MSE +=mse
                PSNR +=psnr_value
                SSIM +=ssim_value

        MSE /=len(train_data_loader)
        PSNR /=len(train_data_loader)
        SSIM /=len(train_data_loader)     
                



        if(save_model):
          print("MSE : ",MSE," PSNR : ",PSNR," SSIM : ",SSIM)
          torch.save(self.CCN.state_dict(), 'ccn_weights.pth')
          torch.save(self.MFN.state_dict(), 'mfn_weights.pth')
          torch.save(self.DEN.state_dict(), 'den_weights.pth')

    def load_model(self):
        self.CCN.load_state_dict(torch.load('ccn_weights.pth'))
        self.MFN.load_state_dict(torch.load('mfn_weights.pth'))
        self.DEN.load_state_dict(torch.load('den_weights.pth'))

    def Test(self,img_tensor):


        batch = img_tensor.unsqueeze(0).to(self.device)

        RG,GB,RB  = Extract_Channels(batch,self.device)

        RG_1 = self.CCN(RG)
        GB_1 = self.CCN(GB)
        RB_1 = self.CCN(RB)




        w_rb =self.MFN(RB_1)
        w_gb =self.MFN(GB_1)
        w_rg =self.MFN(RG_1)
        A = torch.matmul(w_rb,RB_1)
        B = torch.matmul(w_gb,GB_1)
        C = torch.matmul(w_rg,RG_1)
        R_fus = A[:,0,:,:].unsqueeze(1) + C[:,0,:,:].unsqueeze(1)
        G_fus = B[:,0,:,:].unsqueeze(1) + C[:,1,:,:].unsqueeze(1)
        B_fus = A[:,1,:,:].unsqueeze(1) + B[:,1,:,:].unsqueeze(1)
        I_fus = torch.cat([R_fus,G_fus,B_fus],dim=1)
        I_enh = self.DEN(I_fus)
        I_enh=I_enh[0]
        # img_tensor_normalized = img_tensor.clamp(0, 1)
        img_array = I_enh.cpu().detach().permute(1, 2, 0).numpy()

        # Display the image using matplotlib
        plt.imshow(img_array)
        plt.axis('off')  # Turn off axis labels
        plt.show()






        


data_folder = 'Train_Data'

train_data_loader = Get_Data(data_folder)


ccn_weight_r=1.0
ccn_weight_c=0.7
ccn_weight_cr=1.0
mfn_weight_r=0.1
mfn_weight_c=5
mfn_weight_p=3
mfn_weight_s=1.0
lr=0.001             
save_model  =False
epochs=1              
Model = Low_light_image_enhancement(ccn_weight_r,ccn_weight_c,ccn_weight_cr,mfn_weight_r,mfn_weight_c,mfn_weight_p,mfn_weight_s,lr,device)
Model.Train(2,train_data_loader,save_model)

# Model.load_model()
# for batch in train_data_loader:
#     img  =batch[0]
    
#     Model.Test(img)
    
#     break              


            
            
            




