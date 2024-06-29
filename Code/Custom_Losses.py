import torch
import torch.nn as nn
import torch.nn.functional as F


class CCNLoss(nn.Module):
    def __init__(self, weight_r,weight_c,weight_cr):
        super(CCNLoss, self).__init__()
        self.weight_r = weight_r
        self.weight_c = weight_c
        self.weight_cr = weight_cr
    def angle_difference(self,pixel_a,pixel_b):
        normalized_vector_A = F.normalize(pixel_a, p=2, dim=0)
        normalized_vector_B = F.normalize(pixel_b, p=2, dim=0)

        # Calculate cosine similarity
        cosine_similarity = torch.sum(normalized_vector_A * normalized_vector_B)

        # Clamp the values to avoid numerical instability in arccos
        cosine_similarity = torch.clamp(cosine_similarity, -1.0 + 1e-7, 1.0 - 1e-7)

        # Calculate the angle difference (arccos)
        angle_difference = torch.acos(cosine_similarity)

        return angle_difference
    
    def color_loss(self,image_a,image_b):
        c_loss=0
        for i in range(image_a.shape[2]):
            for j in range(image_a.shape[3]):
                pixel_a = image_a[:,:,i,j]
                pixel_b = image_b[:,:,i,j]
                angle_diff = self.angle_difference(pixel_a,pixel_b)
                if(i==0 and j==0):
                    c_loss = angle_diff
                else:
                    c_loss +=angle_diff
        return c_loss
    def channel_relevance_map(self,image_a):
        P = image_a[0,:,:].unsqueeze(0)
        Q = image_a[1,:,:].unsqueeze(0)
        P = P.view(-1,P.shape[1],P.shape[2])
        Q = Q.view(-1,P.shape[1],P.shape[2])
        flatten = P.shape[1]*P.shape[2]
        P = P.view(-1,flatten)
        Q = Q.view(-1,flatten)

        Q = torch.transpose(Q,0,1)

        X = torch.matmul(Q,P)
        X = F.softmax(X,dim=0)
        X = torch.mean(X)
        return X
    def channel_correlation_loss(self,image_a,image_b):
        X_pre = self.channel_relevance_map(image_a)
        X_nor = self.channel_relevance_map(image_b)
        
        cr_loss = torch.abs(X_pre - X_nor)
        return cr_loss
    def reconstruction_loss(self,image_a,image_b):
       criterion = nn.L1Loss(reduction='mean')
       return criterion(image_a,image_b)
        
    def forward(self, predictions, targets):
        # Your custom loss calculation
        
        r_loss = self.reconstruction_loss(predictions,targets)
        c_loss = self.color_loss(predictions,targets)
        cr_loss = self.channel_correlation_loss(predictions,targets)
        ccn_loss = self.weight_r*r_loss + self.weight_c*c_loss + self.weight_cr*cr_loss
        return ccn_loss
    


    
class MFNLoss(nn.Module):
    def __init__(self, weight_r,weight_c,weight_p,weight_s,device):
        super(MFNLoss, self).__init__()
        self.weight_r = weight_r
        self.weight_c = weight_c
        self.weight_p = weight_p
        self.weight_s = weight_s
        self.device =device
        self.sobel_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        self.kernel_left = torch.tensor([[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=torch.float32).to(self.device)
        self.kernel_right = torch.tensor([[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=torch.float32).to(self.device)
        self.kernel_up = torch.tensor([[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=torch.float32).to(self.device)
        self.kernel_down = torch.tensor([[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=torch.float32).to(self.device)
    
    def reconstruction_loss(self,image_a,image_b):
       criterion = nn.L1Loss(reduction='mean')
       return criterion(image_a,image_b)
        
    
    def angle_difference(self,pixel_a,pixel_b):
        normalized_vector_A = F.normalize(pixel_a, p=2, dim=0)
        normalized_vector_B = F.normalize(pixel_b, p=2, dim=0)

        # Calculate cosine similarity
        cosine_similarity = torch.sum(normalized_vector_A * normalized_vector_B)

        # Clamp the values to avoid numerical instability in arccos
        cosine_similarity = torch.clamp(cosine_similarity, -1.0 + 1e-7, 1.0 - 1e-7)

        # Calculate the angle difference (arccos)
        angle_difference = torch.acos(cosine_similarity)

        return angle_difference
    def color_loss(self,image_a,image_b):
        c_loss=0
        for i in range(image_a.shape[2]):
            for j in range(image_a.shape[3]):
                pixel_a = image_a[:,:,i,j]
                pixel_b = image_b[:,:,i,j]
                angle_diff = self.angle_difference(pixel_a,pixel_b)
                if(i==0 and j==0):
                    c_loss = angle_diff
                else:
                    c_loss += angle_diff
        return c_loss
    
    def gradient_operator(self,image_a):          
        

         # Apply Sobel filter separately to each channel
        R = image_a[0,0,:,:].unsqueeze(0) 
        G = image_a[0,1,:,:].unsqueeze(0) 
        B = image_a[0,2,:,:].unsqueeze(0) 
        gradients_per_r = F.conv2d(R, self.sobel_filter, padding=1)
        gradients_per_g = F.conv2d(G, self.sobel_filter, padding=1)
        gradients_per_b = F.conv2d(B, self.sobel_filter, padding=1)
        
        
        return  gradients_per_r,gradients_per_g,gradients_per_b
       
    def smoothness_loss(self,image_a,image_b):
        gradients_r,gradients_g,gradients_b = self.gradient_operator(image_a)
        gradients_r1,gradients_g1,gradients_b1 = self.gradient_operator(image_b)
        s_loss = torch.sum(torch.abs(gradients_r-gradients_r1)+torch.abs(gradients_g-gradients_g1)+torch.abs(gradients_b-gradients_b1))
        return  s_loss
    
    def spatial_consistency_loss(self,image_a,image_b):
       
        
        enh_image_pooled = F.avg_pool2d(image_a, kernel_size=4, stride=4, padding=0)
        inp_image_pooled = F.avg_pool2d(image_b, kernel_size=4, stride=4, padding=0)
        
        D_inp_left = F.conv2d(inp_image_pooled, self.kernel_left, stride=1, padding='same')
        D_inp_right = F.conv2d(inp_image_pooled, self.kernel_right, stride=1, padding='same')
        D_inp_up = F.conv2d(inp_image_pooled, self.kernel_up, stride=1, padding='same')
        D_inp_down = F.conv2d(inp_image_pooled, self.kernel_down, stride=1, padding='same')

        D_enh_left = F.conv2d(enh_image_pooled, self.kernel_left, stride=1, padding='same')
        D_enh_right = F.conv2d(enh_image_pooled, self.kernel_right, stride=1, padding='same')
        D_enh_up = F.conv2d(enh_image_pooled, self.kernel_up, stride=1, padding='same')
        D_enh_down = F.conv2d(enh_image_pooled, self.kernel_down, stride=1, padding='same')

        D_left = torch.square(D_inp_left - D_enh_left)
        D_right = torch.square(D_inp_right - D_enh_right)
        D_up = torch.square(D_inp_up - D_enh_up)
        D_down = torch.square(D_inp_down - D_enh_down)

        return torch.mean(D_left + D_right + D_up + D_down)

    def forward(self,predictions,targets):
        
        r_loss = self.reconstruction_loss(predictions,targets)
        c_loss = self.color_loss(predictions,targets)
        s_loss  =self.smoothness_loss(predictions,targets)
        p_loss = self.spatial_consistency_loss(predictions,targets)
        
        mfs_loss = self.weight_c*c_loss  +self.weight_r*r_loss + self.weight_p*p_loss + self.weight_s*s_loss
        return mfs_loss
 
class DENLoss(nn.Module):
    def __init__(self,device):
        super(DENLoss, self).__init__()
        self.device =device
        
        pass
    def fm(self,image_a):
        
        self.mean_filter = (torch.ones(3,1,3, 3) / 9.0).to(self.device)  # Dividing by 9 for normalization
        output_image  = F.conv2d(image_a, self.mean_filter,groups=3, padding=1)

        
        return output_image

    def forward(self,prediction,targets):
        B = (targets-self.fm(targets)) 
        criterion = nn.L1Loss(reduction='mean')
        den_loss = criterion(prediction,B)
        return den_loss


