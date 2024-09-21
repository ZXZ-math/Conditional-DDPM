import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       
        beta_t = beta_1 + (t_s-1)/(T-1)*(beta_T-beta_1)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1-beta_t
        oneover_sqrt_alpha = 1/torch.sqrt(alpha_t)
        B = t_s.size(0)
        alpha_t_bar = torch.zeros((B,1))
        time = t_s.view(-1,)
        
        for i in range(B):
            t = time[i]
            prev_beta_ts = beta_1 + (beta_T - beta_1) * ((torch.linspace(1,t,int(t))-1)/ (T-1))
            alpha_t_bar[i] = torch.cumprod(1 - prev_beta_ts, dim=0)[int(t)-1]
                
        
        
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1-alpha_t_bar)
        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }
        
    def set_rows_to_minus_one(self,input_tensor, p):
            
            # Set each row of the input tensor to be a row of -1 with probability p.
            # Args:
            #     input_tensor (torch.Tensor): The input tensor of shape (B, N).
            #     p (float): The probability of setting each row to be a row of -1
            # Returns:
            #     torch.Tensor: The modified tensor.

            # Iterate over each row
            for i in range(input_tensor.size(0)):
                # Check if this row should be set to -1
                if torch.rand(1).item() < p:
                    input_tensor[i] = -1  # Set the entire row to -1
            return input_tensor

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        num_classes = self.dmconfig.num_classes
        condition_mask_value = self.dmconfig.condition_mask_value
        mask_p = self.dmconfig.mask_p
        B = images.size(0)
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  
        one_hot_vec = F.one_hot(conditions,num_classes = num_classes)
        
        
        
        one_hot_vec = self.set_rows_to_minus_one(one_hot_vec,mask_p)
        time = torch.randint(low=1, high=T+1, size=(B,1))
        time = time.to(device)
        noise = torch.randn_like(images)  
        sch = self.scheduler(time)
        sqrt_alpha_bar = sch['sqrt_alpha_bar'].view(B,1,1,1)
        sqrt_alpha_bar = sqrt_alpha_bar.to(device)
        sqrt_oneminus_alpha_bar = sch['sqrt_oneminus_alpha_bar'].view(B,1,1,1)
        sqrt_oneminus_alpha_bar=sqrt_oneminus_alpha_bar.to(device)
       
        corrupted_images = images * sqrt_alpha_bar + noise * sqrt_oneminus_alpha_bar
        normalized_time = time/T
        
        predicted_noise = self.network(corrupted_images,normalized_time,one_hot_vec)
        noise_loss = self.loss_fn(noise,predicted_noise)

        
        
        



        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        num_classes = self.dmconfig.num_classes
        image_size = self.dmconfig.input_dim
        image_size_ = image_size[0]
        B = conditions.size(0)
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  
        
        one_hot_uncond = -torch.ones_like(conditions)
        X_T = torch.randn(B,1,image_size_,image_size_)
        X_T = X_T.to(device)
        with torch.no_grad():
            for t in range(T,0,-1):
                if t>1:
                    z = torch.randn_like(X_T).to(device)
                else:
                    z = torch.zeros_like(X_T).to(device)
                time = torch.ones((B,1)) * t 
                time = time.to(device)
                normalized_time = time/T
    
                eps_tilde = (1 + omega) * self.network(X_T,normalized_time,conditions) - omega*self.network(X_T,normalized_time,one_hot_uncond)
                sch = self.scheduler(time)
                oneover_sqrt_alpha = sch['oneover_sqrt_alpha'].view(B,1,1,1)
                alpha_t = sch['alpha_t'].view(B,1,1,1)
                sqrt_oneminus_alpha_bar = sch['sqrt_oneminus_alpha_bar'].view(B,1,1,1)
                sqrt_beta_t = sch['sqrt_beta_t'].view(B,1,1,1)
                sqrt_oneminus_alpha_bar = sqrt_oneminus_alpha_bar.to(device)
                X_T = oneover_sqrt_alpha * (X_T - (1-alpha_t)/sqrt_oneminus_alpha_bar*eps_tilde) + sqrt_beta_t * z


        

        # ==================================================== #
        generated_images = (X_T * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images