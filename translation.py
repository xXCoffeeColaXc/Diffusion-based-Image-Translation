import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from ddim_modules import UNet
from seg_modules import Segmentor, MyCityscapesDataset, transform, transform_mask, mean, std, decode_segmap, encode_segmap
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchviz import make_dot

class DiffusionTranslationSGG(object):
    def __init__(self, cfg, test_image, test_mask, model_checkpoint_path, seg_model_path):
        self.config = cfg
        self.build_model(model_checkpoint_path)
        self.build_seg_model(seg_model_path)
        self.test_image = test_image
        self.test_mask = test_mask

        self.test_image = self.test_image.to(self.config['device'])
        self.test_maks = self.test_mask.to(self.config['device'])

    def build_model(self, model_checkpoint_path):
        # Create
        self.unet = UNet(self.config['c_in'], self.config['c_out'], self.config['image_size'], self.config['conv_dim'], self.config['block_depth'], self.config['time_emb_dim'])
        
        # Load
        checkpoint = torch.load(model_checkpoint_path, map_location=lambda storage, loc: storage)
        self.unet.load_state_dict(checkpoint['model_state_dict'])

        # Compute alpha, beta, alpha_hat
        self.beta = self.prepare_noise_schedule().to(self.config['device'])
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.unet.to(self.config['device'])
        
    def build_seg_model(self, seg_model_path):
        self.seg_model = Segmentor(seg_model_path, device=self.config['device'], backbone_name=self.config['seg_backbone'])
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')

        
    def denorm(self, image):
        mean_expanded = mean.view(3, 1, 1).cpu().detach()
        std_expanded = std.view(3, 1, 1).cpu().detach()

        # Denormalize
        x_adj = (image * std_expanded + mean_expanded) * 255
        x_adj = x_adj.clamp(0, 255).type(torch.uint8)
        return x_adj
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.config['beta_start'], self.config['beta_end'], self.config['noise_steps'])
    
    def forward_process(self, x, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        
        # Remove the batch dimension if it was originally a single image
        if noisy_image.size(0) == 1:
            noisy_image = noisy_image.squeeze(0)
        
        return noisy_image
    
    def add_noise_for_steps(self):
        # Ensure test_image is a single image (not batched)
        test_image_single = self.test_image.squeeze(0)  # Removes the batch dimension if it's present
        noisy_images = [test_image_single.cpu().detach()]

        for step in range(1, self.config['num_steps']): # NOTE: num_steps is the number of steps to visualize ? 
            # Gradually increase the noise level
            noise_level = step / self.config['num_steps']
            t = torch.tensor([int(self.config['noise_steps'] * noise_level)]).to(self.config['device'])

            noise = torch.randn_like(test_image_single).to(self.config['device'])
            noisy_image = self.forward_process(test_image_single, noise, t)
            noisy_images.append(noisy_image.cpu().detach())

        # Add pure noise in the last step
        pure_noise = torch.randn_like(test_image_single).to(self.config['device'])
        noisy_images.append(pure_noise.cpu().detach())

        return noisy_images


    def remove_noise_for_steps(self, noise_image):
        # Start with pure noise
        #noise_image = torch.randn((1, 3, self.config['image_size'], self.config['image_size'])).to(self.config['device'])
        x_t = noise_image.unsqueeze(0).to(self.config['device'])
        print("x_t shape: ", x_t.shape)
        denoised_images = [noise_image.squeeze(0).cpu().detach()]
        decoded_outputs = []

        y = encode_segmap(self.test_mask) # [128, 128]
        y = y.unsqueeze(0).to(self.config['device']) # [1, 128, 128]

        lambda_val = 10

        step_interval = self.config['noise_steps'] // self.config['num_steps']  # Calculate the interval for saving images

        # TODO: ensure dimensions are correct
        # TODO: Look into element-wise multiplication
        # TODO: Look into .detach etc..

        for step in reversed(range(1, self.config['noise_steps'])):
            self.unet.eval()
            with torch.no_grad():
                t = torch.tensor([step]).to(self.config['device'])
                one_minus_alpha_hat = 1.0 - self.alpha_hat[t][:, None, None, None]
                
                predicted_noise = self.unet(x_t, one_minus_alpha_hat) if self.unet.requires_alpha_hat_timestep else self.unet(noise_image, t)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x_t) if step < self.config['noise_steps'] - 1 else torch.zeros_like(noise_image)
                noise = noise.to(self.config['device'])

                mu = 1 / torch.sqrt(alpha) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise)
                
                if step % 2 == 0:
                    # Apply LCG
                    mu_hat = mu
                    #mu_hat = self.apply_lcg(x_t, mu, y, lambda_val) # x_t is the same as x_t_minus_1(updated in-place)
                elif step % 2 == 1:
                    # Apply GSG
                    mu_hat, decoded_output = self.apply_gsg(x_t, mu, y, lambda_val) # x_t is the same as x_t_minus_1 (updated in-place)

                x_t = mu_hat + torch.sqrt(beta) * noise # sampling x_t from adjusted mu with added noise

                # Save and visualize every 100th step
                if step % step_interval == 0 or step == 1:
                    denoised_images.append(x_t.squeeze(0).cpu().detach())
                    decoded_outputs.append(decoded_output)

        return denoised_images, decoded_outputs
    
    def apply_gsg(self, x_t, mu, y, lambda_val):
        # Adjust mu based on global guidance
        # L(global)[x, y] = L(ce)[g(x), y], 
        # mu_hat(x,k) = mu(x,k) + lambda * gradient(L(global)[x,y])

        
        
        with torch.set_grad_enabled(True):
            #self.seg_model.model.train()
            x_t = x_t.clone().detach().requires_grad_(True) # enable grad computation
            #print(x_t.requires_grad)
            
            predicted_mask = self.seg_model.predict(x_t) # pred = [1, 20, 128, 128]

            decoded_output=self.seg_model.decode_pred_mask(predicted_mask)

            
            if not predicted_mask.requires_grad:
                raise RuntimeError("Model output does not require gradients, check model configuration.")

            
            #loss = F.cross_entropy(predicted_mask, y.long()) # y = [1, 128, 128]
            loss = self.seg_model.model.criterion(predicted_mask, y.long())
            #print("dice loss:", loss.item())

            # If loss is zero (no overlap), we cannot backpropagate
            if loss.item() == 0:
                raise ValueError('Dice loss is zero, gradient cannot be computed.')
            if not loss.requires_grad:
                raise RuntimeError("Loss does not require gradients, ensure inputs and model are correctly configured.")


            # NOTE: there might be uncomputed gradients, if there is no overlap between the predicted mask and the ground truth mask
            self.seg_model.model.zero_grad()
            
            try:
                loss.backward()
            except Exception as e:
                print("Error during backward pass:", e)
            
            grad = x_t.grad
            if grad is None:
                raise RuntimeError('No gradients found for x_t.')

            mu_hat = mu + lambda_val * grad # * cov ?
                #print("mu_hat shape: ", mu_hat.shape) 
        x_t.requires_grad_(False)  # Turn off gradients for x_t
        mu_hat = mu_hat.detach()
        #self.seg_model.model.eval()

        return mu_hat, decoded_output
    
    def apply_lcg(self, x_t, mu, y, lambda_val, beta, noise):
        x_t_minus_1_c_hat = []
        num_classes = 20
        mc = []

        for c in range(num_classes-1):
            # generate mask for class c using segmentor
            mask = self.get_class_mask(x_t, self.seg_model, c)
            mc.append(mask)
            
            x_t.requires_grad = True

            # L(local)[x_t,y,c] = L(ce)[g(x_t ** mc),y ** mc], 
            # Calculate the cross-entropy loss for the class-specific region
            loss = self.calculate_class_specific_loss(x_t, y, mask, self.seg_model)

            # Compute the gradient of the loss with respect to x_t.
            loss.backward()
        
            # Adjust the mean
            with torch.no_grad():
                grad = x_t.grad
                # mu_hat(x_t,k,c) = mu(x_t,k) + lambda * gradient(L(local)[x,y,c])
                mu_hat_c = mu - lambda_val * grad  # Adjust μ_{t-1} using the gradient
            
            x_t.requires_grad_(False)  # Turn off gradients for x_t

            # Sampling x_{t-1} from adjusted μ with added noise
            x_t_minus_1_c = mu_hat_c.detach()  + torch.sqrt(beta) * noise
            x_t_minus_1_c_hat.append(x_t_minus_1_c)
            
        x_t_minus_1 = sum(x_t_minus_1_c * mc) # NOTE element-wise multiplication
        return x_t_minus_1
    
    def get_class_mask(self, image, segmentor, class_id):
        """
        Generate a binary mask for the specified class_id using the segmentation model.
        
        :param image: Input image tensor.
        :param segmentor: Segmentation model.
        :param class_id: Target class ID for which the mask is generated.
        :return: Binary mask for the specified class.
        """
        with torch.no_grad():
            output = segmentor(image)  # Assuming output is [N, C, H, W]
            mask = torch.argmax(output, dim=1) == class_id  # [N, H, W]
            mask = mask.float()  # Convert to float for multiplication
        return mask.unsqueeze(1)  # Add channel dimension back [N, 1, H, W]

    def calculate_class_specific_loss(self, x_t, y, mask, segmentor):
        """
        Calculate the cross-entropy loss for the specified class region.
        
        :paramd x_t: Input image tensor at timestep t.
        :param y: Target labels tensor.
        :param mask: Binary mask for the class-specific region.
        :param segmentor: Segmentation model.
        :return: Cross-entropy loss for the class-specific region.
        """

        # apply mask to x_t to focus on the class-specific region
        masked_input = x_t * mask # NOTE : element-wise multiplication 
        with torch.no_grad():
            predictions = segmentor(masked_input)
        masked_labels = y * mask # NOTE : element-wise multiplication  

        loss = F.cross_entropy(predictions, masked_labels, reduction='none')
        loss = (loss * mask).mean()  # Only consider loss where mask is 1
        return loss



    def visualize(self, images, title, denorm=True):
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
        for i, image in enumerate(images):
            ax = axes[i]
            if denorm:
                image = self.denorm(image)
                ax.imshow(image.permute(1, 2, 0))  # Adjust for PyTorch channel order
            else:
                ax.imshow(image)
            ax.axis('off')
        plt.suptitle(title)
        plt.show()

    def visualize_translation(self, source_image, target_image):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        source_image = self.denorm(source_image)
        axes[0].imshow(source_image.permute(1, 2, 0))
        axes[0].set_title('Source Image')
        axes[0].axis('off')
        target_image = self.denorm(target_image)
        print(target_image.shape)
        axes[1].imshow(target_image.permute(1, 2, 0))
        axes[1].set_title('Target Image')
        axes[1].axis('off')
        plt.show()

    

if __name__ == '__main__':

    cfg = {
        'c_in': 3,
        'c_out': 3,
        'image_size': 128,
        'conv_dim': 64,
        'block_depth': 3,
        'time_emb_dim': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'noise_steps': 1000, # controlls how many steps
        'num_steps': 10, # controlls how many steps to visualize
        'seg_backbone': 'resnet101'
    }

    model_path = 'outputs/checkpoints/run_12/500-checkpoint.ckpt'
    seg_model_path = 'model-resnet101.pth'

    test_dataset = MyCityscapesDataset('data/cityscapes', split='val', mode='fine',
                        target_type='semantic',transform=transform, target_transform=transform_mask)
    test_loader=DataLoader(test_dataset, batch_size=12, shuffle=False)

    sample = 11
    image, mask = next(iter(test_loader))
    #print(image.shape, mask.shape)
    
    translator = DiffusionTranslationSGG(cfg=cfg, 
                                         test_image=image[sample], 
                                         test_mask=mask[sample], 
                                         model_checkpoint_path=model_path, 
                                         seg_model_path=seg_model_path
                                         )

    noisy_images = translator.add_noise_for_steps() 
    #print("full noise image: ", noisy_images[-1].shape)
    #translator.visualize(noisy_images, 'Noisy Images', True)
    denoised_images, decoded_outputs = translator.remove_noise_for_steps(noisy_images[-1])
    #print("denoised image: ", denoised_images[-1].shape)
    translator.visualize(denoised_images, 'Denoised Images', True)
    
    

    translator.visualize(decoded_outputs, 'Mask Images', False)

    translator.visualize_translation(noisy_images[0], denoised_images[-1])
