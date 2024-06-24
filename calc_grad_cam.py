import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
import cv2

class GradCAM:
    def __init__(self, model, layer_name='center'):
        self.model = model
        self.feature_maps = []
        self.gradients = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Register hook to capture feature maps and gradients
        self.hooks = []

        # Ensure the layer exists in the model
        if hasattr(self.model, layer_name) or (isinstance(self.model, nn.DataParallel) and hasattr(self.model.module, layer_name)):
            target_layer = getattr(self.model, layer_name) if hasattr(self.model, layer_name) else getattr(self.model.module, layer_name)

            if isinstance(self.model, nn.DataParallel):
                self.hooks.append(target_layer.register_forward_hook(self.save_feature_maps))
                self.hooks.append(target_layer.register_backward_hook(self.save_gradients))
            else:
                self.hooks.append(target_layer.register_forward_hook(self.save_feature_maps))
                self.hooks.append(target_layer.register_backward_hook(self.save_gradients))
        else:
            raise ValueError(f"'{layer_name}' layer not found in the model")

    def save_feature_maps(self, module, input, output):
        # Save feature maps
        self.activations = output.to(self.device)

    def save_gradients(self, module, grad_input, grad_output):
        # Save gradients corresponding to the feature maps
        self.gradients.append(grad_output[0].to(self.device))
    
    def get_bounding_box(self, heatmap, threshold=None, window_size=None):
        """
        Return the bounding box coordinates (x, y, width, height) for the area with the maximum activation.
        Either based on a threshold or using a fixed-size sliding window.
        """
        heatmap_np = heatmap.cpu().detach().numpy()

        if threshold:
            # Get a binary mask of the activations above the threshold
            binary_mask = (heatmap_np > threshold).astype(np.uint8)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest contour area
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                return x, y, w, h

        elif window_size:
            # Use convolution to slide the window and compute sums
            kernel = torch.ones((1, 1, window_size[0], window_size[1]))
            sums = F.conv2d(heatmap, kernel, stride=1, padding=0)
            _, _, h_idx, w_idx = torch.max(sums.view(-1), dim=0)

            h_idx = h_idx.item()
            w_idx = w_idx.item()

            return w_idx, h_idx, window_size[1], window_size[0]

        return None
    

    def get_bounding_box_3d(self, heatmap, window_size=None, threshold=None):
        """
        Return the bounding box coordinates (x, y, z, depth, height, width) for the area with the maximum activation.
        Using either a fixed-size sliding window or a threshold.
        """
        if window_size:
            # Use convolution to slide the window and compute sums
            kernel = torch.ones((1, 1, window_size[0], window_size[1], window_size[2]))
            sums = F.conv3d(heatmap, kernel, stride=1, padding=0)
            _, _, d_idx, h_idx, w_idx = torch.max(sums.view(-1), dim=0)

            d_idx = d_idx.item()
            h_idx = h_idx.item()
            w_idx = w_idx.item()

            return w_idx, h_idx, d_idx, window_size[2], window_size[1], window_size[0]

        elif threshold:
            # Find the pixels above the threshold
            above_thresh = (heatmap >= threshold).nonzero(as_tuple=False)
            
            # If no values are above the threshold, return None
            if above_thresh.shape[0] == 0:
                return None
            
            # Get the min and max coordinates in each dimension
            min_coords = torch.min(above_thresh, dim=0)[0]
            max_coords = torch.max(above_thresh, dim=0)[0]

            x, y, z = min_coords[2].item(), min_coords[1].item(), min_coords[0].item()
            depth = max_coords[2].item() - x
            height = max_coords[1].item() - y
            width = max_coords[0].item() - z

            return x, y, z, depth, height, width

        return None

    def __call__(self, input_tensor, channel_idx=None, upsample_size=None):
        # Forward pass
        input_tensor = input_tensor[:1]
        model_output = self.model(input_tensor)
        
        # If no specific channel index is provided, compute heatmaps for all channels
        if channel_idx is None:
            channel_indices = list(range(model_output.shape[1]))
        else:
            channel_indices = [channel_idx]
        
        heatmaps = []
        for idx in channel_indices:
            # Zero grads
            self.model.zero_grad()
            
            # Compute the backward pass with respect to the specified channel
            one_hot_output = torch.zeros_like(model_output)
            one_hot_output[0, idx] = 1
            model_output.backward(gradient=one_hot_output, retain_graph=True)

            # Compute the Grad-CAM heatmap
            # mean_grads = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)  # Assuming 3D images
            mean_grads = torch.mean(self.gradients[-1], dim=(2, 3, 4), keepdim=True)  # Using the latest gradients
            heatmap = torch.sum(self.activations * mean_grads, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)
            
            # Upsample heatmap to input size
            if upsample_size is None:
                upsample_size = input_tensor.shape[2:]
            heatmap = F.interpolate(heatmap, size=upsample_size, mode='trilinear', align_corners=False)
            
            # Normalize the heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            heatmaps.append(heatmap)

        return heatmaps, model_output
    
    def visualize(self, iters, input_tensor, channel_idx=None, upsample_size=None, save_dir=None):
        input_tensor = input_tensor[:1]
        heatmaps, model_output = self.__call__(input_tensor, channel_idx, upsample_size)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, f'grad_cam_{iters}.png')
        fig = plt.figure(figsize=(16, 8))

        for idx, heatmap in enumerate(heatmaps):
            ax = fig.add_subplot(1, len(heatmaps) + 2, idx + 1)  
            ax.set_title(f'Channel {idx}')

            # Display the raw image
            ax.imshow(input_tensor[0, 0, 0].cpu().detach().numpy(), cmap='gray')

            # Overlay the heatmap on the image with some transparency
            heatmap_data = heatmap[0, 0, 0].cpu().detach().numpy()
            ax.imshow(heatmap_data, cmap='jet', alpha=0.5)  # `alpha` controls the transparency
            
            # Draw bounding box on the most attentive region
            bbox = self.get_bounding_box(heatmap_data, window_size=(20, 20))
            if bbox:
                x, y, w, h = bbox
                rect = plt.Rectangle((x, y), w, h, edgecolor='red', linewidth=2, fill=False)
                ax.add_patch(rect)

            ax.axis('off')
        
        # Visualize raw input image alone
        ax = fig.add_subplot(1, len(heatmaps) + 2, len(heatmaps) + 1)
        ax.set_title('Input')
        ax.imshow(input_tensor[0, 0, 0].cpu().detach().numpy(), cmap='gray')
        ax.axis('off')
        
        # Visualize model's output (assuming output is an image)
        ax = fig.add_subplot(1, len(heatmaps) + 2, len(heatmaps) + 2)
        ax.set_title('Output')
        ax.imshow(np.transpose(model_output[0, :, 0].cpu().detach().numpy(), (1, 2, 0)))
        ax.axis('off')

        fig.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)



    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()

if __name__ == '__main__':
    from model_superhuman2 import UNet_PNI
    import numpy as np
    import argparse
    import os
    import yaml
    import time
    from attrdict import AttrDict
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from main import load_dataset, build_model

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_3d_ac4_data80', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)
    config_path = os.path.join('/data/ydchen/VLP/wafer4/config', cfg_file)
    cfg = OmegaConf.load(config_path)
    # with open(config_path, 'r') as f:
    #     cfg = AttrDict(yaml.safe_load(f))
    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)
    train_provider, valid_provider = load_dataset(cfg)
    model = build_model(cfg)
    model = model.to(device)
    print('done !')
    for iters in range(10):
        input_img, label, _ = train_provider.next()
        input_img = input_img.to(device)
        label = label.to(device)
        
        # input_img = train_provider.next()[0].to(device)
        # label = train_provider.next()[1].to(device)
        grad_cam = GradCAM(model, layer_name='center')
        # attention_map = grad_cam(input_img)
        grad_cam.visualize(iters, input_img, save_dir='/data/ydchen/VLP/wafer4/grad_cam')
        output = model(input_img)
        print(output.shape)
      
    # print(attention_map.shape)
