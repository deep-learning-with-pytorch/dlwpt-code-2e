"""
Various graphing and plotting utilities.
"""
import os
import requests
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
import json
import torch.nn.functional as F

def get_sam_model(size="large"):
    """
    Three sizes to choose from in https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
    from largest to smallest:
    "huge": https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    "large": https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    "base": https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    """

    # Model weights
    if size == "huge":
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        model_weights_path  = "p2ch15/sam_vit_h_4b8939.pth"
        model_config = "vit_h"
    elif size == "large":
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        model_weights_path  = "p2ch15/sam_vit_l_0b3195.pth"
        model_config = "vit_l"
    elif size == "base":
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        model_weights_path  = "p2ch15/sam_vit_b_01ec64.pth"
        model_config = "vit_b"
    else:
        raise ValueError(f"Invalid size: {size}")
    
    # Download the file if it doesn't exist locally
    if not os.path.exists(model_weights_path ):
        print(f"Downloading {model_weights_path }...")
        
        # Stream the download to handle large files
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        # Use tqdm to display the progress bar
        with open(model_weights_path , 'wb') as f, tqdm(
            desc=model_weights_path ,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
        
        print("Download complete.")
    return model_config, model_weights_path 

def plot_mask(mask):
    """
    Plots a boolean mask as a black and white image.
    
    Parameters:
    mask (numpy.ndarray or torch.Tensor): A 2D array or tensor.
    """
    # Check if the input is a PyTorch tensor
    if isinstance(mask, torch.Tensor):
        # Move to CPU if necessary and convert to NumPy array
        if mask.is_cuda:
            mask = mask.cpu()
        mask = mask.numpy()
    
    # Check if the input is a NumPy array
    if not isinstance(mask, np.ndarray):
        raise ValueError("Input must be a numpy array or a torch tensor")
    
    # Convert the boolean mask to an integer array (True -> 1, False -> 0)
    int_mask = mask.astype(int)
    
    # Plot the mask
    plt.imshow(int_mask, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()

def plot_original_image_with_masks(image, masks):
    """
    Plots the image with masks overlaid in different colors.
    Args:
        image (PIL.Image): The original image.
        masks (list of np.ndarray): List of boolean masks.
    """
    # Convert image to numpy array
    image_np = np.array(image)
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image_np, cmap='gray')
    # Generate a colormap
    cmap = plt.get_cmap('hsv')
    num_colors = len(masks)
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    # Plot each mask
    for i, mask in enumerate(masks):
        # Find contours of the mask
        plt.contourf(mask, levels=[0.5, 1], colors=[colors[i]], alpha=0.4, linewidths=1, hatches=['//'])
    plt.axis('off')
    plt.show()

def plot_tensor_values(tensor, title='Tensor as Image', cmap='gray'):
    """
    Plots a 2D slice of a tensor using Matplotlib.
    Parameters:
    - tensor: A PyTorch tensor of shape (N, C, H, W) or (C, H, W).
    - title: Title of the plot.
    - cmap: Colormap to use for plotting (default is 'gray').
    """
    # Check if the tensor has a batch dimension
    if tensor.dim() == 4:
        # Select the first image in the batch and the first channel
        image = tensor[0, 0, :, :]
    elif tensor.dim() == 3:
        # Select the first channel
        image = tensor[0, :, :]
    else:
        raise ValueError("Tensor must have 3 or 4 dimensions (C, H, W) or (N, C, H, W).")
    # Convert the tensor to a NumPy array
    image_np = image.numpy()
    # Plot the image
    plt.imshow(image_np, cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Turn off axis labels
    plt.show()

def plot_tensor_histogram(tensor, bins=30, title='Tensor Value Distribution'):
    """
    Plots a histogram of the values in a tensor.
    Parameters:
    - tensor: A PyTorch tensor of any shape.
    - bins: Number of bins to use in the histogram (default is 30).
    - title: Title of the plot.
    """
    # Flatten the tensor to a 1D array
    tensor_flat = tensor.flatten().numpy()
    # Plot the histogram
    plt.hist(tensor_flat, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

fine_tuning_dir = "data-unversioned/part2/fine-tuning/dataset"
ct_folder = f"{fine_tuning_dir}/ct"
mask_folder = f"{fine_tuning_dir}/mask"
metadata_folder = f"{fine_tuning_dir}/"

class FineTuningDataset(Dataset):
    def __init__(
        self,
        fine_tuning_dir=fine_tuning_dir,
        ct_folder=ct_folder,
        mask_folder=mask_folder,
        metadata_folder=metadata_folder,
        split="train",  # 'train' or 'val'
        train_ratio=0.8,  # Ratio of data to use for training
    ):
        self.fine_tuning_dir = fine_tuning_dir
        self.metadata_folder = metadata_folder

        # Create folders if they don't exist
        if (
            not os.path.exists(ct_folder)
            or not os.path.exists(mask_folder)
            or not os.path.exists(metadata_folder)
        ):
            raise RuntimeError("'generate_ct_images_and_masks()' must be called first")

        metadata_filepath = os.path.join(metadata_folder, "metadata.jsonl")
        self.metadata_list = []
        # Read the JSONL file
        with open(metadata_filepath, "r") as f:
            for line in f:
                # Parse each line as a JSON object
                row = json.loads(line.strip())
                # Check if the row is not empty
                if row:  # This checks if the row is not an empty dictionary
                    self.metadata_list.append(row)

        # Determine the split indices
        total_length = len(self.metadata_list)
        train_length = int(total_length * train_ratio)

        if split == "train":
            self.metadata_list = self.metadata_list[:train_length]
        elif split == "val":
            self.metadata_list = self.metadata_list[train_length:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        metadata = self.metadata_list[index]
        ct_image_path = f"{self.fine_tuning_dir}/{metadata['ct_file_name']}"
        mask_image_path = f"{self.fine_tuning_dir}/{metadata['mask_file_name']}"
        series_uid = metadata["series_uid"]
        center_irc = metadata["center_irc"]

        return {
            "series_uid": series_uid,
            "center_irc": torch.tensor(center_irc),
            "ct_image_path": ct_image_path,
            "mask_image_path": mask_image_path,
        }

def plot_image_and_masks(image, predicted_mask, ground_truth_mask):
    # Convert masks to numpy arrays if they're not already
    if isinstance(predicted_mask, Image.Image):
        predicted_mask = np.array(predicted_mask)
    if isinstance(ground_truth_mask, Image.Image):
        ground_truth_mask = np.array(ground_truth_mask)
    
    # Ensure the masks are binary (0s and 1s)
    predicted_mask = predicted_mask.astype(np.uint8)
    ground_truth_mask = ground_truth_mask.astype(np.uint8)
    
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot the image
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Plot the predicted mask
    axes[1].imshow(predicted_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot the ground truth mask
    axes[2].imshow(ground_truth_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # Show the plots
    plt.tight_layout()
    plt.show()