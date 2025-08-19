# These functions are defined in the notebooks for Chapter 10 of the book.
# We are copying them here for use across multiple notebooks.

import os
from PIL import Image, ImageFilter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# START forward_diffusion.ipynb
def points_from_img(img_file, min_distance=5):
    """
    Returns a list of (x, y) tuples of the pixel coordinates 
    for the outline of the image, ensuring that no two points are too close to each other.
    
    :param img_file: Path to the image file.
    :param min_distance: Minimum distance between consecutive points.
    """
    # Open the image file
    img = Image.open(img_file).convert("L")  # Convert to grayscale
    # Apply edge detection filter
    img = img.filter(ImageFilter.FIND_EDGES)
    
    # Initialize a list to store the coordinates
    points = []
    last_point = None  # Track the last point added
    
    # Iterate over each pixel
    for y in range(img.height):
        for x in range(img.width):
            # Get the value of the pixel
            pixel = img.getpixel((x, y))
            # If the pixel is not black, it is part of the outline
            if pixel != 0:
                current_point = (x, y)
                # Check if it's the first point or if it's sufficiently far from the last added point
                if last_point is None or np.linalg.norm(np.array(current_point) - np.array(last_point)) >= min_distance:
                    points.append(current_point)
                    last_point = current_point  # Update the last point added
    # Normalize the points
    points = np.array(points)
    mean = np.mean(points, axis=0)
    normalized_points = points - mean
    
    return normalized_points.tolist()

def plot_points(all_points, plot_titles=None, highlight_index=None):
    """
    Plots multiple sets of points side by side using seaborn.
    Each set of points should be a 2D numpy array where each row is [x, y].
    An optional list of highlight_indices can be passed to color specific points differently in each set.
    
    :param points_list: List of 2D numpy arrays of points.
    :param plot_titles: List of titles for each subplot.
    :param highlight_indices: List of indices of the points to highlight in each set.
    """
    # If only a single point, turn it into a list
    if not isinstance(all_points, list):
        points_list = [all_points]
    else:
        points_list = all_points
    # Number of plots
    num_plots = len(points_list)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # If there's only one plot, wrap axes in a list
    if num_plots == 1:
        axes = [axes]
    
    # Loop through each set of points and corresponding axis
    for idx, (points, ax) in enumerate(zip(points_list, axes)):
        # Ensure points is a numpy array
        points = np.array(points.cpu())
        
        # Extract x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        # Create a hue array, default hue is 'Normal'
        hues = ['Normal'] * len(points)
        
        # If highlight_indices is provided and valid, change the hue of the specified point
        if highlight_index and highlight_index < len(points):
            hues[highlight_index] = 'Highlight'  # Highlight category
        
        # Define a custom palette
        palette = {'Normal': 'blue', 'Highlight': 'red'}
        
        # Create a scatter plot on the specified axis
        sns.scatterplot(x=x, y=y, hue=hues, palette=palette, legend=None, ax=ax)
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        
        # Set title for each subplot
        if plot_titles and idx < len(plot_titles):
            ax.set_title(plot_titles[idx])
        else:
            ax.set_title(f"Plot {idx + 1}")  # Default title if no custom title provided
    
    plt.tight_layout()
    plt.show()

def reshape_for_x(a, x):
    ones_to_broadcast = len(x.shape) - 1
    return a.view(-1, *[1] * ones_to_broadcast).to(x.device)

def forward_diffusion_sample(x, t):
    x = x.to(device)
    noise = torch.randn_like(x)
    sqrt_alphas_cumprod_t = reshape_for_x(sqrt_alphas_cumprod[t], x)
    sqrt_one_minus_alphas_cumprod_t =  reshape_for_x(sqrt_one_minus_alphas_cumprod[t], x)
    # mean + variance
    return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
# END forward_diffusion.ipynb

# START training.ipynb
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.tensor([10000.0], device=device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size, device=device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    
def get_loss(model, x, t):
    x_noisy, noise = forward_diffusion_sample(x, t)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(model, x, t):
    betas_t = reshape_for_x(betas[t], x)
    sqrt_one_minus_alphas_cumprod_t = reshape_for_x(sqrt_one_minus_alphas_cumprod[t], x)
    sqrt_recip_alphas_t = reshape_for_x(sqrt_recip_alphas[t], x)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = reshape_for_x(posterior_variance[t], x)
    
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
# END training.ipynb

# Define global variables
#######################
device = "cuda" if torch.cuda.is_available() else "cpu"
points = points_from_img("../data/p2ch10/pytorch_logo.png")
x0 = torch.tensor(points, dtype=torch.float32).to(device)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

with torch.device(device):
    T = 1000
    betas = linear_beta_schedule(T)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)