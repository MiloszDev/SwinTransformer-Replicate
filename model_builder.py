import torch
from torch import nn

# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
  """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
  """

  # 2. Initialize the class with appropriate variables
  def __init__(self,
               in_channels: int = 3,
               patch_size: int = 4,
               embedding_dim: int = 128):
    super().__init__()

    self.patch_size = patch_size
    # 3. Create a layer to turn an image into patches
    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)
    # 4. Create a layer to flatten the patch feature maps into a single dimension
    self.flatten = nn.Flatten(start_dim=2, # Only flatten the feature map dimensions into a single vector
                              end_dim=-1)

  def forward(self, x):
    # Create assertion to check that inputs are the correct shape
    image_resolution = x.shape[-1]
    assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}"

    # Perform the forward pass
    x_patched = self.patcher(x)
    x_flattened = self.flatten(x_patched)

    # 6. make sure the output has the right order
    return x_flattened.permute(0, 2, 1) # adjusting so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowbasedSelfAttention(nn.Module):
  def __init__(self,
               window_size: int = 7,
               embedding_dim: int = 128,
               num_heads: int = 8) -> None:
    super().__init__()

    assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by heads."

    self.window_size = window_size
    self.head_dim = embedding_dim // num_heads
    self.num_heads = num_heads
    self.scale = self.head_dim ** -0.5

    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
    self.out = nn.Linear(embedding_dim, embedding_dim)

  def divide_windows(self, x : torch.Tensor) -> torch.Tensor:
    batch_size, num_patches, color_channels = x.shape

    height = int(num_patches ** 0.5)
    width = height

    assert height % self.window_size == 0 and width % self.window_size == 0, "Image must be divisible by window_size"

    windows = x.view(batch_size, height // self.window_size, self.window_size, width // self.window_size, self.window_size, color_channels).permute(0, 1, 3, 2, 4, 5).contiguous()

    return windows.view(-1, self.window_size, self.window_size, color_channels)

  def merge_windows(self, windows: torch.Tensor, height: int, width: int, batch_size: int) -> torch.Tensor:
    windows = windows.view(batch_size, height // self.window_size, width // self.window_size, self.window_size, self.window_size, -1)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(batch_size, height, width, -1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      # Ensure that x is of shape [batch_size, num_patches, embedding_dim]
      x = self.layer_norm(x)
      batch_size, num_patches, embedding_dim = x.shape  # Should match the expected 3D shape

      height = int(num_patches ** 0.5)
      width = height

      windows = self.divide_windows(x).view(-1, self.window_size**2, embedding_dim)

      QKV = self.qkv(windows).view(-1, self.window_size**2, 3, self.num_heads, self.head_dim)
      Q, K, V = QKV.permute(2, 0, 3, 1, 4).unbind(0)

      attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
      attention_probs = F.softmax(attention_scores, dim=-1)
      attention_output = torch.matmul(attention_probs, V).permute(0, 2, 1, 3).contiguous()

      attention_output = attention_output.view(-1, self.window_size**2, embedding_dim)
      attention_output = self.out(attention_output)

      # Merging the windows back into their original shape
      return self.merge_windows(attention_output.view(-1, self.window_size, self.window_size, embedding_dim), height, width, batch_size)

class FeedForwardNetwork(nn.Module):
  def __init__(self,
               embedding_dim: int = 128,
               ffn_dim: int = 512,
               dropout_rate: float = 0.1) -> None:
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    self.ffn = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=ffn_dim),
        nn.GELU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features=ffn_dim,
                  out_features=embedding_dim),
        nn.Dropout(p=dropout_rate))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.layer_norm(x)

    output = self.ffn(x)
    return output

class SwinTransformerBlock(nn.Module):
    def __init__(self, window_size: int = 7, num_heads: int = 8, embedding_dim: int = 128, ffn_dim: int = 512, dropout_rate: float = 0.1) -> None:
        super().__init__()

        self.wmsa = WindowbasedSelfAttention(window_size=window_size, embedding_dim=embedding_dim, num_heads=num_heads)
        self.fnn = FeedForwardNetwork(embedding_dim=embedding_dim, ffn_dim=ffn_dim, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_connection = x

        # Forward through Window-based Multi-head Self-Attention (WMSA)
        x = self.wmsa(x)

        # Adding residual and reshaping it back to [batch_size, num_patches, embedding_dim] format
        x = x.view_as(residual_connection) + residual_connection

        # Forward through Feed-Forward Network (FFN)
        x = self.fnn(x) + x

        return x

class SwinTransformer(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 window_size: int = 7,
                 num_heads: int = 8,
                 embedding_dim: int = 128,
                 ffn_dim: int = 512,
                 dropout_rate: float = 0.1,
                 in_channels: int = 3,
                 patch_size: int = 4,
                 depths: list = [2, 2, 6, 2]) -> None:
        super().__init__()

        self.patchify = PatchEmbedding(in_channels=in_channels,
                                       patch_size=patch_size,
                                       embedding_dim=embedding_dim)
        self.stages = nn.ModuleList()

        for depth in depths:
            stage = nn.ModuleList([
                SwinTransformerBlock(window_size=window_size,
                                     num_heads=num_heads,
                                     embedding_dim=embedding_dim,
                                     ffn_dim=ffn_dim,
                                     dropout_rate=dropout_rate)
                for _ in range(depth)
            ])
            self.stages.append(stage)

            self.downsample = nn.Conv2d(in_channels=embedding_dim,
                                        out_channels=embedding_dim,
                                        kernel_size=2,
                                        stride=2)

        self.classifier = nn.Linear(in_features=embedding_dim,
                                    out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Patchify the input image
        x = self.patchify(x)  # [batch_size, num_patches, embedding_dim]

        # Track number of patches and embedding dimension
        batch_size, num_patches, embedding_dim = x.shape
        side_length = int(num_patches ** 0.5)

        for stage in self.stages:
            for block in stage:
                x = block(x)

            # Reshape x for downsampling
            x = x.permute(0, 2, 1).view(batch_size, embedding_dim, side_length, side_length)  # Reshape into 2D
            x = self.downsample(x)  # Apply downsampling

            # Update the new side length after downsampling (half the original)
            side_length = side_length // 2

            # Flatten x back into the 3D tensor [batch_size, num_patches, embedding_dim]
            num_patches = side_length * side_length
            x = x.flatten(2).permute(0, 2, 1)

        # Step 2: Global Average Pooling
        x = x.permute(0, 2, 1).view(batch_size, embedding_dim, side_length, side_length)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)  # Pool and flatten

        # Step 3: Classification
        x = self.classifier(x)

        return x
