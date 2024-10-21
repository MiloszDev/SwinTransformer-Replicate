import torch
from torch import nn
from torchvision import transforms
from utils import set_seeds
from data_setup import setup_data
from model_builder import SwinTransformer
from engine import train

def remove_alpha_channel(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image

# Define data transformations including the alpha channel removal
transform = transforms.Compose([
    transforms.Lambda(lambda image: remove_alpha_channel(image)),  # Remove alpha channel if present
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for 3 channels
])

train_dataloader, test_dataloader, class_names = setup_data(transform=transform,
                                                            targ_dir='./data/images/Images',
                                                            batch_size=64)

swin_transformer = SwinTransformer(num_classes= len(class_names),
                                   window_size= 7,
                                   num_heads= 8,
                                   embedding_dim= 128,
                                   ffn_dim= 512,
                                   dropout_rate= 0.1,
                                   in_channels= 3,
                                   patch_size= 4,
                                   depths= [2, 2, 6, 2])
print(swin_transformer)
# Setup the optimizer to optimizer ViT model parameters using hyperparameters from the ViT paper
optimizer = torch.optim.Adam(params=swin_transformer.parameters(),
                             lr=3e-3, # Base LR from Table 3 for ViT
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-Tuning)
                             weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-Tuning)

# Setup the loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Set the seeds
set_seeds()

# Train the model and save the training results to a dictionary
results = train(model=swin_transformer,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=criterion,
                epochs=10,
                device='cpu')