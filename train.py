from torchvision import transforms
from data_setup import setup_data
from model_builder import SwinTransformer

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


