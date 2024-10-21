"""
"""
import os
import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
from pathlib import Path
from PIL import Image
from typing import Tuple
from typing import Dict, Tuple, List
# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:

        # 3. Create class attributes
        # Get all image paths
        self.paths = list(Path('./data/images/Images').glob("*/*.jpg"))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx, self.breed_names = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in dasta_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
        
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int], List[str]]:
    """
    Finds the class folder names in a target directory and extracts breed names from the image paths.

    Args:
        directory (str): Target directory containing class folders (e.g., dog breeds).

    Returns:
        Tuple[List[str], Dict[str, int], List[str]]:
        (list_of_class_names, dict(class_name: idx...), list_of_breed_names_extracted_from_paths)

    Example:
        find_classes("/path/to/images/Images")
        >>> (["Chihuahua", "Maltese"], {"Chihuahua": 0, "Maltese": 1}, ["Chihuahua", ...])
    """

    # 1. Get the class (breed) names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if no class folders are found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a class-to-index dictionary (for numerical labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # 4. Gather all image paths and extract breed names from the directory structure
    image_dir = Path(directory)
    image_paths = list(image_dir.glob("*/*.jpg"))

    # 5. Extract breed names by splitting the folder name at the '-' character
    # Using a set to avoid duplicates
    breed_names = list(set(str(path.parent.name) for path in image_paths))

    return classes, class_to_idx, breed_names

def split_data(dataset: torch.utils.data.Dataset):
    # Set the sizes for the train/test split
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def setup_data(transform: torchvision.transforms,
               targ_dir: str = './data/images/Images',
               batch_size: int = 64):
    
    dataset = ImageFolderCustom(targ_dir=targ_dir, 
                                transform=transform)
    
    class_names = dataset.breed_names
    
    train_dataset, test_dataset = split_data(dataset=dataset)

    # Create train and test dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    
    return train_dataloader, test_dataloader, class_names