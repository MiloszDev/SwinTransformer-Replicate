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

class ImageFolderCustom(Dataset):

    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(Path('./data/images/Images').glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx, self.breed_names = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
        
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

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    image_dir = Path(directory)
    image_paths = list(image_dir.glob("*/*.jpg"))

    breed_names = list(set(str(path.parent.name) for path in image_paths))

    return classes, class_to_idx, breed_names

def split_data(dataset: torch.utils.data.Dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def setup_data(transform: torchvision.transforms,
               targ_dir: str = './data/images/Images',
               batch_size: int = 64):
    
    dataset = ImageFolderCustom(targ_dir=targ_dir, 
                                transform=transform)
    
    class_names = dataset.breed_names
    
    train_dataset, test_dataset = split_data(dataset=dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    
    return train_dataloader, test_dataloader, class_names
