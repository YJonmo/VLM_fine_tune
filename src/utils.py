from dataclasses import dataclass
import random
from pathlib import Path
from typing import Union, Dict, Tuple, List, Callable
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from transformers import CLIPProcessor
from matplotlib import pyplot as plt

class PneumoniaDict(BaseModel):
    """
    Pydantic model for pneumonia data dictionary.
    """
    labels: Dict[str, int] = {'NORMAL':[0], 'PNEUMONIA':[1]}
    ids: Dict[int, str] = [0, 1]
    prompts: Dict[str, List[str]] = {
                "PNEUMONIA": ["chest X-ray with pneumonia", "lung infection visible", "subject is not healthy", "the X-ray is not normal", "there are infections visibale", "pneumonia is apparent", "subject suffers from pneumonia"],
                "NORMAL": ["chest X-ray without pneumonia", "no lung infection visible", "healthy chest X-ray",    "normal lung scan",        "no infection is found",       "no sign of pneumonia", "the X-ray is normal"]
            }
    
@dataclass
class CLIPOutput:
    """
    Pydantic model for CLIP output.
    """
    logits_per_image: torch.Tensor
    logits_per_text: torch.Tensor
    loss: torch.Tensor
    text_feature: torch.Tensor
    image_feature: torch.Tensor

@dataclass
class CLIPClassifierOutput:
    """
    Pydantic model for CLIP output.
    """
    pred: torch.Tensor
    loss: torch.Tensor
    

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Compute the contrastive loss for CLIP model.
    Args:
        similarity (torch.Tensor): Similarity matrix between image and text features.
    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def image_preproc()-> Dict:
    """
    Get preprocessing transformations for image data.
    
    Returns:
        Dict: A dictionary containing the preprocessing transformations pipeline including:
              - Resize to 224x224
              - Convert to tensor 
              - Normalize with ImageNet mean/std values
    """

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #    )
    ])
    return preprocess

def image_aug()->Dict:
    """
    Get data augmentation transformations for training and testing.
    
    Returns:
        Dict: A dictionary containing two transform pipelines:
              - 'train': Applies random horizontal/vertical flips, rotation and crop
              - 'test': Empty transform pipeline for test data
    """
    transformations = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ]),
        'test': transforms.Compose([
        ])
    }
    return transformations

def get_device()->torch.device:
    """
    Get the available computing device (GPU or CPU).
    
    Returns:
        torch.device: Returns 'cuda' if GPU is available, otherwise returns 'cpu'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device



class XRayTextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 rooth_path:Union[Path, str],
                 data_path:Union[Path, str],
                 preprocess_image:Union[Dict,None]=None,
                 transformations:Union[Dict,None]=None,
                 sub_folders:Tuple[str,...]=('NORMAL', 'PNEUMONIA'),
                 extent:str='jpeg',
                 text_template:Dict=None,
                 seed:int=0):
        
        data_path = Path(rooth_path) / Path(data_path)
        self.transformations = transformations        
        self.preprocess_image = preprocess_image
        self.text_template = text_template
        self.data = [[],[]]
        sums = [0, 0] # NORMAL, PNEUMONIA
        for sub_folder in sub_folders:
            files = list(data_path.glob(f'{sub_folder}/*.{extent}'))
            self.data[0].extend(files)    
            if sub_folder == 'NORMAL':
                self.data[1].extend(['NORMAL']*len(files))
                sums[0] += len(files)
            else:
                self.data[1].extend(['PNEUMONIA']*len(files))
                sums[1] += len(files)

        assert len(self.data[0]) > 0, "Data must not be empty"
        assert len(self.data[1]) > 0, "Labels must not be empty"
        self.data = list(zip(self.data[0], self.data[1]))
        self.data = sorted(self.data, key=lambda x: x[1]) # Normal smalpes are set to be first
        weights = (1/sums[0], 1/sums[1]) 
        self.weights = (sums[0]*[weights[0]], sums[1]*[weights[1]]) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int)->List:
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB') # use 'RGB' for grey scale images
        if self.transformations:
            img = self.transformations(img)
        img = self.preprocess_image(img)
        text = random.choice(self.text_template[label])
        return img, text


class PneumoniaDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and preprocessing chest X-ray images for pneumonia detection.

    Args:
        rooth_path (Union[Path, str]): Root directory path containing the dataset
        data_path (Union[Path, str]): Path to the data directory relative to root_path
        preprocess (Union[Dict, None]): Preprocessing transformations to apply to images
        transformations (Union[Dict, None]): Data augmentation transformations to apply
        sub_folders (Tuple[str,...]): Tuple of subfolder names containing the image classes. Default: ('NORMAL', 'PNEUMONIA')
        extent (str): File extension of the images. Default: 'jpeg'
        reorder (bool): Whether to randomly shuffle the dataset. Default: True
        seed (int): Random seed for shuffling. Default: 0

    The dataset expects a directory structure like:
    root_path/data_path/NORMAL/*.jpeg
    root_path/data_path/PNEUMONIA/*.jpeg
    """
    def __init__(self,
                 rooth_path:Union[Path, str],
                 data_path:Union[Path, str],
                 preprocess:Union[Dict,None]=None,
                 transformations:Union[Dict,None]=None,
                 sub_folders:Tuple[str,...]=('NORMAL', 'PNEUMONIA'),
                 extent:str='jpeg',
                 reorder:bool=True,
                 seed:int=0):
        
        data_path = Path(rooth_path) / Path(data_path)
        self.transformations = transformations
        self.preprocess = preprocess
        self.data = [[],[]]

        for sub_folder in sub_folders:
            self.data[0].extend(list(data_path.glob(f'{sub_folder}/*.{extent}')))    
            if sub_folder == 'NORMAL':
                self.data[1].extend([0]*len(self.data[0]))
            else:
                self.data[1].extend([1]*len(self.data[0]))
        assert len(self.data[0]) > 0, "Data must not be empty"
        assert len(self.data[1]) > 0, "Labels must not be empty"
        self.data = list(zip(self.data[0], self.data[1]))
        if reorder:
            random.Random(seed).shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int)->List:
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB') # use 'RGB' for grey scale images
        if self.preprocess:
            img = self.preprocess(img)
        if self.transformations:
            img = self.transformations(img)
        return [img, label]


def similiarity_plot(similarity:torch.Tensor, prompts:List[str], 
                     images:List[torch.Tensor], image_names:List[str]):
    """
    Plot the similarity matrix between prompts and images.
    Args:
        similarity (torch.Tensor): Similarity matrix between prompts and images.
        prompts (List[str]): List of prompts.
        images (List[torch.Tensor]): List of image tensors.
        image_names (List[str]): List of image names.
    """
    similarity = similarity.cpu().numpy()
    count = len(prompts)
    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(left=0.22)
    plt.imshow(similarity, vmin=0.0, vmax=similarity.max(), origin="upper")
    plt.yticks(range(count), prompts, fontsize=10)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", color=[1,1,1], size=8)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xticks(range(similarity.shape[1]), image_names, rotation=90, va='top', ha='right', y=0.075)
    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    plt.show()



def train_loop(model: torch.nn.Module, train_loader:DataLoader, 
               val_loader:DataLoader, optimizer:torch.optim.Optimizer,
               num_epoch:int, processor:CLIPProcessor, device:torch.device)->torch.nn.Module:
    """"
    Training loop function that trains a model for specified number of epochs
    Args:
       model (torch.nn.Module): Neural network model to train
       train_loader (DataLoader): DataLoader for training data
       val_loader (DataLoader): DataLoader for validation data  
       optimizer (torch.optim.Optimizer): Optimizer for updating model weights
       num_epoch (int): Number of epochs to train for
    """
    model = model.to(device)
    for epoch in range(num_epoch):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epoch}', leave=False)
        sum_loss = 0.0
        for i, [images, labels] in enumerate(progress_bar):
            if isinstance(labels[0], torch.Tensor):
                prompts = None
            else:
                prompts = labels
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k,v in inputs.items()} # Move inputs to the device

            optimizer.zero_grad()
            outputs= model(inputs, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            av_loss = sum_loss/(i+1)
            progress_bar.set_postfix({'Loss': f'{av_loss:.4f}'})
        train_loss = sum_loss/(i+1)

        model.eval()
        sum_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epoch}', leave=False)
        with torch.no_grad():
            sum_loss = 0.0
            for i, [images, labels] in enumerate(val_progress_bar):
                if isinstance(labels[0], torch.Tensor):
                    prompts = None
                else:
                    prompts = labels
                inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k,v in inputs.items()} # Move inputs to the device

                outputs= model(inputs, labels)
                loss = outputs.loss
                sum_loss += loss.item()
                av_loss = sum_loss/(i+1)
                val_progress_bar.set_postfix({'Loss': f'{av_loss:.4f}'})
            tqdm.write(f"Epoch: {epoch+1}/{num_epoch}, Average losses for train: {train_loss:.4f}, validation: {av_loss:0.4f}")
    return model