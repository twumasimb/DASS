import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
from scipy.spatial.distance import cdist
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from datasets import concatenate_datasets, DatasetDict


def compute_text_embeddings(texts: List[str], batch_size: int = 32, model: str = "all-MiniLM-L6-v2", device: str = None) -> np.ndarray:
        """
        Compute SentenceBERT embeddings for a list of texts using batching for GPU optimization.
        
        Args:
            texts: List of text documents or sentences
            batch_size: Batch size for processing
            model: SentenceTransformers model name
            device: "cuda" / "cpu" (auto-detected when None)

        Returns:
            Array of embeddings [text_count x embedding_dim]
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        all_embeddings = []
        sentence_model = SentenceTransformer(model).to(device)
        
        # Process in batches for GPU efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings with {model}"):
            batch = texts[i:i+batch_size]
            embeddings = sentence_model.encode(
                batch, 
                batch_size=batch_size,
                convert_to_numpy=True,
                device=device,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)


def compute_image_embeddings(images, model="resnet50", batch_size=32, device=None):
    """
    Compute embeddings for images using pre-trained models.
    
    Args:
        images: List of image data or paths
        model: Model name to use for embeddings. Options:
               'resnet50', 'resnet101', 'vit', 'efficientnet', 'dino'
        batch_size: Batch size for processing
        device: Device to use for computation (None for auto-detection)
        
    Returns:
        numpy.ndarray: Image embeddings of shape (n_images, embedding_dim)
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up model
    if model == "resnet50":
        model_instance = models.resnet50(weights="DEFAULT")
        embedding_dim = 2048
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    elif model == "resnet101":
        model_instance = models.resnet101(weights="DEFAULT")
        embedding_dim = 2048
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    elif model == "vit":
        model_instance = models.vit_b_16(weights="DEFAULT")
        embedding_dim = 768
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    elif model == "efficientnet":
        model_instance = models.efficientnet_b0(weights="DEFAULT")
        embedding_dim = 1280
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    elif model == "dino":
        # DINO is a self-supervised model with good transfer learning properties
        model_instance = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        embedding_dim = 384
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    else:
        raise ValueError(f"Unsupported model: {model}. Choose from 'resnet50', 'resnet101', 'vit', 'efficientnet', 'dino'")
    
    # Remove the classification layer to get embeddings
    if model.startswith('resnet'):
        model_instance = torch.nn.Sequential(*list(model_instance.children())[:-1])
    elif model == 'vit':
        original_forward = model_instance.forward
        model_instance.forward = lambda x: original_forward(x)[0]  # extract embeddings
    elif model == 'efficientnet':
        # Replace the classifier with an identity layer
        model_instance.classifier = torch.nn.Identity()
    
    model_instance = model_instance.to(device)
    model_instance.eval()
    
    # Process images in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Computing image embeddings"):
        batch_images = images[i:i+batch_size]
        
        # Process each image in the batch
        batch_tensors = []
        for img in batch_images:
            # Handle different input types
            if isinstance(img, str):  # Path to image
                img_data = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):  # PIL Image
                img_data = img
            elif isinstance(img, np.ndarray):  # Numpy array
                img_data = Image.fromarray(img.astype('uint8')).convert('RGB')
            elif isinstance(img, torch.Tensor):  # PyTorch tensor
                if img.shape[0] == 3:  # Already in CxHxW format
                    img_data = transforms.ToPILImage()(img)
                else:
                    img_data = transforms.ToPILImage()(img.permute(2, 0, 1))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            
            # Apply preprocessing
            tensor = preprocess(img_data).unsqueeze(0)
            batch_tensors.append(tensor)
        
        # Stack tensors and process
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        
        with torch.no_grad():
            batch_embedding = model_instance(batch_tensor)
            
            # Handle different model outputs
            if isinstance(batch_embedding, tuple):
                batch_embedding = batch_embedding[0]
            
            # Flatten if needed
            if len(batch_embedding.shape) > 2:
                batch_embedding = batch_embedding.reshape(batch_embedding.size(0), -1)
                
            all_embeddings.append(batch_embedding.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
    return embeddings

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two datasets X and Y.

    Parameters:
    - X: np.ndarray of shape (n_samples_1, d)
    - Y: np.ndarray of shape (n_samples_2, d)
    - kernel: 'rbf' or 'linear'
    - gamma: kernel bandwidth for RBF

    Returns:
    - mmd_value: float
    """
    if kernel == 'linear':
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)
    elif kernel == 'rbf':
        XX = np.exp(-cdist(X, X, 'sqeuclidean') * gamma)
        YY = np.exp(-cdist(Y, Y, 'sqeuclidean') * gamma)
        XY = np.exp(-cdist(X, Y, 'sqeuclidean') * gamma)
    else:
        raise ValueError("Unsupported kernel")

    mmd_value = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd_value


def split_source_target(dataset_dict: DatasetDict, target_key: str):
    if target_key not in dataset_dict:
        raise ValueError(f"'{target_key}' not found in dataset_dict keys: {list(dataset_dict.keys())}")
    
    target_dataset = dataset_dict[target_key]
    
    # Get all datasets except the target
    source_datasets = [ds for key, ds in dataset_dict.items() if key != target_key]
    
    # Concatenate them into one dataset
    source_dataset = concatenate_datasets(source_datasets)
    
    return source_dataset, target_dataset
