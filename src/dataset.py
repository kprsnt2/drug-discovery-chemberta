"""
PyTorch Dataset for Drug Discovery

Custom Dataset class for loading drug molecules and their labels.
Handles SMILES tokenization using ChemBERTa tokenizer.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, PROCESSED_DATA_DIR


class DrugDiscoveryDataset(Dataset):
    """
    PyTorch Dataset for drug discovery classification.
    
    Args:
        data_path: Path to CSV file with 'smiles' and 'label' columns
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = None
    ):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length or MODEL_CONFIG['max_length']
        
        # Ensure required columns exist
        required_cols = ['smiles', 'label']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Drop any rows with missing values
        self.data = self.data.dropna(subset=required_cols)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        smiles = str(row['smiles'])
        label = int(row['label'])
        
        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def get_labels(self) -> List[int]:
        """Get all labels for computing class weights."""
        return self.data['label'].tolist()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.
        
        Returns:
            Tensor with weights for each class
        """
        labels = self.get_labels()
        class_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)
        
        weights = []
        for i in range(len(class_counts)):
            weight = total / (len(class_counts) * class_counts[i])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class SMILESCollator:
    """
    Data collator for SMILES tokenization.
    Handles batch tokenization with padding.
    """
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 2
):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = DrugDiscoveryDataset(train_path, tokenizer)
    val_dataset = DrugDiscoveryDataset(val_path, tokenizer)
    test_dataset = DrugDiscoveryDataset(test_path, tokenizer)
    
    # Create collator
    collator = SMILESCollator(tokenizer)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_class_weights()


if __name__ == "__main__":
    # Test dataset loading
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if train_path.exists():
        dataset = DrugDiscoveryDataset(str(train_path), tokenizer)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"Sample label: {sample['labels']}")
    else:
        print(f"Training data not found at {train_path}")
        print("Run 'python scripts/download_all.py' first")
