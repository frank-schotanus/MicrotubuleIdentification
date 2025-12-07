"""
Training script for microtubule detection models.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import load_annotations, get_image_list, split_dataset
from src.dataset import MicrotubuleDataset, SimpleAugmentation
from src.models import create_model


class Trainer:
    """
    Trainer class for microtubule detection models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 output_dir: str,
                 max_epochs: int = 100,
                 patience: int = 10,
                 use_amp: bool = False,
                 gradient_accumulation_steps: int = 1):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            output_dir: Directory to save checkpoints and logs
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience (epochs without improvement)
            use_amp: Use automatic mixed precision training
            gradient_accumulation_steps: Steps to accumulate gradients
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Setup automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Accumulate loss (multiply back to get true loss)
            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass with optional mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Accumulate loss
                total_loss += loss.item()
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """
        Main training loop.
        """
        print(f"Starting training for {self.max_epochs} epochs...")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save history
            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
            else:
                self.epochs_without_improvement += 1
            
            # Save latest checkpoint
            self.save_checkpoint('latest_model.pth')
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train microtubule detection model')
    
    # Data arguments
    parser.add_argument('--mrc_dir', type=str, required=True,
                        help='Directory containing MRC files')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='Path to annotation text file')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save checkpoints and logs')
    
    # Dataset arguments
    parser.add_argument('--target_type', type=str, default='heatmap',
                        choices=['heatmap', 'distance'],
                        help='Type of target representation')
    parser.add_argument('--normalization', type=str, default='zscore',
                        choices=['zscore', 'minmax', 'percentile'],
                        help='Image normalization method')
    parser.add_argument('--heatmap_sigma', type=float, default=3.0,
                        help='Sigma for Gaussian heatmaps')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--resize_to', type=int, nargs=2, default=None,
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Resize all images to this size (e.g., --resize_to 512 512). Required for variable-sized images.')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'simple'],
                        help='Model architecture')
    parser.add_argument('--init_features', type=int, default=32,
                        help='Initial number of features for U-Net')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, mps, or cpu)')
    
    # Memory optimization arguments
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (AMP) training to save memory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients (effective_batch_size = batch_size * gradient_accumulation_steps)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(args.annotation_file)
    print(f"Loaded annotations for {len(annotations)} images")
    
    # Get image list (only use annotated images for training)
    print("Scanning MRC directory...")
    image_names = get_image_list(args.mrc_dir, annotations, include_unlabeled=False)
    print(f"Found {len(image_names)} annotated MRC files")
    
    # Split dataset
    print("Splitting dataset...")
    train_names, val_names, test_names = split_dataset(
        image_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")
    
    # Save split information
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_info = {
        'train': train_names,
        'val': val_names,
        'test': test_names
    }
    with open(output_dir / 'split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create datasets
    print("Creating datasets...")
    augmentation = SimpleAugmentation() if args.augment else None
    
    # Convert resize_to to tuple if provided
    resize_to = tuple(args.resize_to) if args.resize_to else None
    if resize_to:
        print(f"All images will be resized to {resize_to[0]}x{resize_to[1]}")
    
    train_dataset = MicrotubuleDataset(
        mrc_dir=args.mrc_dir,
        image_names=train_names,
        annotations=annotations,
        target_type=args.target_type,
        normalization=args.normalization,
        heatmap_sigma=args.heatmap_sigma,
        transform=augmentation,
        resize_to=resize_to
    )
    
    val_dataset = MicrotubuleDataset(
        mrc_dir=args.mrc_dir,
        image_names=val_names,
        annotations=annotations,
        target_type=args.target_type,
        normalization=args.normalization,
        heatmap_sigma=args.heatmap_sigma,
        transform=None,  # No augmentation for validation
        resize_to=resize_to
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        in_channels=1,
        out_channels=1,
        init_features=args.init_features
    )
    model = model.to(device)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Use MSE loss for heatmap/distance regression
    criterion = nn.MSELoss()
    
    # TODO: Consider other loss functions:
    # - Focal Loss for handling class imbalance
    # - Combined losses (MSE + BCE for probability maps)
    # - Custom losses that penalize false negatives more heavily
    
    # Save training configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print memory optimization settings
    if args.use_amp:
        print("Using automatic mixed precision (AMP) training")
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        print(f"Using gradient accumulation: {args.gradient_accumulation_steps} steps")
        print(f"Effective batch size: {effective_batch_size}")
    
    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
