#!/usr/bin/env python3
"""
Rotation Classification Training Script

Converts images to classify rotation angles (0, 90, 180, 270 degrees)
using a ResNet-18 model with enhanced training feedback.
"""

import os
import time
import argparse
import logging
import csv
from collections import Counter
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_dir: str = "logs") -> tuple:
    """Setup logging configuration and create metrics CSV file."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup text logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/rotation_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup CSV metrics logging
    metrics_file = f"{log_dir}/training_metrics_{timestamp}.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                        'epoch_time', 'learning_rate', 'best_so_far'])
    
    return logger, metrics_file


def summarize_dataset(dataset: ImageFolder, name: str = "dataset") -> None:
    """Print dataset summary with class distribution."""
    class_counts = Counter([label for _, label in dataset.samples])
    class_names = dataset.classes

    print(f"📊 {name.upper()} SET SUMMARY:")
    print(f"Total images: {len(dataset)}")
    for idx, count in sorted(class_counts.items()):
        print(f"  Class '{class_names[idx]}': {count} images")
    print()


def create_data_loaders(train_dir: str, val_dir: str, batch_size: int = 256, 
                       image_size: int = 300) -> tuple:
    """Create training and validation data loaders."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(train_dir, transform=transform)
    val_data = ImageFolder(val_dir, transform=transform)
    
    summarize_dataset(train_data, name="train")
    summarize_dataset(val_data, name="validation")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader, train_data.classes


def create_model(num_classes: int, device: torch.device) -> nn.Module:
    """Create and setup the ResNet-18 model."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def validate_model(model: nn.Module, val_loader: DataLoader, 
                  criterion: nn.Module, device: torch.device) -> tuple:
    """Validate the model and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        for imgs, labels in val_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            val_pbar.set_postfix({'acc': f'{100*correct/total:.1f}%'})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
               criterion: nn.Module, device: torch.device, epoch: int) -> tuple:
    """Train for one epoch with progress tracking."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (imgs, labels) in enumerate(train_pbar):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        current_acc = 100 * correct / total
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                   train_loss: float, val_loss: float, val_acc: float,
                   checkpoint_dir: str, is_best: bool = False) -> None:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1:02d}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = f"{checkpoint_dir}/best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"🏆 New best model saved! Val Acc: {val_acc:.2f}%")


def log_metrics(metrics_file: str, epoch: int, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, epoch_time: float, lr: float, 
               is_best: bool) -> None:
    """Log metrics to CSV file."""
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{train_loss:.6f}", f"{train_acc:.4f}", 
                        f"{val_loss:.6f}", f"{val_acc:.4f}", f"{epoch_time:.2f}", 
                        f"{lr:.2e}", is_best])


def evaluate_model(model: nn.Module, val_loader: DataLoader, 
                  class_names: list, device: torch.device) -> None:
    """Comprehensive model evaluation with confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\n🔍 Running final evaluation...")
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    correct = sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    accuracy = 100 * correct / total
    print(f"✅ Final Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("🌀 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train rotation classification model")
    parser.add_argument("--train_dir", type=str, default="../data/rotation/classification/train",
                       help="Training data directory")
    parser.add_argument("--val_dir", type=str, default="../data/rotation/classification/test", 
                       help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=300, help="Input image size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                       help="Directory to save checkpoints")
    parser.add_argument("--patience", type=int, default=8, 
                       help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.01,
                       help="Minimum improvement for early stopping")
    
    args = parser.parse_args()
    
    # Setup
    logger, metrics_file = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Training arguments: {vars(args)}")
    
    # Data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        args.train_dir, args.val_dir, args.batch_size, args.image_size
    )
    
    # Model setup
    model = create_model(len(class_names), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    # Training variables
    best_val_acc = 0.0
    patience_counter = 0
    train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    print(f"🚀 Training started! Target: {args.epochs} epochs")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Track history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Check if best model
        is_best = val_acc > best_val_acc
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1:02d}/{args.epochs} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if is_best:
            print(f"   🌟 New best validation accuracy!")
        
        # Save checkpoint and log metrics
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, 
                       args.checkpoint_dir, is_best)
        log_metrics(metrics_file, epoch, train_loss, train_acc, val_loss, val_acc,
                   epoch_time, optimizer.param_groups[0]['lr'], is_best)
        
        # Log to file
        logger.info(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                   f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Time={epoch_time:.1f}s")
        
        # Early stopping check
        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stopping triggered! No improvement for {args.patience} epochs")
            break
        
        print("-" * 80)
    
    # Final evaluation
    print(f"\n🎯 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(f"{args.checkpoint_dir}/best_model.pth")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    evaluate_model(model, val_loader, class_names, device)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()