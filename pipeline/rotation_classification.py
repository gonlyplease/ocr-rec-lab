#!/usr/bin/env python3
"""
Text Orientation Classification Training Script
Optimized for single characters and short text snippets with rectangular training
"""

import os
import time
import argparse
import logging
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


class AspectPreservingResize:
    """Resize images preserving aspect ratio with padding"""
    def __init__(self, size, fill_color=(255, 255, 255)):
        self.size = size
        self.fill_color = fill_color
    
    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create padded image
        new_img = Image.new('RGB', (self.size, self.size), self.fill_color)
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img, w/h  # Return image and original aspect ratio


class TextOrientationDataset(Dataset):
    """Dataset for text orientation with aspect ratio preservation"""
    def __init__(self, root_dir, transform=None, image_size=224, return_paths=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.return_paths = return_paths
        self.aspect_resize = AspectPreservingResize(image_size)
        
        # Collect all images with labels
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Apply aspect preserving resize
        img, aspect_ratio = self.aspect_resize(img)
        
        if self.transform:
            img = self.transform(img)
        
        if self.return_paths:
            return img, label, aspect_ratio, img_path
        return img, label, aspect_ratio
    
    def __len__(self):
        return len(self.samples)


class LightweightTextNet(nn.Module):
    """Lightweight CNN optimized for text orientation"""
    def __init__(self, num_classes=4, use_aspect_ratio=True):
        super().__init__()
        self.use_aspect_ratio = use_aspect_ratio
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Aspect ratio embedding
        if use_aspect_ratio:
            self.aspect_embedding = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU()
            )
            classifier_input = 128 * 4 * 4 + 32
        else:
            classifier_input = 128 * 4 * 4
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, aspect_ratios=None):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        if self.use_aspect_ratio and aspect_ratios is not None:
            aspect_features = self.aspect_embedding(aspect_ratios.unsqueeze(1).float())
            features = torch.cat([features, aspect_features], dim=1)
        
        return self.classifier(features)


class ResNetWithAspect(nn.Module):
    """ResNet-18 with aspect ratio awareness"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Aspect ratio processing
        self.aspect_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, aspect_ratios):
        visual_features = self.resnet(x)
        aspect_features = self.aspect_embedding(aspect_ratios.unsqueeze(1).float())
        combined = torch.cat([visual_features, aspect_features], dim=1)
        return self.classifier(combined)


def create_text_specific_transforms(image_size=224, training=True):
    """Transforms optimized for text images"""
    if training:
        return transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def analyze_predictions(model, val_loader, device, class_names, save_dir="analysis"):
    """Detailed analysis of model predictions"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    confidence_threshold = 0.9
    
    low_confidence_samples = []
    misclassified_samples = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing predictions"):
            if len(batch) == 4:
                imgs, labels, aspect_ratios, paths = batch
                all_paths.extend(paths)
            else:
                imgs, labels, aspect_ratios = batch
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            aspect_ratios = aspect_ratios.to(device)
            
            outputs = model(imgs, aspect_ratios)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Track low confidence and misclassified
            for i in range(len(labels)):
                if confidences[i] < confidence_threshold:
                    low_confidence_samples.append({
                        'true_label': class_names[labels[i].item()],
                        'pred_label': class_names[preds[i].item()],
                        'confidence': confidences[i].item(),
                        'all_probs': probs[i].cpu().numpy().tolist(),  # Convert to list here
                        'path': paths[i] if all_paths else None
                    })
                
                if preds[i] != labels[i]:
                    misclassified_samples.append({
                        'true_label': class_names[labels[i].item()],
                        'pred_label': class_names[preds[i].item()],
                        'confidence': confidences[i].item(),
                        'all_probs': probs[i].cpu().numpy().tolist(),  # Add this line if you want probabilities
                        'path': paths[i] if all_paths else None
                    })
    
    # Calculate metrics
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # Save analysis report
    report = {
        'accuracy': accuracy,
        'total_samples': len(all_labels),
        'low_confidence_count': len(low_confidence_samples),
        'misclassified_count': len(misclassified_samples),
        'confidence_threshold': confidence_threshold,
        'classification_report': classification_report(all_labels, all_preds, 
                                                      target_names=class_names, 
                                                      output_dict=True)
    }
    
    with open(f"{save_dir}/analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save problem cases
    if misclassified_samples:
        with open(f"{save_dir}/misclassified_samples.json", 'w') as f:
            json.dump(misclassified_samples[:50], f, indent=2)
    
    if low_confidence_samples:
        with open(f"{save_dir}/low_confidence_samples.json", 'w') as f:
            json.dump(low_confidence_samples[:50], f, indent=2)
    
    return report


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, use_aspect=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        imgs, labels, aspect_ratios = batch[:3]
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        aspect_ratios = aspect_ratios.to(device)
        
        optimizer.zero_grad()
        
        if use_aspect:
            outputs = model(imgs, aspect_ratios)
        else:
            outputs = model(imgs)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    return total_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device, use_aspect=True):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch in pbar:
            imgs, labels, aspect_ratios = batch[:3]
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            aspect_ratios = aspect_ratios.to(device)
            
            if use_aspect:
                outputs = model(imgs, aspect_ratios)
            else:
                outputs = model(imgs)
                
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train text orientation classifier")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="lightweight", 
                       choices=["lightweight", "resnet18"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_aspect_ratio", action="store_true", default=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--analyze_every", type=int, default=5)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create datasets
    train_transform = create_text_specific_transforms(args.image_size, training=True)
    val_transform = create_text_specific_transforms(args.image_size, training=False)
    
    train_dataset = TextOrientationDataset(args.train_dir, train_transform, args.image_size)
    val_dataset = TextOrientationDataset(args.val_dir, val_transform, args.image_size, return_paths=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(train_dataset.classes)
    if args.model_type == "lightweight":
        model = LightweightTextNet(num_classes, use_aspect_ratio=args.use_aspect_ratio)
    else:
        model = ResNetWithAspect(num_classes)
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                          criterion, device, epoch, args.use_aspect_ratio)
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.use_aspect_ratio)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, f"{args.checkpoint_dir}/best_model.pth")
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Periodic analysis
        if (epoch + 1) % args.analyze_every == 0:
            print("\nRunning detailed analysis...")
            report = analyze_predictions(model, val_loader, device, 
                                       train_dataset.classes, 
                                       save_dir=f"analysis_epoch_{epoch+1}")
            print(f"Analysis complete. Low confidence: {report['low_confidence_count']}, "
                  f"Misclassified: {report['misclassified_count']}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()


#python .\pipeline\rotation_classification.py --train_dir data\rotation\classification\train --val_dir data\rotation\classification\test --model_type lightweight --use_aspect_ratio