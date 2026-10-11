"""
Training script for Hand Gesture Intent Classification
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

from ws_racr.perception.gesture.pose_dataset import create_datasets, HandGestureDataset
from ws_racr.perception.gesture.pose_model import HandGestureClassifier


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, class_names):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    # Per-class accuracy
    class_acc = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc[name] = 100 * class_correct[i] / class_total[i]
        else:
            class_acc[name] = 0.0
    
    return epoch_loss, epoch_acc, class_acc


def train_model(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset = create_datasets(
        args.data_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # Get class names from the full dataset
    full_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    class_names = full_dataset.class_names
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Create model
    print("\nCreating model...")
    model = HandGestureClassifier(
        input_dim=args.input_dim,
        num_classes=num_classes
    )
    model.to(device)
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    best_model_path = None

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_fields = ["epoch", "train_loss", "train_acc", "train_lr", "val_loss", "val_acc"]
    with open(metrics_path, "w", newline="", encoding="utf-8") as mf:
        csv.DictWriter(mf, fieldnames=metrics_fields).writeheader()

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, class_acc = validate(model, val_loader, criterion, device, class_names)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Per-class Val Acc:")
        for name, acc in class_acc.items():
            print(f"    {name}: {acc:.2f}%")

        with open(metrics_path, "a", newline="", encoding="utf-8") as mf:
            csv.DictWriter(mf, fieldnames=metrics_fields).writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.4f}",
                "train_lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.4f}",
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Create checkpoint directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"hand_gesture_model_acc{val_acc:.2f}_{timestamp}.pth"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_acc': class_acc,
                'input_dim': args.input_dim,
                'num_classes': num_classes,
                'class_names': class_names
            }, checkpoint_path)
            
            print(f"  Saved best model: {checkpoint_name}")
            best_model_path = checkpoint_path
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train Hand Gesture Intent Classifier')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Path to dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=68,
                        help='Input feature dimension (default: 68)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    
    # System parameters
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints (default: checkpoints)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train model
    best_model_path = train_model(args)


if __name__ == "__main__":
    main()
