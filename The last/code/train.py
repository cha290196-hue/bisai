"""
训练代码 - train.py
用于训练花卉识别模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
from model import FlowerClassifier
from utils import FlowerDataset, set_seed, save_checkpoint, load_checkpoint

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    # ========== 配置参数 ==========
    config = {
        'seed': 42,
        'num_classes': 100,
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'image_size': 600,
        'model_name': 'efficientnet_b4',
        'pretrained': True,
        'num_workers': 4,
        'use_amp': True,  # 混合精度训练
        'data_path': 'data/train.csv',
        'save_dir': 'model/',
    }
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # ========== 数据预处理 ==========
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集数据增强
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========== 加载数据 ==========
    print('正在加载数据...')
    df = pd.read_csv(config['data_path'])
    
    # 分层划分训练集和验证集
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2,
        stratify=df['label'],
        random_state=config['seed']
    )
    
    print(f'训练集样本数: {len(train_df)}')
    print(f'验证集样本数: {len(val_df)}')
    
    # 创建数据集
    train_dataset = FlowerDataset(train_df, transform=train_transform)
    val_dataset = FlowerDataset(val_df, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # ========== 创建模型 ==========
    print(f'\n正在创建模型: {config["model_name"]}')
    model = FlowerClassifier(
        num_classes=config['num_classes'],
        model_name=config['model_name'],
        pretrained=config['pretrained']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型总参数量: {total_params:,}')
    
    # ========== 损失函数和优化器 ==========
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None
    
    # ========== 训练循环 ==========
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print('\n开始训练...\n')
    
    for epoch in range(config['num_epochs']):
        print(f'{"="*70}')
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}]')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'{"="*70}')
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f'\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                filepath=os.path.join(config['save_dir'], 'best_model.pth'),
                config=config
            )
            print(f'✓ 已保存最佳模型! 验证准确率: {val_acc:.2f}%')
        
        print()
    
    print(f'{"="*70}')
    print(f'训练完成!')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    print(f'{"="*70}')
    
    # 保存训练历史
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f'训练历史已保存到: {history_path}')
    
    # 保存配置文件
    config_path = os.path.join(config['save_dir'], 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'配置文件已保存到: {config_path}')

if __name__ == '__main__':
    main()