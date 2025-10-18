"""
预测代码 - predict.py
用于生成测试集的预测结果
运行方式: python code/predict.py 测试集文件路径
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
import sys
from tqdm import tqdm
from model import FlowerClassifier
from utils import FlowerDataset, load_checkpoint

def predict_with_tta(model, images, device, n_tta=5):
    """使用测试时增强(TTA)进行预测"""
    model.eval()
    batch_size = images.size(0)
    
    # 定义TTA变换
    tta_transforms = [
        lambda x: x,  # 原图
        lambda x: torch.flip(x, dims=[3]),  # 水平翻转
        lambda x: torch.flip(x, dims=[2]),  # 垂直翻转
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 旋转90度
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 旋转270度
    ]
    
    all_outputs = []
    
    with torch.no_grad():
        for i, transform in enumerate(tta_transforms[:n_tta]):
            augmented = transform(images)
            outputs = model(augmented)
            all_outputs.append(outputs)
    
    # 对所有TTA结果取平均
    avg_outputs = torch.stack(all_outputs).mean(dim=0)
    return avg_outputs

def main(test_csv_path='data/test.csv'):
    # ========== 配置参数 ==========
    config = {
        'num_classes': 100,
        'image_size': 600,
        'batch_size': 16,
        'model_name': 'efficientnet_b4',
        'num_workers': 4,
        'use_tta': True,  # 是否使用测试时增强
        'n_tta': 5,       # TTA次数
        'model_path': 'model/best_model.pth',
        'output_path': 'results/submission.csv'
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # ========== 数据预处理 ==========
    test_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========== 加载测试数据 ==========
    print(f'\n正在加载测试数据: {test_csv_path}')
    test_df = pd.read_csv(test_csv_path)
    print(f'测试集样本数: {len(test_df)}')
    
    test_dataset = FlowerDataset(test_df, transform=test_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # ========== 加载模型 ==========
    print(f'\n正在加载模型: {config["model_path"]}')
    model = FlowerClassifier(
        num_classes=config['num_classes'],
        model_name=config['model_name'],
        pretrained=False
    )
    model = model.to(device)
    
    # 加载训练好的权重
    checkpoint = load_checkpoint(config['model_path'], model)
    print(f'模型加载成功!')
    print(f'  - Epoch: {checkpoint["epoch"]}')
    print(f'  - 验证准确率: {checkpoint["val_acc"]:.2f}%')
    
    # ========== 开始预测 ==========
    print(f'\n开始预测...')
    print(f'TTA状态: {"启用" if config["use_tta"] else "禁用"}')
    
    model.eval()
    all_predictions = []
    all_image_paths = []
    
    with torch.no_grad():
        for images, image_paths in tqdm(test_loader, desc='预测中'):
            images = images.to(device)
            
            # 使用TTA或普通预测
            if config['use_tta']:
                outputs = predict_with_tta(model, images, device, config['n_tta'])
            else:
                outputs = model(images)
            
            # 获取预测类别
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_image_paths.extend(image_paths)
    
    # ========== 保存结果 ==========
    print(f'\n正在保存预测结果...')
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'image_path': all_image_paths,
        'label': all_predictions
    })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    
    # 保存CSV文件
    submission_df.to_csv(config['output_path'], index=False)
    
    print(f'预测完成!')
    print(f'结果已保存到: {config["output_path"]}')
    print(f'\n预测统计:')
    print(f'  - 总预测数: {len(all_predictions)}')
    print(f'  - 预测类别数: {len(set(all_predictions))}')
    print(f'  - 标签范围: {min(all_predictions)} - {max(all_predictions)}')
    
    # 显示前几行结果
    print(f'\n前5行预测结果:')
    print(submission_df.head())

if __name__ == '__main__':
    # 从命令行获取测试集路径
    if len(sys.argv) > 1:
        test_csv = sys.argv[1]
    else:
        test_csv = 'data/test.csv'
    
    main(test_csv)