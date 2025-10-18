"""
工具函数 - utils.py
包含数据加载、模型保存加载等工具函数
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    设置随机种子，确保实验可复现
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'随机种子已设置为: {seed}')


class FlowerDataset(Dataset):
    """
    花卉数据集类
    
    参数:
        dataframe: pandas DataFrame，包含image_path和label列
        transform: 数据增强变换
        is_test: 是否为测试集
    """
    
    def __init__(self, dataframe, transform=None, is_test=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图片 {img_path}: {e}")
            # 返回黑色图片作为备用
            image = Image.new('RGB', (600, 600), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            # 测试集返回图片和路径
            return image, img_path
        else:
            # 训练集返回图片和标签
            label = int(row['label'])
            return image, label


def save_checkpoint(model, optimizer, epoch, val_acc, filepath, config=None):
    """
    保存模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        val_acc: 验证准确率
        filepath: 保存路径
        config: 配置信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }
    
    if config:
        checkpoint['config'] = config
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存检查点
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    加载模型检查点
    
    参数:
        filepath: 检查点文件路径
        model: 模型
        optimizer: 优化器 (可选)
    
    返回:
        checkpoint字典
    """
    # 加载检查点
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态 (如果提供)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class AverageMeter:
    """
    计算和存储平均值和当前值
    用于统计训练过程中的损失和准确率
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    计算top-k准确率
    
    参数:
        output: 模型输出 [batch_size, num_classes]
        target: 真实标签 [batch_size]
        topk: 计算top-k准确率的k值元组
    
    返回:
        准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # 获取top-k预测
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class CutMix:
    """
    CutMix数据增强
    在训练图像中随机裁剪一个区域并用另一张图像的对应区域替换
    
    参数:
        alpha: Beta分布的参数
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        """
        应用CutMix增强
        
        参数:
            images: 图像batch [batch_size, C, H, W]
            labels: 标签batch [batch_size]
        
        返回:
            混合后的图像、原标签、混合标签、混合比例
        """
        indices = torch.randperm(images.size(0))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # 采样混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 计算裁剪区域
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # 替换区域
        images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
        
        # 调整混合比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return images, labels, shuffled_labels, lam
    
    def _rand_bbox(self, size, lam):
        """生成随机裁剪框"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # 计算裁剪框坐标
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class MixUp:
    """
    MixUp数据增强
    线性插值两张图像及其标签
    
    参数:
        alpha: Beta分布的参数
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        """
        应用MixUp增强
        
        参数:
            images: 图像batch
            labels: 标签batch
        
        返回:
            混合后的图像、原标签、混合标签、混合比例
        """
        indices = torch.randperm(images.size(0))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # 采样混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 线性插值
        mixed_images = lam * images + (1 - lam) * shuffled_images
        
        return mixed_images, labels, shuffled_labels, lam


def prepare_data_csv(data_dir, output_path):
    """
    从目录结构生成CSV文件
    
    假设目录结构:
    data_dir/
        class_0/
            img1.jpg
            img2.jpg
        class_1/
            img1.jpg
            img2.jpg
        ...
    
    参数:
        data_dir: 数据目录路径
        output_path: 输出CSV路径
    
    返回:
        DataFrame
    """
    import pandas as pd
    
    data = []
    classes = sorted(os.listdir(data_dir))
    
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        
        # 跳过非目录文件
        if not os.path.isdir(class_path):
            continue
        
        # 遍历该类别的所有图片
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_path, img_name)
                data.append({
                    'image_path': img_path,
                    'label': label,
                    'class_name': class_name
                })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存CSV
    df.to_csv(output_path, index=False)
    print(f'已保存 {len(df)} 个样本到 {output_path}')
    print(f'类别数: {df["label"].nunique()}')
    print(f'每类样本数统计:')
    print(df['label'].value_counts().describe())
    
    return df


def prepare_test_csv(test_dir, output_path):
    """
    为测试集生成CSV文件
    
    参数:
        test_dir: 测试图片目录
        output_path: 输出CSV路径
    
    返回:
        DataFrame
    """
    import pandas as pd
    
    data = []
    
    # 遍历测试目录中的所有图片
    for img_name in sorted(os.listdir(test_dir)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(test_dir, img_name)
            data.append({
                'image_path': img_path
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存CSV
    df.to_csv(output_path, index=False)
    print(f'已保存 {len(df)} 个测试样本到 {output_path}')
    
    return df


# 测试代码
if __name__ == '__main__':
    print("工具函数模块测试")
    
    # 测试设置随机种子
    set_seed(42)
    
    # 测试AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"平均值: {meter.avg}")
    
    print("\n所有工具函数测试通过!")