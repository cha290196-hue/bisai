"""
模型定义 - model.py
定义花卉分类模型结构
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm


class FlowerClassifier(nn.Module):
    """
    花卉分类器
    
    参数:
        num_classes: 分类类别数 (默认100)
        model_name: 预训练模型名称 (默认resnet50)
        pretrained: 是否使用预训练权重 (默认False)
        dropout: Dropout比率 (默认0.5)
    """
    
    def __init__(self, num_classes=100, model_name='resnet50', 
                 pretrained=False, dropout=0.5):
        super(FlowerClassifier, self).__init__()
        
        # 使用timm库加载EfficientNet模型
        if model_name.startswith('efficientnet'):
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        else:
            # 保持原有的ResNet实现
            if model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                self.feature_dim = self.backbone.fc.in_features
                # 移除原始分类头
                self.backbone.fc = nn.Identity()
            elif model_name == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
                self.feature_dim = self.backbone.fc.in_features
                # 移除原始分类头
                self.backbone.fc = nn.Identity()
            elif model_name == 'resnet152':
                self.backbone = models.resnet152(pretrained=pretrained)
                self.feature_dim = self.backbone.fc.in_features
                # 移除原始分类头
                self.backbone.fc = nn.Identity()
            else:
                raise ValueError(f"不支持的模型: {model_name}")
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量 [batch_size, 3, H, W]
        
        返回:
            输出logits [batch_size, num_classes]
        """
        # 提取特征
        features = self.backbone(x)
        
        # 分类
        output = self.classifier(features)
        
        return output


class EnsembleModel(nn.Module):
    """
    集成模型 - 组合多个模型的预测结果
    
    参数:
        models: 模型列表
        weights: 每个模型的权重 (可选)
    """
    
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        # 设置权重
        if weights is None:
            # 如果没有指定权重，使用平均权重
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "权重数量必须与模型数量一致"
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def forward(self, x):
        """
        前向传播 - 加权平均所有模型的输出
        
        参数:
            x: 输入图像张量
        
        返回:
            加权平均后的输出
        """
        outputs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
            outputs.append(output)
        
        # 加权平均
        ensemble_output = sum(w * o for w, o in zip(self.weights, outputs))
        
        return ensemble_output


# 推荐的模型配置
MODEL_CONFIGS = {
    'resnet50': {
        'model_name': 'resnet50',
        'image_size': 224,
        'batch_size': 32,
        'params': '25M',
        'description': '经典模型，训练速度快'
    },
    'resnet101': {
        'model_name': 'resnet101',
        'image_size': 224,
        'batch_size': 24,
        'params': '45M',
        'description': '更深的ResNet，准确率更高'
    },
    'resnet152': {
        'model_name': 'resnet152',
        'image_size': 224,
        'batch_size': 16,
        'params': '60M',
        'description': '最深的ResNet，最高准确率'
    }
}


def get_model_info(model_name):
    """
    获取模型配置信息
    
    参数:
        model_name: 模型名称
    
    返回:
        模型配置字典
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    else:
        return {
            'model_name': model_name,
            'image_size': 224,
            'batch_size': 32,
            'params': 'Unknown',
            'description': '自定义模型'
        }


# 测试代码
if __name__ == '__main__':
    # 创建模型
    print("测试模型创建...")
    model = FlowerClassifier(num_classes=100, model_name='resnet50')
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {output.shape}')
    print(f'模型总参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # 显示可用模型
    print("\n可用的模型:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  - {name}")
        print(f"    参数量: {config['params']}, 推荐batch_size: {config['batch_size']}")
        print(f"    说明: {config['description']}")