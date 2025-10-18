import os
import pandas as pd
from sklearn.model_selection import train_test_split
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
def convert_raw_images_to_dataset(image_dir, output_dir="data", test_size=0.2, class_name="default_class"):
    """
    将纯图片文件夹转换为可用数据集
    1. 收集所有图片路径
    2. 生成带标签的CSV（单类别默认标签0，可后续手动修改）
    3. 划分训练集和验证集并保存
    
    参数:
        image_dir: 存放图片的文件夹路径（直接包含.jpg/.png等文件）
        output_dir: 输出CSV文件的目录（默认"data"，与代码库一致）
        test_size: 验证集比例（默认0.2）
        class_name: 类别名称（默认"default_class"，可修改）
    """
    # 检查图片目录是否存在
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # 收集所有图片路径
    image_paths = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            # 获取绝对路径（避免相对路径问题）
            img_path = os.path.abspath(os.path.join(image_dir, filename))
            image_paths.append(img_path)
    
    # 检查是否有图片
    if not image_paths:
        raise ValueError(f"目录中未找到图片文件: {image_dir}")
    
    # 生成带标签的数据集（单类别默认标签0，可手动修改）
    dataset = pd.DataFrame({
        "image_path": image_paths,
        "label": 0,  # 所有图片默认标签为0，多类别可后续手动修改此列
        "class_name": class_name
    })
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 划分训练集和验证集（保持与train.py一致的划分方式）
    train_df, val_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,  # 固定随机种子，确保划分一致
        stratify=dataset["label"]  # 按标签分层（单类别时等价于随机划分）
    )
    
    # 保存为CSV（与代码库train.py的config['data_path']对应）
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "val.csv")
    dataset.to_csv(os.path.join(output_dir, "full_dataset.csv"), index=False)  # 全量数据备份
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # 输出信息
    print(f"数据集转换完成！")
    print(f"总图片数: {len(dataset)}")
    print(f"训练集: {len(train_df)} 张图片 -> {train_csv}")
    print(f"验证集: {len(val_df)} 张图片 -> {val_csv}")
    print(f"标签说明: 所有图片默认标签为0（单类别），如需多类别，可手动编辑CSV的'label'列")

if __name__ == "__main__":
    # 设置随机种子（确保划分结果可复现）
    set_seed(42)
    
    # 配置你的图片目录（替换为实际路径）
    IMAGE_DIR = r"C:\Users\25185\Desktop\X\train"  # 你的纯图片文件夹
    
    # 执行转换
    try:
        convert_raw_images_to_dataset(IMAGE_DIR)
    except Exception as e:
        print(f"转换失败: {str(e)}")