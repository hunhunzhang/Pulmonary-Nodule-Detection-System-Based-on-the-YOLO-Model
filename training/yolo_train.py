import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

# 设置路径和参数
input_root = "/kaggle/input/yolodata/yolo_dataset"  # 只读数据集路径
working_dir = "/kaggle/working/"  # 可写工作目录
yaml_path = os.path.join(working_dir, "lung_nodule.yaml")  # 数据集配置文件

# 创建YOLO数据集配置文件
yaml_content = f"""
path: {input_root}
train: images/train
val: images/val

names:
  0: nodule
"""

with open(yaml_path, 'w') as f:
    f.write(yaml_content)

# 训练YOLOv8模型 - 使用更大模型和优化策略
model = YOLO("yolov8l.pt")  # 使用large版本模型（比nano大4倍）

# 高级训练配置
results = model.train(
    data=yaml_path,
    epochs=50,  # 增加训练轮次
    batch=32,    # 减小批次大小以适应更大模型
    imgsz=512,  
    patience=10, # 增加早停耐心值
    
    # 优化学习率策略
    lr0=0.001,   # 初始学习率
    lrf=0.01,    # 最终学习率 = lr0 * lrf
    momentum=0.9,
    weight_decay=0.0005,
    
    # 高级数据增强
    augment=True,
    hsv_h=0.3,   # 增强色调变化
    hsv_s=0.8,   # 增强饱和度变化
    translate=0.2,# 增加平移增强
    scale=0.5,   # 增加缩放增强
    shear=0.1,   # 增加剪切变换
    perspective=0.001,  # 增加透视变换
    
    # 损失函数权重调整（针对肺结节特点）
    box=7.5,     # 增加边界框损失权重
    cls=0.5,     # 降低分类损失权重（单类别）
    dfl=1.5,     # 增加分布焦点损失权重
    
    close_mosaic=10, # 最后10轮关闭马赛克增强
    
    project=os.path.join(working_dir, "results"),
    name="lung_nodule_detection_enhanced",
    optimizer="AdamW",  # 使用更先进的AdamW优化器
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_period=10,  # 每10轮保存一次模型
)

print("训练完成! ")