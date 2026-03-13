import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

# 设置路径
working_dir = "/kaggle/working/"
model_path = "/kaggle/working/results/lung_nodule_detection_enhanced/weights/best.pt"
yolo_dataset_path = "/kaggle/input/yolodata/yolo_dataset"
val_images_path = os.path.join(yolo_dataset_path, "images", "val")
val_labels_path = os.path.join(yolo_dataset_path, "labels", "val")

model = YOLO(model_path)

# 1. 模型性能评估 (修复指标访问)
def evaluate_model():
    # 在验证集上评估模型
    metrics = model.val(
        data=os.path.join(working_dir, "lung_nodule.yaml"),
        split='val',
        imgsz=512,
        batch=32,
        conf=0.1,
        iou=0.45
    )
    
    # 打印关键指标 - 修复指标访问方式
    print("\n模型评估结果:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 处理数组类型的指标
    print(f"精确率: {metrics.box.mp[0]:.4f}" if isinstance(metrics.box.mp, np.ndarray) else f"精确率: {metrics.box.mp:.4f}")
    print(f"召回率: {metrics.box.mr[0]:.4f}" if isinstance(metrics.box.mr, np.ndarray) else f"召回率: {metrics.box.mr:.4f}")
    
    return metrics


if __name__ == '__main__':
    evaluate_model()