import os
import random
import shutil
import yaml
from PIL import Image
import numpy as np
from skimage.measure import label, regionprops
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# 设置路径和参数
input_root = ""  # 只读数据集路径
working_dir = ""  # 可写工作目录
output_root = os.path.join(working_dir, "yolo_dataset")  # YOLO格式数据集路径
yaml_path = os.path.join(working_dir, "lung_nodule.yaml")  # 数据集配置文件

# 创建YOLO格式的目录结构
os.makedirs(os.path.join(output_root, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_root, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_root, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(output_root, "labels", "val"), exist_ok=True)

# 收集所有图像和标注信息
image_info = []

# 遍历所有病人文件夹
for patient_id in os.listdir(input_root):
    patient_dir = os.path.join(input_root, patient_id)
    if not os.path.isdir(patient_dir):
        continue
    
    # 获取所有原始切片图像
    for file in os.listdir(patient_dir):
        if not file.endswith(".png"):
            continue
        if "_roi_" in file or "_all_nodules" in file or "_nodule_" in file:
            continue
        
        # 检查是否负样本
        base_name = file.replace(".png", "")
        no_nodule_txt = os.path.join(patient_dir, f"{base_name}_no_nodule.txt")
        is_negative = os.path.exists(no_nodule_txt)
        
        # 获取结节mask文件
        nodule_files = [f for f in os.listdir(patient_dir) 
                       if f.startswith(f"{base_name}_nodule_") and f.endswith(".png")]
        
        image_path = os.path.join(patient_dir, file)
        image_info.append({
            "patient_id": patient_id,
            "image_path": image_path,
            "base_name": base_name,
            "is_negative": is_negative,
            "nodule_files": nodule_files,
            "patient_dir": patient_dir
        })

# 划分训练集和验证集 (70% 训练, 30% 验证)
train_data, val_data = train_test_split(
    image_info, test_size=0.3, random_state=42
)

# 处理训练集数据
def process_dataset(data, dataset_type):
    for item in data:
        # 生成唯一文件名 (病人ID + 图像名)
        new_filename = f"{item['patient_id']}_{item['base_name']}.png"
        dest_image = os.path.join(output_root, "images", dataset_type, new_filename)
        dest_label = os.path.join(output_root, "labels", dataset_type, new_filename.replace(".png", ".txt"))
        
        # 复制图像文件
        shutil.copy(item["image_path"], dest_image)
        
        # 如果是负样本，创建空标签文件
        if item["is_negative"]:
            open(dest_label, 'w').close()
            continue
        
        # 处理结节标注
        img = Image.open(item["image_path"])
        img_width, img_height = img.size
        label_lines = []
        
        # 处理每个结节mask
        for mask_file in item["nodule_files"]:
            mask_path = os.path.join(item["patient_dir"], mask_file)
            mask_img = Image.open(mask_path)
            mask_array = np.array(mask_img)
            
            # 二值化处理
            binary_mask = mask_array > 0
            
            # 获取边界框
            labeled = label(binary_mask)
            regions = regionprops(labeled)
            if regions:
                region = regions[0]  # 每个mask只有一个结节
                min_row, min_col, max_row, max_col = region.bbox
                
                # 转换为YOLO格式 (归一化中心坐标和宽高)
                x_center = (min_col + (max_col - min_col) / 2) / img_width
                y_center = (min_row + (max_row - min_row) / 2) / img_height
                width = (max_col - min_col) / img_width
                height = (max_row - min_row) / img_height
                
                # 添加到标注 (类别0为结节)
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 写入标签文件
        with open(dest_label, 'w') as f:
            f.write("\n".join(label_lines))

# 处理训练集和验证集
process_dataset(train_data, "train")
process_dataset(val_data, "val")

