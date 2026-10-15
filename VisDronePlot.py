import cv2
import matplotlib.pyplot as plt

def read_annotation(ann_path):
    """
    读取 VisDrone 标注文件
    返回: list of dict, 每个元素包含 bbox 和类别信息
    """
    annotations = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            
            ann = {
                'bbox_left': int(parts[0]),
                'bbox_top': int(parts[1]),
                'bbox_width': int(parts[2]),
                'bbox_height': int(parts[3]),
                'score': int(parts[4]),           # 1=有效, 0=忽略
                'category': int(parts[5]),         # 类别 ID
                'truncation': int(parts[6]),       # 截断
                'occlusion': int(parts[7])         # 遮挡
            }
            annotations.append(ann)
    return annotations


def visualize_annotation(img_path, ann_path, show_ignored=False):
    """可视化图像和标注框"""
    # 读取图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 读取标注
    anns = read_annotation(ann_path)
    
    # 类别颜色映射
    colors = {
        1: (255, 0, 0),    # pedestrian - 红
        2: (255, 100, 0),  # people - 橙红
        3: (0, 255, 0),    # bicycle - 绿
        4: (0, 0, 255),    # car - 蓝
        5: (255, 0, 255),  # van - 紫
        6: (0, 255, 255),  # truck - 黄
        9: (255, 255, 0),  # bus - 青
    }
    
    for ann in anns:
        # 跳过忽略区域（score=0）
        if ann['score'] == 0 and not show_ignored:
            continue
        
        x1, y1 = ann['bbox_left'], ann['bbox_top']
        x2 = x1 + ann['bbox_width']
        y2 = y1 + ann['bbox_height']
        
        color = colors.get(ann['category'], (128, 128, 128))
        
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 显示类别
        label = f"ID:{ann['category']}"
        cv2.putText(img, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# 使用示例
img_path = "VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"
ann_path = "VisDrone2019-DET-val/annotations/0000001_02999_d_0000005.txt"
visualize_annotation(img_path, ann_path)