#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式标注格式转换和抠图工具
支持YOLO、VOC、COCO等格式转换为统一的COCO JSON格式
支持基于标注框的目标抠图功能

作者: gddi
版本: 2.0
"""

import os
import json
import sys
from PIL import Image
from xml.dom.minidom import parse
from tqdm import tqdm
from pathlib import Path
import warnings
from collections import defaultdict


class AnnotationConverter:
    """标注格式转换器"""
    
    def __init__(self):
        self.categories = []
        self.images = []
        self.annotations = []
        self.cat2id = {}
        self.img2id = {}
        self.img_count = 0
        self.cat_count = 0
        self.ann_count = 0
        
    def reset(self):
        """重置所有计数器和数据结构"""
        self.categories = []
        self.images = []
        self.annotations = []
        self.cat2id = {}
        self.img2id = {}
        self.img_count = 0
        self.cat_count = 0
        self.ann_count = 0
        
    @staticmethod
    def detect_format(input_path):
        """
        自动检测标注格式
        
        参数:
            input_path: 标注文件目录
            
        返回:
            检测到的格式: 'yolo', 'voc', 'coco_single', 'labelme', 'hk' 或 None
        """
        def _is_float(value):
            """检查字符串是否可以转换为浮点数"""
            try:
                float(value)
                return True
            except:
                return False
                
        if not os.path.exists(input_path):
            return None
            
        # 获取目录下的所有文件
        files = os.listdir(input_path)
        if not files:
            return None
            
        # 统计文件类型
        txt_count = sum(1 for f in files if f.endswith('.txt') and f != 'classes.txt')
        xml_count = sum(1 for f in files if f.endswith('.xml'))
        json_count = sum(1 for f in files if f.endswith('.json'))
        
        # 检查是否存在classes.txt
        has_classes_txt = 'classes.txt' in files
        
        # 尝试检查文件内容特征
        format_scores = {
            'yolo': 0,
            'voc': 0,
            'coco_single': 0,
            'labelme': 0,
            'hk': 0
        }
        
        # 采样检查一些文件(最多5个)
        sample_files = [f for f in files[:5] if not f.startswith('.')]
        
        for filename in sample_files:
            filepath = os.path.join(input_path, filename)
            
            # 检查txt文件是否符合YOLO格式
            if filename.endswith('.txt') and filename != 'classes.txt':
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        line = f.readline().strip()
                        if line:  # 不是空文件
                            # YOLO格式: class_id x_center y_center width height
                            parts = line.split()
                            if len(parts) == 5 and all(_is_float(p) for p in parts):
                                format_scores['yolo'] += 1
                except:
                    pass
                    
            # 检查XML文件是否符合VOC格式
            elif filename.endswith('.xml'):
                try:
                    root = parse(filepath)
                    # 检查关键标签
                    if (root.getElementsByTagName('annotation') 
                        and root.getElementsByTagName('object')
                        and root.getElementsByTagName('bndbox')):
                        format_scores['voc'] += 1
                except:
                    pass
                    
            # 检查JSON文件格式
            elif filename.endswith('.json'):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 检查是否是LabelMe格式
                        if ('shapes' in data 
                            and 'imagePath' in data 
                            and isinstance(data['shapes'], list)):
                            format_scores['labelme'] += 1
                        # 检查是否是hk json格式
                        elif ('imgName' in data 
                              and 'list' in data 
                              and isinstance(data['list'], list)):
                            # 进一步检查list中的项是否包含hk特有字段
                            if data['list'] and any('tagInfo' in item and 'tagName' in item for item in data['list']):
                                format_scores['hk'] += 1
                        # 检查是否包含典型的单图COCO字段
                        elif ('annotations' in data 
                              and isinstance(data['annotations'], list)):
                            format_scores['coco_single'] += 1
                except:
                    pass
                    
        # 根据特征判断格式
        if format_scores['labelme'] > 0:
            return 'labelme'
        elif format_scores['hk'] > 0:
            return 'hk'
        elif format_scores['yolo'] > 0 or (has_classes_txt and txt_count > 0):
            return 'yolo'
        elif format_scores['voc'] > 0 and xml_count > 0:
            return 'voc'
        elif format_scores['coco_single'] > 0 and json_count > 0:
            return 'coco_single'
            
        return None
        
    def match_files(self, annotation_path, image_path, format_type):
        """
        匹配标注文件和图片文件
        
        返回:
            matched_pairs: [(annotation_file, image_file), ...]
            unmatched_annotations: [annotation_file, ...]
            unmatched_images: [image_file, ...]
        """
        # 获取所有图片文件
        image_files = []
        for f in os.listdir(image_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(f)
        
        # 获取所有标注文件
        annotation_files = []
        if format_type == 'yolo':
            annotation_files = [f for f in os.listdir(annotation_path) 
                              if f.endswith('.txt') and f != 'classes.txt']
        elif format_type == 'voc':
            annotation_files = [f for f in os.listdir(annotation_path) if f.endswith('.xml')]
        elif format_type == 'coco_single':
            annotation_files = [f for f in os.listdir(annotation_path) if f.endswith('.json')]
        elif format_type == 'labelme':
            annotation_files = [f for f in os.listdir(annotation_path) if f.endswith('.json')]
        elif format_type == 'hk':
            annotation_files = [f for f in os.listdir(annotation_path) if f.endswith('.json')]
        
        matched_pairs = []
        unmatched_annotations = []
        unmatched_images = list(image_files)  # 初始化为所有图片
        
        for ann_file in annotation_files:
            if format_type in ['yolo', 'voc']:
                # 基于文件名匹配
                base_name = os.path.splitext(ann_file)[0]
                matched_img = None
                
                # 尝试不同的图片扩展名
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_name = base_name + ext
                    if img_name in image_files:
                        matched_img = img_name
                        break
                
                if matched_img:
                    matched_pairs.append((ann_file, matched_img))
                    unmatched_images.remove(matched_img)
                else:
                    unmatched_annotations.append(ann_file)
                    
            elif format_type == 'coco_single':
                # 对于单图COCO格式，需要读取JSON文件获取图片名
                try:
                    with open(os.path.join(annotation_path, ann_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 尝试从JSON中获取图片文件名
                    img_name = None
                    if 'image' in data and 'file_name' in data['image']:
                        img_name = data['image']['file_name']
                    else:
                        # 如果没有指定，使用相同的基础名称
                        base_name = os.path.splitext(ann_file)[0]
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                            test_name = base_name + ext
                            if test_name in image_files:
                                img_name = test_name
                                break
                    
                    if img_name and img_name in image_files:
                        matched_pairs.append((ann_file, img_name))
                        unmatched_images.remove(img_name)
                    else:
                        unmatched_annotations.append(ann_file)
                        
                except:
                    unmatched_annotations.append(ann_file)
                    
            elif format_type == 'labelme':
                # 对于LabelMe格式，需要读取JSON文件获取图片名
                try:
                    with open(os.path.join(annotation_path, ann_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 尝试从JSON中获取图片文件名
                    img_name = None
                    image_path = data.get('imagePath', '')
                    if image_path:
                        # 提取文件名（可能包含路径）
                        img_filename = os.path.basename(image_path)
                        
                        # 首先尝试原始文件名
                        if img_filename in image_files:
                            img_name = img_filename
                        else:
                            # 尝试不同的扩展名（忽略大小写）
                            base_name = os.path.splitext(img_filename)[0]
                            for img_file in image_files:
                                if os.path.splitext(img_file)[0].lower() == base_name.lower():
                                    img_name = img_file
                                    break
                    
                    if not img_name:
                        # 如果从imagePath找不到，尝试使用json文件的基础名称
                        base_name = os.path.splitext(ann_file)[0]
                        for img_file in image_files:
                            if os.path.splitext(img_file)[0].lower() == base_name.lower():
                                img_name = img_file
                                break
                    
                    if img_name and img_name in image_files:
                        matched_pairs.append((ann_file, img_name))
                        unmatched_images.remove(img_name)
                    else:
                        unmatched_annotations.append(ann_file)
                        
                except Exception as e:
                    print(f"读取LabelMe文件 {ann_file} 失败: {e}")
                    unmatched_annotations.append(ann_file)
                    
            elif format_type == 'hk':
                # 对于hk json格式，需要读取JSON文件获取图片名
                try:
                    with open(os.path.join(annotation_path, ann_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 从hk json中获取图片文件名
                    img_name = None
                    img_path = data.get('imgName', '')
                    if img_path:
                        # 提取文件名（可能包含路径）
                        img_filename = os.path.basename(img_path)
                        
                        # 首先尝试原始文件名
                        if img_filename in image_files:
                            img_name = img_filename
                        else:
                            # 尝试不同的扩展名（忽略大小写）
                            base_name = os.path.splitext(img_filename)[0]
                            for img_file in image_files:
                                if os.path.splitext(img_file)[0].lower() == base_name.lower():
                                    img_name = img_file
                                    break
                    
                    if not img_name:
                        # 如果从imgName找不到，尝试使用json文件的基础名称
                        base_name = os.path.splitext(ann_file)[0]
                        for img_file in image_files:
                            if os.path.splitext(img_file)[0].lower() == base_name.lower():
                                img_name = img_file
                                break
                    
                    if img_name and img_name in image_files:
                        matched_pairs.append((ann_file, img_name))
                        unmatched_images.remove(img_name)
                    else:
                        unmatched_annotations.append(ann_file)
                        
                except Exception as e:
                    print(f"读取hk json文件 {ann_file} 失败: {e}")
                    unmatched_annotations.append(ann_file)
        
        return matched_pairs, unmatched_annotations, unmatched_images
    
    def convert_to_coco(self, annotation_path, image_path, format_type, output_path, 
                       class_mapping=None, include_unmatched_images=False, area_threshold=25):
        """
        转换标注为COCO格式
        
        参数:
            annotation_path: 标注文件目录
            image_path: 图片目录
            format_type: 标注格式类型
            output_path: 输出JSON文件路径
            class_mapping: 类别映射字典 {id: name}
            include_unmatched_images: 是否包含没有标注的图片
        """
        self.reset()
        
        # 匹配文件
        matched_pairs, unmatched_annotations, unmatched_images = self.match_files(
            annotation_path, image_path, format_type)
        
        print(f"\n文件匹配结果:")
        print(f"- 成功匹配: {len(matched_pairs)} 对")
        print(f"- 未匹配的标注: {len(unmatched_annotations)} 个")
        print(f"- 未匹配的图片: {len(unmatched_images)} 个")
        
        if unmatched_annotations:
            print(f"\n未匹配的标注文件:")
            for f in unmatched_annotations[:5]:  # 只显示前5个
                print(f"  - {f}")
            if len(unmatched_annotations) > 5:
                print(f"  ... 还有 {len(unmatched_annotations) - 5} 个")
        
        if not matched_pairs:
            print("错误: 没有找到匹配的文件对!")
            return False
        
        # 转换匹配的文件对
        print(f"\n开始转换 {len(matched_pairs)} 对文件...")
        
        for ann_file, img_file in tqdm(matched_pairs, desc="转换进度"):
            try:
                self._process_file_pair(annotation_path, image_path, ann_file, img_file, 
                                      format_type, class_mapping)
            except Exception as e:
                print(f"处理文件对 ({ann_file}, {img_file}) 时出错: {e}")
                continue
        
        # 如果需要，添加未匹配的图片
        if include_unmatched_images and unmatched_images:
            print(f"\n添加 {len(unmatched_images)} 张未标注的图片...")
            for img_file in tqdm(unmatched_images, desc="添加图片"):
                try:
                    img_path = os.path.join(image_path, img_file)
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                    
                    self.images.append({
                        'file_name': img_file,
                        'id': self.img_count,
                        'width': img_w,
                        'height': img_h
                    })
                    self.img_count += 1
                except Exception as e:
                    print(f"处理图片 {img_file} 时出错: {e}")
                    continue
        
        # 保存结果
        self._save_coco(output_path, area_threshold)
        return True
    
    def _process_file_pair(self, annotation_path, image_path, ann_file, img_file, 
                          format_type, class_mapping):
        """处理单个文件对"""
        # 读取图片信息
        img_path = os.path.join(image_path, img_file)
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # 添加图片信息
        self.images.append({
            'file_name': img_file,
            'id': self.img_count,
            'width': img_w,
            'height': img_h
        })
        current_img_id = self.img_count
        self.img_count += 1
        
        # 根据格式处理标注
        ann_path = os.path.join(annotation_path, ann_file)
        
        if format_type == 'yolo':
            self._process_yolo_file(ann_path, current_img_id, img_w, img_h, class_mapping)
        elif format_type == 'voc':
            self._process_voc_file(ann_path, current_img_id)
        elif format_type == 'coco_single':
            self._process_coco_single_file(ann_path, current_img_id)
        elif format_type == 'labelme':
            self._process_labelme_file(ann_path, current_img_id)
        elif format_type == 'hk':
            self._process_hk_file(ann_path, current_img_id)
    
    def _process_yolo_file(self, ann_path, img_id, img_w, img_h, class_mapping):
        """处理YOLO标注文件"""
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) != 5:
                    continue
                    
                try:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    
                    # 转换为像素坐标
                    x_center *= img_w
                    y_center *= img_h
                    w *= img_w
                    h *= img_h
                    
                    # 转换为COCO格式的bbox [x, y, width, height]
                    x = int(x_center - w / 2)
                    y = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    # 获取类别名称
                    if class_mapping and class_id in class_mapping:
                        category_name = class_mapping[class_id]
                    else:
                        category_name = f"class_{class_id}"
                    
                    # 确保类别存在
                    if category_name not in self.cat2id:
                        self.cat2id[category_name] = self.cat_count
                        self.categories.append({
                            'id': self.cat_count,
                            'name': category_name
                        })
                        self.cat_count += 1
                    
                    # 添加标注
                    self.annotations.append({
                        'id': self.ann_count,
                        'image_id': img_id,
                        'category_id': self.cat2id[category_name],
                        'bbox': [x, y, w, h],
                        'area': w * h
                    })
                    self.ann_count += 1
                    
                except ValueError:
                    continue
    
    def _process_voc_file(self, ann_path, img_id):
        """处理VOC XML标注文件"""
        try:
            root = parse(ann_path)
            objects = root.getElementsByTagName('object')
            
            for obj in objects:
                try:
                    category_name = obj.getElementsByTagName('name')[0].firstChild.data
                    
                    # 获取边界框坐标
                    bndbox = obj.getElementsByTagName('bndbox')[0]
                    xmin = int(float(bndbox.getElementsByTagName('xmin')[0].firstChild.data))
                    ymin = int(float(bndbox.getElementsByTagName('ymin')[0].firstChild.data))
                    xmax = int(float(bndbox.getElementsByTagName('xmax')[0].firstChild.data))
                    ymax = int(float(bndbox.getElementsByTagName('ymax')[0].firstChild.data))
                    
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    # 确保类别存在
                    if category_name not in self.cat2id:
                        self.cat2id[category_name] = self.cat_count
                        self.categories.append({
                            'id': self.cat_count,
                            'name': category_name
                        })
                        self.cat_count += 1
                    
                    # 添加标注
                    self.annotations.append({
                        'id': self.ann_count,
                        'image_id': img_id,
                        'category_id': self.cat2id[category_name],
                        'bbox': [xmin, ymin, w, h],
                        'area': w * h
                    })
                    self.ann_count += 1
                    
                except Exception as e:
                    print(f"处理VOC对象时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析VOC文件时出错: {e}")
    
    def _process_coco_single_file(self, ann_path, img_id):
        """处理单图COCO JSON标注文件"""
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            for ann in annotations:
                try:
                    # 获取类别名称
                    category_name = ann.get('category_name')
                    if not category_name:
                        category_id = ann.get('category_id', 0)
                        category_name = f"class_{category_id}"
                    
                    # 获取边界框
                    bbox = ann.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    
                    x, y, w, h = map(int, bbox)
                    
                    # 确保类别存在
                    if category_name not in self.cat2id:
                        self.cat2id[category_name] = self.cat_count
                        self.categories.append({
                            'id': self.cat_count,
                            'name': category_name
                        })
                        self.cat_count += 1
                    
                    # 添加标注
                    self.annotations.append({
                        'id': self.ann_count,
                        'image_id': img_id,
                        'category_id': self.cat2id[category_name],
                        'bbox': [x, y, w, h],
                        'area': w * h
                    })
                    self.ann_count += 1
                    
                except Exception as e:
                    print(f"处理COCO标注时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析COCO文件时出错: {e}")
    
    def _process_labelme_file(self, ann_path, img_id):
        """处理LabelMe JSON标注文件"""
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            shapes = data.get('shapes', [])
            for shape in shapes:
                try:
                    # LabelMe的类别名称是 'label'
                    category_name = shape.get('label')
                    if not category_name:
                        continue
                    
                    # 获取边界框
                    points = shape.get('points', [])
                    if len(points) != 2:  # LabelMe的bbox是两个点 (x1, y1) 和 (x2, y2)
                        continue
                    
                    x1, y1 = map(int, points[0])
                    x2, y2 = map(int, points[1])
                    
                    # 计算宽高，确保坐标正确
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # 确保类别存在
                    if category_name not in self.cat2id:
                        self.cat2id[category_name] = self.cat_count
                        self.categories.append({
                            'id': self.cat_count,
                            'name': category_name
                        })
                        self.cat_count += 1
                    
                    # 添加标注
                    self.annotations.append({
                        'id': self.ann_count,
                        'image_id': img_id,
                        'category_id': self.cat2id[category_name],
                        'bbox': [x_min, y_min, w, h],
                        'area': w * h
                    })
                    self.ann_count += 1
                    
                except Exception as e:
                    print(f"处理LabelMe形状时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析LabelMe文件时出错: {e}")
    
    def _get_deepest_category_name(self, annotation_item):
        """
        获取最深层的类别名称
        如果存在children嵌套，递归获取最内层的tagName
        如果没有children，返回当前层的tagName
        """
        def find_deepest_tag(item):
            children = item.get('children', [])
            if not children:
                # 没有children，返回当前tagName
                return item.get('tagName', '')
            
            # 有children，递归查找每个child的最深层tagName
            deepest_tags = []
            for child in children:
                tag = find_deepest_tag(child)
                if tag:
                    deepest_tags.append(tag)
            
            # 如果找到了最深层的标签，返回第一个（或者可以返回所有，看需求）
            if deepest_tags:
                return deepest_tags[0]  # 返回第一个最深层标签
            else:
                # 如果children没有有效的tagName，返回当前层的tagName
                return item.get('tagName', '')
        
        return find_deepest_tag(annotation_item)
    
    def _process_hk_file(self, ann_path, img_id):
        """处理hk json标注文件"""
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = data.get('list', [])
            for ann in annotations:
                try:
                    # 获取类别名称（支持嵌套结构）
                    category_name = self._get_deepest_category_name(ann)
                    if not category_name:
                        continue
                    
                    # 解析tagInfo中的坐标信息
                    tag_info = ann.get('tagInfo', '')
                    if not tag_info:
                        continue
                    
                    # tagInfo是一个字符串化的JSON数组
                    try:
                        tag_data = json.loads(tag_info)
                        if not tag_data or not isinstance(tag_data, list):
                            continue
                        
                        coord_info = tag_data[0]  # 取第一个元素
                        coords = coord_info.get('coord', [])
                        if len(coords) != 2:  # 需要两个点：左上角和右下角
                            continue
                        
                        # 解析坐标点
                        x1, y1 = coords[0]['x'], coords[0]['y']
                        x2, y2 = coords[1]['x'], coords[1]['y']
                        
                        # 确保坐标正确（左上角和右下角）
                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        # 确保类别存在
                        if category_name not in self.cat2id:
                            self.cat2id[category_name] = self.cat_count
                            self.categories.append({
                                'id': self.cat_count,
                                'name': category_name
                            })
                            self.cat_count += 1
                        
                        # 添加标注
                        self.annotations.append({
                            'id': self.ann_count,
                            'image_id': img_id,
                            'category_id': self.cat2id[category_name],
                            'bbox': [x_min, y_min, w, h],
                            'area': w * h
                        })
                        self.ann_count += 1
                        
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"解析tagInfo失败: {e}")
                        continue
                        
                except Exception as e:
                    print(f"处理hk标注时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析hk json文件时出错: {e}")
    
    def _check_and_filter_small_boxes(self, area_threshold=25):
        """
        检查并过滤小面积标注框
        
        参数:
            area_threshold: 面积阈值，小于此值的标注将被删除，默认为25
        """
        if not self.annotations:
            return
        
        # 统计小面积标注框
        small_boxes = []
        for ann in self.annotations:
            bbox = ann['bbox']
            area = bbox[2] * bbox[3]  # width * height
            if area < area_threshold:
                small_boxes.append({
                    'id': ann['id'],
                    'image_id': ann['image_id'], 
                    'category_id': ann['category_id'],
                    'area': area,
                    'bbox': bbox
                })
        
        if not small_boxes:
            print(f"\n>> 质量检查: 没有发现面积小于{area_threshold}的标注框")
            return
        
        # 显示小面积标注框信息
        print(f"\n[质量检查] 发现 {len(small_boxes)} 个面积小于{area_threshold}的标注框:")
        
        # 按类别统计
        category_stats = defaultdict(int)
        for box in small_boxes:
            category_stats[box['category_id']] += 1
        
        print(f"按类别统计:")
        for cat_id, count in category_stats.items():
            cat_name = next((cat['name'] for cat in self.categories if cat['id'] == cat_id), f"class_{cat_id}")
            print(f"  - {cat_name}: {count} 个")
        
        print(f"\n面积分布:")
        areas = [box['area'] for box in small_boxes]
        areas.sort()
        print(f"  - 最小面积: {min(areas)}")
        print(f"  - 最大面积: {max(areas)}")
        print(f"  - 平均面积: {sum(areas)/len(areas):.1f}")
        
        # 自动删除小面积标注框 (平台要求)
        print(f"\n根据平台要求，自动删除这些小面积标注框...")
        self._remove_small_boxes(small_boxes, area_threshold)
    
    def _remove_small_boxes(self, small_boxes, area_threshold):
        """
        删除小面积标注框并处理相关数据
        
        参数:
            small_boxes: 要删除的小面积标注框列表
            area_threshold: 面积阈值
        """
        # 获取要删除的标注ID
        remove_ann_ids = {box['id'] for box in small_boxes}
        
        # 过滤标注
        valid_annotations = []
        for ann in self.annotations:
            if ann['id'] not in remove_ann_ids:
                valid_annotations.append(ann)
        
        # 统计每张图片的剩余标注数量
        image_ann_count = defaultdict(int)
        for ann in valid_annotations:
            image_ann_count[ann['image_id']] += 1
        
        # 过滤掉没有标注的图片
        valid_images = []
        removed_images = []
        image_id_map = {}  # 用于映射新旧image_id
        
        for img in self.images:
            if img['id'] in image_ann_count:
                # 更新图片id
                new_id = len(valid_images) + 1
                image_id_map[img['id']] = new_id
                img['id'] = new_id
                valid_images.append(img)
            else:
                removed_images.append(img['file_name'])
        
        # 更新annotations中的image_id和id
        updated_annotations = []
        for ann in valid_annotations:
            if ann['image_id'] in image_id_map:
                ann['image_id'] = image_id_map[ann['image_id']]
                ann['id'] = len(updated_annotations) + 1
                updated_annotations.append(ann)
        
        # 更新数据
        self.images = valid_images
        self.annotations = updated_annotations
        
        # 输出清理结果
        print(f"\n>> 质量检查完成 (平台要求: 面积≥{area_threshold}像素):")
        print(f"- 删除了 {len(small_boxes)} 个小面积标注 (面积<{area_threshold})")
        print(f"- 删除了 {len(removed_images)} 个没有标注的图片")
        print(f"- 保留了 {len(self.annotations)} 个有效标注")
        print(f"- 保留了 {len(self.images)} 个有效图片")
        
        if removed_images:
            print(f"\n被移除的图片:")
            for img_name in removed_images[:10]:  # 只显示前10个
                print(f"  - {img_name}")
            if len(removed_images) > 10:
                print(f"  ... 还有 {len(removed_images) - 10} 个")
    
    def _save_coco(self, output_path, area_threshold=25):
        """保存为COCO格式"""
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 检查并处理小面积标注框
        self._check_and_filter_small_boxes(area_threshold)
        
        # 保存JSON文件
        coco_data = {
            'categories': self.categories,
            'images': self.images,
            'annotations': self.annotations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n>> 转换完成！")
        print(f">> 输出文件: {output_path}")
        print(f"\n>> 统计信息:")
        print(f"- 图片数量: {len(self.images)}")
        print(f"- 类别数量: {len(self.categories)}")
        print(f"- 标注数量: {len(self.annotations)}")
        
        if self.categories:
            print(f"\n>> 类别列表:")
            for cat in self.categories:
                print(f"  {cat['id']}: {cat['name']}")


class ImageCropper:
    """图像抠图器"""
    
    def __init__(self):
        self.crop_count = 0
        self.category_counts = defaultdict(int)
    
    def crop_images(self, annotation_path, image_path, format_type, output_path, 
                   class_mapping=None, expansion_ratio=1.0):
        """
        基于标注框抠取图像
        
        参数:
            annotation_path: 标注文件目录
            image_path: 图片目录
            format_type: 标注格式类型
            output_path: 输出目录
            class_mapping: 类别映射字典 {id: name}
            expansion_ratio: 扩展比例，1.0表示不扩展，>1.0表示向外扩展
        """
        self.crop_count = 0
        self.category_counts = defaultdict(int)
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 获取匹配的文件对
        converter = AnnotationConverter()
        matched_pairs, unmatched_annotations, unmatched_images = converter.match_files(
            annotation_path, image_path, format_type)
        
        print(f"\n文件匹配结果:")
        print(f"- 成功匹配: {len(matched_pairs)} 对")
        print(f"- 未匹配的标注: {len(unmatched_annotations)} 个")
        print(f"- 未匹配的图片: {len(unmatched_images)} 个")
        
        if not matched_pairs:
            print("错误: 没有找到匹配的文件对!")
            return False
        
        # 处理匹配的文件对
        print(f"\n开始抠图处理 {len(matched_pairs)} 对文件...")
        
        for ann_file, img_file in tqdm(matched_pairs, desc="抠图进度"):
            try:
                self._process_image_crop(annotation_path, image_path, ann_file, img_file, 
                                       format_type, output_path, class_mapping, 
                                       expansion_ratio)
            except Exception as e:
                print(f"处理文件对 ({ann_file}, {img_file}) 时出错: {e}")
                continue
        
        # 输出统计信息
        self._print_crop_statistics(output_path)
        return True
    
    def _process_image_crop(self, annotation_path, image_path, ann_file, img_file, 
                           format_type, output_path, class_mapping, expansion_ratio):
        """处理单个图像的抠图"""
        # 读取图片
        img_path = os.path.join(image_path, img_file)
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # 获取标注信息
        ann_path = os.path.join(annotation_path, ann_file)
        
        if format_type == 'yolo':
            annotations = self._parse_yolo_annotations(ann_path, img_w, img_h, class_mapping)
        elif format_type == 'voc':
            annotations = self._parse_voc_annotations(ann_path)
        elif format_type == 'coco_single':
            annotations = self._parse_coco_single_annotations(ann_path)
        elif format_type == 'labelme':
            annotations = self._parse_labelme_annotations(ann_path, img_w, img_h)
        elif format_type == 'hk':
            annotations = self._parse_hk_annotations(ann_path)
        else:
            return
        
        # 基础文件名（不含扩展名）
        base_name = os.path.splitext(img_file)[0]
        
        # 处理每个标注框
        for i, ann in enumerate(annotations):
            try:
                category_name = ann['category']
                bbox = ann['bbox']  # [x, y, w, h]
                
                # 应用扩展比例和边界检查
                x, y, w, h = self._apply_expansion_and_bounds(bbox, img_w, img_h, expansion_ratio, 
                                                             base_name, i, category_name)
                
                # 裁剪图像
                cropped_img = img.crop((x, y, x + w, y + h))
                
                # 创建类别目录
                category_dir = os.path.join(output_path, category_name)
                os.makedirs(category_dir, exist_ok=True)
                
                # 生成输出文件名
                crop_filename = f"{base_name}_{i:03d}.jpg"
                crop_path = os.path.join(category_dir, crop_filename)
                
                # 保存裁剪图像
                cropped_img.save(crop_path, 'JPEG', quality=95)
                
                # 更新统计
                self.crop_count += 1
                self.category_counts[category_name] += 1
                
            except Exception as e:
                print(f"处理标注框 {i} 时出错: {e}")
                continue
    
    def _parse_yolo_annotations(self, ann_path, img_w, img_h, class_mapping):
        """解析YOLO标注文件"""
        annotations = []
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    
                    # 转换为像素坐标
                    x_center *= img_w
                    y_center *= img_h
                    w *= img_w
                    h *= img_h
                    
                    # 转换为bbox格式 [x, y, width, height]
                    x = int(x_center - w / 2)
                    y = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    # 获取类别名称
                    if class_mapping and class_id in class_mapping:
                        category_name = class_mapping[class_id]
                    else:
                        category_name = f"class_{class_id}"
                    
                    annotations.append({
                        'category': category_name,
                        'bbox': [x, y, w, h]
                    })
                    
                except ValueError:
                    continue
        
        return annotations
    
    def _parse_voc_annotations(self, ann_path):
        """解析VOC XML标注文件"""
        annotations = []
        
        try:
            root = parse(ann_path)
            objects = root.getElementsByTagName('object')
            
            for obj in objects:
                try:
                    category_name = obj.getElementsByTagName('name')[0].firstChild.data
                    
                    # 获取边界框坐标
                    bndbox = obj.getElementsByTagName('bndbox')[0]
                    xmin = int(float(bndbox.getElementsByTagName('xmin')[0].firstChild.data))
                    ymin = int(float(bndbox.getElementsByTagName('ymin')[0].firstChild.data))
                    xmax = int(float(bndbox.getElementsByTagName('xmax')[0].firstChild.data))
                    ymax = int(float(bndbox.getElementsByTagName('ymax')[0].firstChild.data))
                    
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    annotations.append({
                        'category': category_name,
                        'bbox': [xmin, ymin, w, h]
                    })
                    
                except Exception as e:
                    print(f"处理VOC对象时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析VOC文件时出错: {e}")
        
        return annotations
    
    def _parse_coco_single_annotations(self, ann_path):
        """解析单图COCO JSON标注文件"""
        annotations = []
        
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            anns = data.get('annotations', [])
            for ann in anns:
                try:
                    # 获取类别名称
                    category_name = ann.get('category_name')
                    if not category_name:
                        category_id = ann.get('category_id', 0)
                        category_name = f"class_{category_id}"
                    
                    # 获取边界框
                    bbox = ann.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    
                    x, y, w, h = map(int, bbox)
                    
                    annotations.append({
                        'category': category_name,
                        'bbox': [x, y, w, h]
                    })
                    
                except Exception as e:
                    print(f"处理COCO标注时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析COCO文件时出错: {e}")
        
        return annotations
    
    def _parse_labelme_annotations(self, ann_path, img_w, img_h):
        """解析LabelMe JSON标注文件"""
        annotations = []
        
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            shapes = data.get('shapes', [])
            for shape in shapes:
                try:
                    # LabelMe的类别名称是 'label'
                    category_name = shape.get('label')
                    if not category_name:
                        continue
                    
                    # 获取边界框
                    points = shape.get('points', [])
                    if len(points) != 2:  # LabelMe的bbox是两个点 (x1, y1) 和 (x2, y2)
                        continue
                    
                    x1, y1 = map(int, points[0])
                    x2, y2 = map(int, points[1])
                    
                    # 计算宽高，确保坐标正确
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    annotations.append({
                        'category': category_name,
                        'bbox': [x_min, y_min, w, h]
                    })
                    
                except Exception as e:
                    print(f"处理LabelMe形状时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析LabelMe文件时出错: {e}")
        
        return annotations
    
    def _get_deepest_category_name_for_cropper(self, annotation_item):
        """
        获取最深层的类别名称（抠图用）
        如果存在children嵌套，递归获取最内层的tagName
        如果没有children，返回当前层的tagName
        """
        def find_deepest_tag(item):
            children = item.get('children', [])
            if not children:
                # 没有children，返回当前tagName
                return item.get('tagName', '')
            
            # 有children，递归查找每个child的最深层tagName
            deepest_tags = []
            for child in children:
                tag = find_deepest_tag(child)
                if tag:
                    deepest_tags.append(tag)
            
            # 如果找到了最深层的标签，返回第一个
            if deepest_tags:
                return deepest_tags[0]  # 返回第一个最深层标签
            else:
                # 如果children没有有效的tagName，返回当前层的tagName
                return item.get('tagName', '')
        
        return find_deepest_tag(annotation_item)
    
    def _parse_hk_annotations(self, ann_path):
        """解析hk json标注文件"""
        annotations = []
        
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            anns = data.get('list', [])
            for ann in anns:
                try:
                    # 获取类别名称（支持嵌套结构）
                    category_name = self._get_deepest_category_name_for_cropper(ann)
                    if not category_name:
                        continue
                    
                    # 解析tagInfo中的坐标信息
                    tag_info = ann.get('tagInfo', '')
                    if not tag_info:
                        continue
                    
                    # tagInfo是一个字符串化的JSON数组
                    try:
                        tag_data = json.loads(tag_info)
                        if not tag_data or not isinstance(tag_data, list):
                            continue
                        
                        coord_info = tag_data[0]  # 取第一个元素
                        coords = coord_info.get('coord', [])
                        if len(coords) != 2:  # 需要两个点：左上角和右下角
                            continue
                        
                        # 解析坐标点
                        x1, y1 = coords[0]['x'], coords[0]['y']
                        x2, y2 = coords[1]['x'], coords[1]['y']
                        
                        # 确保坐标正确（左上角和右下角）
                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        annotations.append({
                            'category': category_name,
                            'bbox': [x_min, y_min, w, h]
                        })
                        
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"解析tagInfo失败: {e}")
                        continue
                        
                except Exception as e:
                    print(f"处理hk标注时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"解析hk json文件时出错: {e}")
        
        return annotations
    
    def _apply_expansion_and_bounds(self, bbox, img_w, img_h, expansion_ratio, 
                                   base_name, ann_idx, category_name):
        """应用扩展比例并确保在图像边界内"""
        x, y, w, h = bbox
        
        # 如果扩展比例为1.0，直接返回原始bbox
        if expansion_ratio == 1.0:
            return x, y, w, h
        
        # 计算扩展后的尺寸
        new_w = int(w * expansion_ratio)
        new_h = int(h * expansion_ratio)
        
        # 计算扩展后的左上角坐标（保持中心不变）
        center_x = x + w // 2
        center_y = y + h // 2
        new_x = center_x - new_w // 2
        new_y = center_y - new_h // 2
        
        # 检查是否超出边界
        out_of_bounds = False
        
        # 检查左边界
        if new_x < 0:
            new_x = 0
            out_of_bounds = True
        
        # 检查上边界  
        if new_y < 0:
            new_y = 0
            out_of_bounds = True
        
        # 检查右边界
        if new_x + new_w > img_w:
            new_w = img_w - new_x
            out_of_bounds = True
        
        # 检查下边界
        if new_y + new_h > img_h:
            new_h = img_h - new_y
            out_of_bounds = True
        
        # 如果超出边界，打印告警信息
        if out_of_bounds:
            print(f"告警: {base_name}_{ann_idx:03d}({category_name}) 扩展后超出图像边界，已调整到边界范围内")
        
        return new_x, new_y, new_w, new_h
    
    def _print_crop_statistics(self, output_path):
        """打印抠图统计信息"""
        print(f"\n>> 抠图完成！")
        print(f">> 输出目录: {output_path}")
        print(f"\n>> 统计信息:")
        print(f"- 总抠图数量: {self.crop_count}")
        print(f"- 类别数量: {len(self.category_counts)}")
        
        if self.category_counts:
            print(f"\n>> 各类别抠图统计:")
            for category, count in sorted(self.category_counts.items()):
                print(f"  {category}: {count} 张")


class InteractiveConverter:
    """交互式转换器"""
    
    def __init__(self):
        self.converter = AnnotationConverter()
        self.cropper = ImageCropper()
    
    def run(self):
        """运行交互式程序"""
        print("=" * 60)
        print("         标注处理工具集")
        print("    支持格式转换和目标抠图功能")
        print("=" * 60)
        
        while True:
            try:
                # 显示主菜单
                choice = self.show_main_menu()
                
                if choice == '1':
                    self.run_conversion()
                elif choice == '2':
                    self.run_cropping()
                elif choice == '3':
                    print(f"\n感谢使用！")
                    break
                else:
                    print(f"!! 无效选择，请重试")
                    continue
                    
                # 询问是否继续
                if not self.ask_continue():
                    print(f"\n感谢使用！")
                    break
                    
            except KeyboardInterrupt:
                print(f"\n\n用户取消操作")
                break
            except Exception as e:
                print(f"\n!! 发生错误: {e}")
                if not self.ask_continue():
                    break
    
    def show_main_menu(self):
        """显示主菜单"""
        print(f"\n" + "="*40)
        print(f"           主菜单")
        print(f"="*40)
        print(f"1. 标注格式转换 (转换为平台格式)")
        print(f"2. 目标图像抠图 (基于标注框)")
        print(f"3. 退出程序")
        print(f"="*40)
        
        choice = input(f"请选择功能 (1-3): ").strip()
        return choice
    
    def ask_continue(self):
        """询问是否继续使用"""
        print(f"\n" + "-"*40)
        while True:
            choice = input(f"是否继续使用? (y/n): ").lower()
            if choice in ['y', 'yes', '是']:
                return True
            elif choice in ['n', 'no', '否']:
                return False
            else:
                print("请输入 y 或 n")
    
    def run_conversion(self):
        """运行标注转换功能"""
        print(f"\n" + "="*50)
        print(f"           标注格式转换")
        print(f"    支持目标检测和分割算法的标注转换")
        print(f"="*50)
        
        # 选择算法类型
        algorithm_type = self.select_algorithm_type()
        if not algorithm_type:
            return
        
        if algorithm_type == 'detection':
            self.run_detection_conversion()
        elif algorithm_type == 'segmentation':
            self.run_segmentation_conversion()
    
    def select_algorithm_type(self):
        """选择算法类型"""
        print(f"\n请选择转换类型:")
        print(f"1. 目标检测 (转换为COCO JSON格式)")
        print(f"2. 分割算法 (YOLO分割转换为mask)")
        print(f"3. 返回主菜单")
        
        while True:
            choice = input(f"请选择 (1-3): ").strip()
            if choice == '1':
                return 'detection'
            elif choice == '2':
                return 'segmentation'
            elif choice == '3':
                return None
            else:
                print("!! 无效选择，请输入 1、2 或 3")
    
    def run_detection_conversion(self):
        """运行目标检测转换功能"""
        print(f"\n" + "="*40)
        print(f"        目标检测标注转换")
        print(f"  支持 YOLO/VOC/COCO 转换为统一COCO格式")
        print(f"="*40)
        
        # 获取输入路径
        image_path = self.get_image_path()
        annotation_path = self.get_annotation_path()
        
        # 检测格式
        format_type = self.detect_and_confirm_format(annotation_path)
        if not format_type:
            return
        
        # 处理类别映射（YOLO格式需要）
        class_mapping = None
        if format_type == 'yolo':
            class_mapping = self.get_yolo_class_mapping(annotation_path)
        
        # 获取输出路径
        output_path = self.get_output_path(image_path, 'detection')
        
        # 询问是否包含未标注图片
        include_unmatched = self.ask_include_unmatched_images()
        
        # 确认信息
        self.confirm_conversion_info(annotation_path, image_path, format_type, 
                                   output_path, class_mapping, include_unmatched)
        
        # 执行转换
        success = self.converter.convert_to_coco(
            annotation_path=annotation_path,
            image_path=image_path,
            format_type=format_type,
            output_path=output_path,
            class_mapping=class_mapping,
            include_unmatched_images=include_unmatched
        )
        
        if success:
            print(f"\n>> 转换成功完成!")
        else:
            print(f"\n!! 转换失败!")
    
    def run_segmentation_conversion(self):
        """运行分割标注转换功能"""
        print(f"\n" + "="*40)
        print(f"        分割标注转换")
        print(f"  YOLO分割格式转换为mask图像")
        print(f"="*40)
        
        # 获取输入路径
        image_path = self.get_image_path()
        annotation_path = self.get_annotation_path()
        
        # 获取输出路径
        output_path = self.get_output_path(image_path, 'segmentation')
        
        # 确认信息
        self.confirm_segmentation_info(annotation_path, image_path, output_path)
        
        # 执行分割转换
        success = self.run_segmentation_conversion_process(annotation_path, image_path, output_path)
        
        if success:
            print(f"\n>> 分割转换成功完成!")
        else:
            print(f"\n!! 分割转换失败!")
    
    def confirm_segmentation_info(self, annotation_path, image_path, output_path):
        """确认分割转换信息"""
        print(f"\n" + "="*50)
        print(f"          分割转换信息确认")
        print(f"="*50)
        print(f"标注目录: {annotation_path}")
        print(f"图片目录: {image_path}")
        print(f"输出目录: {output_path}")
        print(f"="*50)
        
        user_input = input(f"\n确认开始转换（回车确认，输入n取消）: ").lower().strip()
        if user_input == 'n':
            raise KeyboardInterrupt("用户取消转换")
    
    def run_segmentation_conversion_process(self, annotation_path, image_path, output_path):
        """执行分割转换过程"""
        try:
            # 统计变量
            successful_count = 0
            txt_no_image_list = []
            image_no_txt_list = []
            
            # 获取所有txt文件
            txt_files = [f for f in os.listdir(annotation_path) if f.lower().endswith('.txt')]
            
            print(f"\n共发现{len(txt_files)}个txt文件，开始处理...")
            
            for txt_name in tqdm(txt_files, desc="转换进度"):
                txt_path = os.path.join(annotation_path, txt_name)
                
                # 查找对应图片
                img_path = self._find_matching_image(txt_name, image_path)
                if img_path is None:
                    txt_no_image_list.append(txt_name)
                    continue
                    
                # 获取图片尺寸
                width, height = self._get_image_size(img_path)
                if width is None or height is None:
                    txt_no_image_list.append(txt_name)
                    continue
                    
                # 生成mask
                mask_name = os.path.splitext(txt_name)[0] + ".png"
                mask_path = os.path.join(output_path, mask_name)
                
                if self._txt_to_mask(txt_path, mask_path, width, height):
                    successful_count += 1

            # 检查有图片但没有txt的情况
            img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            image_files = [f for f in os.listdir(image_path) 
                           if os.path.splitext(f)[1].lower() in img_exts]
            
            for img_name in image_files:
                base_name = os.path.splitext(img_name)[0]
                txt_name = base_name + '.txt'
                txt_path = os.path.join(annotation_path, txt_name)
                if not os.path.exists(txt_path):
                    image_no_txt_list.append(img_name)

            # 输出统计信息
            print(f"\n=== 处理统计 ===")
            print(f"成功处理: {successful_count} 对")
            print(f"有txt无图片: {len(txt_no_image_list)} 个")
            if txt_no_image_list:
                print(f"  详细: {', '.join(txt_no_image_list[:5])}{'...' if len(txt_no_image_list) > 5 else ''}")
            print(f"有图片无txt: {len(image_no_txt_list)} 个")
            if image_no_txt_list:
                print(f"  详细: {', '.join(image_no_txt_list[:5])}{'...' if len(image_no_txt_list) > 5 else ''}")
            print(f"总txt文件: {len(txt_files)} 个")
            print(f"总图片文件: {len(image_files)} 个")

            # 询问是否清理多余文件
            if len(txt_no_image_list) > 0 or len(image_no_txt_list) > 0:
                print(f"\n检测到{len(txt_no_image_list) + len(image_no_txt_list)}个不匹配文件")
                choice = input("是否清理多余文件（回车跳过，输入y确认清理）: ").strip().lower()
                
                if choice == 'y' or choice == 'yes':
                    self._cleanup_unmatched_files(annotation_path, image_path, txt_no_image_list, image_no_txt_list)
            else:
                print("\n所有文件匹配完美，无需清理。")
            
            return True
            
        except Exception as e:
            print(f"分割转换过程出错: {e}")
            return False
    
    def _find_matching_image(self, txt_name, images_dir):
        """根据txt文件名查找对应的图片"""
        base_name = os.path.splitext(txt_name)[0]
        img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        for ext in img_exts:
            img_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(img_path):
                return img_path
        return None
    
    def _get_image_size(self, image_path):
        """获取图片尺寸"""
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"读取图片尺寸失败: {image_path}，错误: {e}")
            return None, None
    
    def _parse_polygon_line(self, line, width, height):
        """解析一行txt，返回类别和像素坐标多边形点"""
        parts = line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            # 至少1个类别+3个点（6个数），且总数为奇数
            return None, None
        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
        except Exception:
            return None, None
        
        pts = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * width))
            y = int(round(coords[i+1] * height))
            x = min(max(x, 0), width-1)
            y = min(max(y, 0), height-1)
            pts.append([x, y])
        
        if len(pts) < 3:
            return None, None  # 不是多边形
        return cls, pts
    
    def _txt_to_mask(self, txt_path, mask_path, width, height, max_cls=254):
        """将单个txt文件转为mask png"""
        from PIL import ImageDraw
        
        # 创建mask图像（灰度模式）
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    cls, pts = self._parse_polygon_line(line, width, height)
                    if pts is None or cls is None:
                        continue
                    if cls < 0 or cls > max_cls:
                        continue
                    gray = cls + 1  # 类别0->1，1->2，依次递增
                    
                    # 将坐标列表转换为PIL可用的坐标元组
                    polygon_coords = [(int(pt[0]), int(pt[1])) for pt in pts]
                    
                    # 使用PIL绘制填充多边形
                    draw.polygon(polygon_coords, fill=gray)
            
            # 保存mask图像
            mask_img.save(mask_path)
            return True
        except Exception as e:
            print(f"处理文件失败: {txt_path}，错误: {e}")
            return False
    
    def _cleanup_unmatched_files(self, annotation_path, image_path, txt_no_image_list, image_no_txt_list):
        """清理不匹配的文件"""
        deleted_count = 0
        
        # 删除无图片的txt文件
        for txt_name in txt_no_image_list:
            txt_path = os.path.join(annotation_path, txt_name)
            try:
                os.remove(txt_path)
                deleted_count += 1
                print(f"已删除txt: {txt_name}")
            except Exception as e:
                print(f"删除txt失败: {txt_name}，错误: {e}")
        
        # 删除无txt的图片文件
        for img_name in image_no_txt_list:
            img_path = os.path.join(image_path, img_name)
            try:
                os.remove(img_path)
                deleted_count += 1
                print(f"已删除图片: {img_name}")
            except Exception as e:
                print(f"删除图片失败: {img_name}，错误: {e}")
        
        print(f"\n清理完成，共删除{deleted_count}个文件")
    
    def run_cropping(self):
        """运行抠图功能"""
        print(f"\n" + "="*50)
        print(f"           目标图像抠图")
        print(f"    基于检测标注框抠取目标区域")
        print(f"="*50)
        
        # 获取输入路径
        image_path = self.get_image_path()
        annotation_path = self.get_annotation_path()
        
        # 检测格式
        format_type = self.detect_and_confirm_format(annotation_path)
        if not format_type:
            return
        
        # 处理类别映射（YOLO格式需要）
        class_mapping = None
        if format_type == 'yolo':
            class_mapping = self.get_yolo_class_mapping(annotation_path)
        
        # 获取输出目录
        output_path = self.get_output_path(image_path, 'cropping')
        
        # 获取抠图参数
        expansion_ratio = self.get_crop_parameters()
        
        # 确认信息
        self.confirm_cropping_info(annotation_path, image_path, format_type, 
                                 output_path, class_mapping, expansion_ratio)
        
        # 执行抠图
        success = self.cropper.crop_images(
            annotation_path=annotation_path,
            image_path=image_path,
            format_type=format_type,
            output_path=output_path,
            class_mapping=class_mapping,
            expansion_ratio=expansion_ratio
        )
        
        if success:
            print(f"\n>> 抠图成功完成!")
        else:
            print(f"\n!! 抠图失败!")

    def get_image_path(self):
        """获取图片路径"""
        while True:
            print(f"\n[图片目录] 请输入图片目录路径:")
            path = input(">>> ").strip().strip('"\'')
            
            if not path:
                print("!! 路径不能为空!")
                continue
            
            if not os.path.exists(path):
                print(f"!! 路径不存在: {path}")
                continue
            
            if not os.path.isdir(path):
                print(f"!! 不是有效的目录: {path}")
                continue
            
            # 检查是否有图片文件
            image_files = [f for f in os.listdir(path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                print(f"!! 目录中没有找到图片文件 (.jpg, .jpeg, .png, .bmp)")
                continue
            
            print(f">> 找到 {len(image_files)} 张图片")
            return os.path.abspath(path)
    
    def get_annotation_path(self):
        """获取标注路径"""
        while True:
            print(f"\n[标注目录] 请输入标注文件目录路径:")
            path = input(">>> ").strip().strip('"\'')
            
            if not path:
                print("!! 路径不能为空!")
                continue
            
            if not os.path.exists(path):
                print(f"!! 路径不存在: {path}")
                continue
            
            if not os.path.isdir(path):
                print(f"!! 不是有效的目录: {path}")
                continue
            
            # 检查是否有标注文件
            ann_files = [f for f in os.listdir(path) 
                        if f.endswith(('.txt', '.xml', '.json'))]
            
            if not ann_files:
                print(f"!! 目录中没有找到标注文件 (.txt, .xml, .json)")
                continue
            
            print(f">> 找到 {len(ann_files)} 个标注文件")
            return os.path.abspath(path)
    
    def detect_and_confirm_format(self, annotation_path):
        """检测并确认格式"""
        print(f"\n[格式检测] 正在检测标注格式...")
        
        detected_format = AnnotationConverter.detect_format(annotation_path)
        
        if detected_format:
            format_names = {
                'yolo': 'YOLO (txt格式)',
                'voc': 'VOC (xml格式)', 
                'coco_single': '单图COCO (json格式)',
                'labelme': 'LabelMe (json格式)',
                'hk': 'HK json (json格式)'
            }
            print(f">> 检测到格式: {format_names[detected_format]}")
            
            confirm = input(f"确认格式正确（回车确认，输入n手动选择）: ").lower().strip()
            if confirm != 'n':
                return detected_format
        
        # 手动选择格式
        print(f"\n请手动选择标注格式:")
        print(f"1. YOLO格式 (.txt文件)")
        print(f"2. VOC格式 (.xml文件)")
        print(f"3. 单图COCO格式 (.json文件)")
        print(f"4. LabelMe格式 (.json文件)")
        print(f"5. HK json格式 (.json文件)")
        
        while True:
            choice = input(f"请选择 (1-5): ").strip()
            if choice == '1':
                return 'yolo'
            elif choice == '2':
                return 'voc'
            elif choice == '3':
                return 'coco_single'
            elif choice == '4':
                return 'labelme'
            elif choice == '5':
                return 'hk'
            else:
                print("!! 无效选择，请输入 1、2、3、4 或 5")
    
    def get_yolo_class_mapping(self, annotation_path):
        """获取YOLO类别映射"""
        print(f"\n[类别映射] YOLO格式需要类别映射信息")
        
        # 首先尝试查找classes.txt
        classes_file = os.path.join(annotation_path, 'classes.txt')
        if os.path.exists(classes_file):
            print(f">> 在标注目录下找到classes.txt文件")
            return self._load_classes_file(classes_file)
        
        # 询问用户是否有classes.txt文件
        print(f"!! 在标注目录下未找到classes.txt文件")
        print(f"请选择类别映射方式:")
        print(f"1. 指定classes.txt文件路径")
        print(f"2. 手动输入类别映射")
        print(f"3. 使用默认名称 (class_0, class_1, ...)")
        
        while True:
            choice = input(f"请选择 (1-3): ").strip()
            if choice == '1':
                return self._get_classes_file_path()
            elif choice == '2':
                return self._manual_input_classes()
            elif choice == '3':
                print(">> 将使用默认类别名称")
                return None
            else:
                print("!! 无效选择，请输入 1、2 或 3")
    
    def _load_classes_file(self, classes_file):
        """加载classes.txt文件"""
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
            
            print(f"类别列表 ({len(classes)} 个):")
            for i, name in enumerate(classes):
                print(f"  {i}: {name}")
            
            confirm = input(f"\n使用此类别映射（回车确认，输入n跳过）: ").lower().strip()
            if confirm != 'n':
                return {i: name for i, name in enumerate(classes)}
            else:
                return None
        except Exception as e:
            print(f"!! 读取classes.txt失败: {e}")
            return None
    
    def _get_classes_file_path(self):
        """获取classes.txt文件路径"""
        while True:
            print(f"\n请输入classes.txt文件路径:")
            path = input(">>> ").strip().strip('"\'')
            
            if not path:
                print("!! 路径不能为空!")
                continue
            
            if not os.path.exists(path):
                print(f"!! 文件不存在: {path}")
                continue
            
            if not os.path.isfile(path):
                print(f"!! 不是有效的文件: {path}")
                continue
            
            if not path.lower().endswith('.txt'):
                print(f"!! 请选择txt格式文件")
                continue
            
            return self._load_classes_file(path)
    
    def _manual_input_classes(self):
        """手动输入类别映射"""
        print(f"\n请手动输入类别映射关系:")
        print(f"格式: id:类别名称 (例如: 0:person)")
        print(f"每行一个，输入空行结束")
        
        class_mapping = {}
        while True:
            line = input(">>> ").strip()
            if not line:
                break
            
            try:
                if ':' in line:
                    id_str, name = line.split(':', 1)
                    class_id = int(id_str.strip())
                    class_name = name.strip()
                    class_mapping[class_id] = class_name
                    print(f">> 添加映射: {class_id} -> {class_name}")
                else:
                    print("!! 格式错误，请使用 id:名称 格式")
            except ValueError:
                print("!! ID必须是数字")
        
        if not class_mapping:
            print("!! 未提供类别映射，将使用默认名称 (class_0, class_1, ...)")
            return None
        
        return class_mapping
    
    def get_output_path(self, image_path, conversion_type):
        """获取输出路径"""
        if conversion_type == 'detection':
            # 目标检测转换的默认输出路径
            default_output = os.path.join(os.path.dirname(image_path), 'train.json')
            print(f"\n[输出设置]")
            user_input = input(f"默认输出路径: {default_output}: ").strip().strip('"\'')
            
            if not user_input:
                return default_output
            
            # 用户自定义路径
            if not user_input.endswith('.json'):
                user_input += '.json'
            
            # 检查输出目录是否存在
            output_dir = os.path.dirname(os.path.abspath(user_input))
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    print(f">> 创建输出目录: {output_dir}")
                except Exception as e:
                    print(f"!! 无法创建目录: {e}")
                    return default_output
            
            return os.path.abspath(user_input)
            
        elif conversion_type == 'segmentation':
            # 分割转换的默认输出目录
            default_output = os.path.join(os.path.dirname(image_path), 'mask')
            print(f"\n[输出设置]")
            user_input = input(f"默认输出目录: {default_output}: ").strip().strip('"\'')
            
            if not user_input:
                output_path = default_output
            else:
                output_path = os.path.abspath(user_input)
            
            # 检查并创建目录
            try:
                os.makedirs(output_path, exist_ok=True)
                print(f">> 输出目录: {output_path}")
                return output_path
            except Exception as e:
                print(f"!! 无法创建目录: {e}")
                return default_output
                
        elif conversion_type == 'cropping':
            # 抠图功能的默认输出目录
            default_output = os.path.join(os.path.dirname(image_path), 'cropped_images')
            print(f"\n[输出设置]")
            user_input = input(f"默认输出目录: {default_output}: ").strip().strip('"\'')
            
            if not user_input:
                output_path = default_output
            else:
                output_path = os.path.abspath(user_input)
            
            # 检查并创建目录
            try:
                os.makedirs(output_path, exist_ok=True)
                print(f">> 输出目录: {output_path}")
                return output_path
            except Exception as e:
                print(f"!! 无法创建目录: {e}")
                return default_output
        else:
            # 未知类型，使用默认设置
            default_output = os.path.join(os.path.dirname(image_path), 'output')
            print(f"\n[输出设置] 未知转换类型，使用默认设置")
            user_input = input(f"默认输出目录: {default_output}: ").strip().strip('"\'')
            
            if not user_input:
                output_path = default_output
            else:
                output_path = os.path.abspath(user_input)
            
            try:
                os.makedirs(output_path, exist_ok=True)
                return output_path
            except Exception as e:
                print(f"!! 无法创建目录: {e}")
                return default_output
    
    def get_crop_parameters(self):
        """获取抠图参数"""
        print(f"\n[抠图参数设置]")
        
        # 扩展比例设置
        while True:
            try:
                expansion_input = input(f"扩展比例(默认1.0，不扩展): ").strip()
                if not expansion_input:
                    expansion_ratio = 1.0
                    break
                expansion_ratio = float(expansion_input)
                if expansion_ratio <= 0:
                    print("!! 扩展比例必须大于0")
                    continue
                break
            except ValueError:
                print("!! 请输入有效的数字")
        
        print(f"\n>> 抠图参数:")
        print(f"- 扩展比例: {expansion_ratio} ({'不扩展' if expansion_ratio == 1.0 else f'向外扩展{(expansion_ratio-1)*100:.1f}%'})")
        
        return expansion_ratio
    
    def confirm_cropping_info(self, annotation_path, image_path, format_type, 
                            output_path, class_mapping, expansion_ratio):
        """确认抠图信息"""
        print(f"\n" + "="*50)
        print(f"          抠图信息确认")
        print(f"="*50)
        print(f"标注目录: {annotation_path}")
        print(f"图片目录: {image_path}")
        print(f"输入格式: {format_type.upper()}")
        print(f"输出目录: {output_path}")
        print(f"扩展比例: {expansion_ratio} ({'不扩展' if expansion_ratio == 1.0 else f'向外扩展{(expansion_ratio-1)*100:.1f}%'})")
        
        if class_mapping:
            print(f"类别映射: {len(class_mapping)} 个类别")
            for class_id, name in sorted(class_mapping.items()):
                print(f"  {class_id}: {name}")
        
        print(f"="*50)
        
        user_input = input(f"\n确认开始抠图（回车确认，输入n取消）: ").lower().strip()
        if user_input == 'n':
            raise KeyboardInterrupt("用户取消抠图")
    
    def ask_include_unmatched_images(self):
        """询问是否包含未匹配的图片"""
        print(f"\n[图片选项] 是否包含没有标注的图片?")
        print(f"   (选择'是'会将所有图片都包含在输出中，即使它们没有标注)")
        
        choice = input(f"包含未标注图片（回车跳过，输入y包含）: ").lower().strip()
        return choice in ['y', 'yes', '是']
    

    
    def confirm_conversion_info(self, annotation_path, image_path, format_type, 
                               output_path, class_mapping, include_unmatched):
        """确认转换信息"""
        print(f"\n" + "="*50)
        print(f"          转换信息确认")
        print(f"="*50)
        print(f"标注目录: {annotation_path}")
        print(f"图片目录: {image_path}")
        print(f"输入格式: {format_type.upper()}")
        print(f"输出文件: {output_path}")
        print(f"包含未标注图片: {'是' if include_unmatched else '否'}")
        print(f"质量检查: 自动删除面积小于25像素的标注框 (平台要求)")
        
        if class_mapping:
            print(f"类别映射: {len(class_mapping)} 个类别")
            for class_id, name in sorted(class_mapping.items()):
                print(f"  {class_id}: {name}")
        
        print(f"="*50)
        
        user_input = input(f"\n确认开始转换（回车确认，输入n取消）: ").lower().strip()
        if user_input == 'n':
            raise KeyboardInterrupt("用户取消转换")


def main():
    """主函数"""
    try:
        converter = InteractiveConverter()
        converter.run()
    except KeyboardInterrupt:
        print(f"\n\n程序已退出")
    except Exception as e:
        print(f"\n程序发生错误: {e}")
    
    input(f"\n按回车键退出...")


if __name__ == "__main__":
    main() 