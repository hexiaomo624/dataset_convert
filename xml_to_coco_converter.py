#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML格式标签转COCO格式标签转换器
支持SAR飞机检测数据集，按8:2比例分割训练集和测试集
"""

import os
import glob
import json
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Any
import argparse


class XMLToCOCOConverter:
    def __init__(self, xml_dir: str, image_dir: str, output_dir: str = "."):
        """
        初始化转换器
        
        Args:
            xml_dir: XML标注文件目录
            image_dir: 图片文件目录
            output_dir: 输出目录
        """
        self.xml_dir = xml_dir
        self.image_dir = image_dir
        self.output_dir = output_dir
        
        # 定义飞机类别
        self.categories = {
            'A220': 1,
            'A320/321': 2,
            'A330': 3,
            'ARJ21': 4,
            'Boeing737': 5,
            'Boeing787': 6,
            'other': 7
        }
        
        # 起始ID
        self.start_bbox_id = 1
        self.start_image_id = 1
        
    def get_element_text(self, root: ET.Element, name: str) -> str:
        """获取XML元素文本"""
        element = root.find(name)
        if element is None:
            raise ValueError(f"找不到元素: {name}")
        return element.text
    
    def get_elements(self, root: ET.Element, name: str) -> List[ET.Element]:
        """获取XML元素列表"""
        return root.findall(name)
    
    def parse_xml_file(self, xml_path: str) -> Dict[str, Any]:
        """
        解析单个XML文件
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            包含图片信息和标注信息的字典
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图片信息
        width = int(self.get_element_text(root.find('size'), 'width'))
        height = int(self.get_element_text(root.find('size'), 'height'))
        
        # 获取文件名
        filename = os.path.basename(xml_path).replace('.xml', '.jpg')
        
        # 获取标注信息
        annotations = []
        objects = self.get_elements(root, 'object')
        
        for obj in objects:
            category_name = self.get_element_text(obj, 'name')
            
            # 跳过未定义的类别
            if category_name not in self.categories:
                print(f"警告: 跳过未定义的类别 '{category_name}' 在文件 {xml_path}")
                continue
            
            category_id = self.categories[category_name]
            
            # 获取边界框信息
            bndbox = obj.find('bndbox')
            xmin = int(float(self.get_element_text(bndbox, 'xmin')))
            ymin = int(float(self.get_element_text(bndbox, 'ymin')))
            xmax = int(float(self.get_element_text(bndbox, 'xmax')))
            ymax = int(float(self.get_element_text(bndbox, 'ymax')))
            
            # 验证边界框
            if xmax <= xmin or ymax <= ymin:
                print(f"警告: 无效的边界框在文件 {xml_path}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
                continue
            
            # 计算宽度和高度
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            annotation = {
                'category_id': category_id,
                'bbox': [xmin, ymin, bbox_width, bbox_height],
                'area': bbox_width * bbox_height,
                'iscrowd': 0,
                'segmentation': []
            }
            annotations.append(annotation)
        
        return {
            'filename': filename,
            'width': width,
            'height': height,
            'annotations': annotations
        }
    
    def convert_to_coco(self, xml_files: List[str], output_file: str) -> None:
        """
        将XML文件列表转换为COCO格式
        
        Args:
            xml_files: XML文件路径列表
            output_file: 输出JSON文件路径
        """
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # 添加类别信息
        for category_name, category_id in self.categories.items():
            coco_data['categories'].append({
                'id': category_id,
                'name': category_name,
                'supercategory': 'aircraft'
            })
        
        image_id = self.start_image_id
        annotation_id = self.start_bbox_id
        
        for xml_file in xml_files:
            try:
                data = self.parse_xml_file(xml_file)
                
                # 添加图片信息
                image_info = {
                    'id': image_id,
                    'file_name': data['filename'],
                    'width': data['width'],
                    'height': data['height']
                }
                coco_data['images'].append(image_info)
                
                # 添加标注信息
                for annotation in data['annotations']:
                    annotation['id'] = annotation_id
                    annotation['image_id'] = image_id
                    coco_data['annotations'].append(annotation)
                    annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"处理文件 {xml_file} 时出错: {e}")
                continue
        
        # 保存COCO格式文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功生成 {output_file}")
        print(f"图片数量: {len(coco_data['images'])}")
        print(f"标注数量: {len(coco_data['annotations'])}")
        print(f"类别数量: {len(coco_data['categories'])}")
    
    def split_dataset(self, train_ratio: float = 0.8) -> tuple:
        """
        按比例分割数据集
        
        Args:
            train_ratio: 训练集比例
            
        Returns:
            (训练集文件列表, 测试集文件列表)
        """
        xml_files = glob.glob(os.path.join(self.xml_dir, "*.xml"))
        xml_files = sorted(xml_files)
        
        # 设置随机种子以确保结果可重现
        np.random.seed(42)
        np.random.shuffle(xml_files)
        
        train_num = int(len(xml_files) * train_ratio)
        train_files = xml_files[:train_num]
        val_files = xml_files[train_num:]
        
        print(f"总文件数: {len(xml_files)}")
        print(f"训练集文件数: {len(train_files)}")
        print(f"验证集文件数: {len(val_files)}")
        
        return train_files, val_files
    
    def convert_dataset(self, train_ratio: float = 0.8) -> None:
        """
        转换整个数据集
        
        Args:
            train_ratio: 训练集比例
        """
        print("开始转换XML到COCO格式...")
        
        # 分割数据集
        train_files, val_files = self.split_dataset(train_ratio)
        
        # 转换训练集
        train_output = os.path.join(self.output_dir, "train.json")
        self.convert_to_coco(train_files, train_output)
        
        # 转换验证集
        val_output = os.path.join(self.output_dir, "val.json")
        self.convert_to_coco(val_files, val_output)
        
        print("转换完成！")
        
        # 生成文件列表
        self.generate_file_lists(train_files, val_files)
    
    def generate_file_lists(self, train_files: List[str], val_files: List[str]) -> None:
        """
        生成训练集和验证集文件列表
        
        Args:
            train_files: 训练集XML文件列表
            val_files: 验证集XML文件列表
        """
        # 生成训练集文件列表
        train_list_file = os.path.join(self.output_dir, "train.txt")
        with open(train_list_file, 'w') as f:
            for xml_file in train_files:
                filename = os.path.basename(xml_file).replace('.xml', '')
                f.write(f"{filename}\n")
        
        # 生成验证集文件列表
        val_list_file = os.path.join(self.output_dir, "val.txt")
        with open(val_list_file, 'w') as f:
            for xml_file in val_files:
                filename = os.path.basename(xml_file).replace('.xml', '')
                f.write(f"{filename}\n")
        
        print(f"生成文件列表: {train_list_file}, {val_list_file}")


def main():
    parser = argparse.ArgumentParser(description='XML转COCO格式转换器')
    parser.add_argument('--xml_dir', type=str, default='./annotations', 
                       help='XML标注文件目录')
    parser.add_argument('--image_dir', type=str, default='./JPEGImages',
                       help='图片文件目录')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例 (默认: 0.8)')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.xml_dir):
        print(f"错误: XML目录不存在: {args.xml_dir}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"错误: 图片目录不存在: {args.image_dir}")
        return
    
    # 创建转换器并执行转换
    converter = XMLToCOCOConverter(args.xml_dir, args.image_dir, args.output_dir)
    converter.convert_dataset(args.train_ratio)


if __name__ == "__main__":
    main()
