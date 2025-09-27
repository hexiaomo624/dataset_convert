# XML转COCO格式转换器使用说明

## 概述
这个转换器可以将SAR飞机检测数据集的XML格式标签转换为COCO格式标签，并按照8:2的比例自动分割训练集和验证集。

## 生成的文件
- `train.json`: 训练集COCO格式标签文件
- `val.json`: 验证集COCO格式标签文件  
- `train.txt`: 训练集文件列表
- `val.txt`: 验证集文件列表

## 数据集统计
- **总文件数**: 4368个XML文件
- **训练集**: 3494个文件，13265个标注
- **验证集**: 874个文件，3198个标注
- **类别数**: 7个飞机类别

## 支持的飞机类别
1. A220 (ID: 1)
2. A320/321 (ID: 2)  
3. A330 (ID: 3)
4. ARJ21 (ID: 4)
5. Boeing737 (ID: 5)
6. Boeing787 (ID: 6)
7. other (ID: 7)

## COCO格式说明
生成的JSON文件包含以下结构：
- `images`: 图片信息（文件名、尺寸、ID）
- `annotations`: 标注信息（边界框、类别、面积等）
- `categories`: 类别定义

## 使用方法
```bash
python xml_to_coco_converter.py --xml_dir ./annotations --image_dir ./JPEGImages --output_dir . --train_ratio 0.8
```

## 参数说明
- `--xml_dir`: XML标注文件目录（默认: ./annotations）
- `--image_dir`: 图片文件目录（默认: ./JPEGImages）
- `--output_dir`: 输出目录（默认: 当前目录）
- `--train_ratio`: 训练集比例（默认: 0.8）

## 注意事项
- 转换器会自动跳过未定义的类别
- 会验证边界框的有效性
- 使用固定随机种子确保结果可重现
- 生成的COCO格式完全符合标准规范
