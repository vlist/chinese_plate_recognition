
# Chinese Plate Recognition
使用OpenCV和CNN搭建的车牌识别程序原型
## Requirements
```
matplotlib==3.5.0
opencv_python==4.5.5.62
numpy==1.19.5
tensorflow==2.7.0
ipykernel==6.4.1
```
## 文件说明
| 文件名        | 描述               |
| ------------- | ------------------ |
| process.ipynb | 项目主要流程       |
| train.py      | 模型训练脚本       |
| cnn.py        | 模型结构和预测功能 |
| temp.py       | 数据整理小工具     |
| test.ipynb    | 模型测试脚本       |

## 数据集

[Google Drive链接](https://drive.google.com/file/d/1dkDPT4ZcEOLze8dCSdpKcGT50I_dK_Es/view?usp=sharing)

```
$ tree -d plate_dataset 
plate_dataset
├── test
├── train
└── val
```

