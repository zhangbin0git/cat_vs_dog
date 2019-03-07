

# 机器学习纳米学位
猫狗大战毕业项目   
张斌   
2019年3月6日   

## 项目说明

本项目是优达学城的一个毕业项目。项目要求使用深度学习方法识别一张图片是猫还是狗

- 输入：一张彩色图片
- 输出：是猫还是狗

## 实验环境
项目使用Anaconda搭建环境。使用environmert目录下的yml进行环境安装。   

$ conda env create -f environmert/environmert.yml



## 数据来源

数据集来自 kaggle 上的一个竞赛：[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)。

下载kaggle猫狗数据集解压后分为 3 个文件 train.zip、 test.zip 和 sample_submission.csv。

train 训练集包含了 25000 张猫狗的图片， 每张图片包含图片本身和图片名。命名规则根据“type.num.jpg”方式命名。

test 测试集包含了 12500 张猫狗的图片， 每张图片命名规则根据“num.jpg”，需要注意的是测试集编号从 1 开始， 而训练集的编号从 0 开始。

sample_submission.csv 需要将最终测试集的测试结果写入.csv 文件中，上传至 kaggle 进行打分。

### 基准模型   
本项目的最低要求是kaggle Public Leaderboard前10%。   
在kaggle上，总共有1314只队伍参加了比赛，前10%为131位之前，131位的得分是0.06127，所以模型预测结果分数要小于0.06127。

### 评价指标
kaggle的评估标准是log损失函数，以下为表达式：   
$LogLoss = -\frac{1}{n}\sum_{i=1}^n [y_ilog(\hat{y}_i)+(1-y_i)log(1- \hat{y}_i)]$

其中：   
n 是测试集中图片数量。   
$\hat{y}_i$是图片预测为狗的概率。   
${y}_i$如果图像是狗，则为1，如果是猫，则为0。   
$log()$是自然对数。   
对数损失越小，代表模型的性能越好。    

交叉熵是分类问题中常用的损失函数，被广泛应用。
我使用的评估指标是kaggle提出了，首先保证的评估指标的适用性和合理性。以上计算公式摘自kaggle“Dogs vs. Cats Redux: Kernels Edition”项目。

### 问题陈述
这次需要解决的问题是：通过利用计算机图像识别技术，对kaggle提供的训练集数据利用CNN的方式进行分析学习训练出优秀的图像识别模型，利用图像识别模型对测试集中的图片进行预测，预测图像是狗或猫的概率，并将结果存入sample_submission.csv文件中，上传至kaggle进行打分。   
1、首先将对数据集中的图片进行筛选，利用箱型图原理，剔除异常的图片，主要包含低分辨率和错误的图片。     
2、建立训练集和验证集。   
3、由于神经网络输入点图像的维度和像素是固定的，通过对图像进行预处理，统一图像的维度。     
4、利用迁移学习，使用开源的优秀的图像识别模型并加载权重作为固定的图像特征提取器，利用多个具有优秀权重的识别模型对数据进行预处理，观察各自的预测结果的准确率和多个模型联合起来的预测结果准确率。   
5、结合迁移学习的模型，建立CNN的数据模型。      
6、对CNN模型进行训练，得到最优的模型。   
7、利用模型预测测试集中的图像是狗的概率，将结果按照要求存入sample_submission.csv文件中。   
期望的结果是训练出的模型可以准确的识别出图像是狗还是猫，并在kaggle上得到优秀评分。 

## 项目部署
项目目录结构：
```
.cat_vs_dog
├── data
│   ├── abnormal_pic
│   ├── pre_data
│   ├── test
│   └── train
├── cat_vs_dog-final.ipynb
├── environmert
├── proposal.ipynb
├── README.md
└── report.ipynb
```
## 使用的库
import os            
import numpy as np            
import shutil            
from PIL import Image            
from collections import Counter            
from glob import glob                
from sklearn.utils import shuffle            
from keras.preprocessing import image            
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_ResNet50            
from keras.applications.xception import Xception, preprocess_input as preprocess_input_Xception            
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_InceptionV3            
from keras.preprocessing.image import ImageDataGenerator            
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as preprocess_input_InceptionResNetV2            
from keras.models import Sequential, Model            
from keras.callbacks import ModelCheckpoint             
from keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, Dense, Dropout, Flatten            
from keras.utils import np_utils            
import pandas as pd            
import re            
import matplotlib.pyplot as plt            

## 机器硬件
由于此项目要求的计算量较大，使用了亚马逊p2.xlarge云服务器来完成该项目。

## 机器操作系统
服务器的操作系统为Linux version 4.4.0-1075-aws (buildd@lgw01-amd64-035) 

## 训练时间
在亚马逊p2.xlarge云服务器上总共训练时间为1个半小时内。


