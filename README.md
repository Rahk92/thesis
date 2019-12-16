# DPPN(Dual Pyramid Pooling Network) with Single Image Haze Removal
by Rahoon, Kang and Jechang Jeong

## Introduction
this repository is for the my master's degree thesis. (Feb. 2020, Hanyang University)

## Dependencies
1. Pytorch >= 1.0.0 (Mine is 1.2.0)
2. Pillow
3. (optional) PyCharm IDE

## Usuage
For training, run 'train.py' with tranining configuration, which is located in line 29 to 39 in 'train.py'
```shell
# Train configuration
batch_size = 2
aug = 10     #min == 1
num_epochs = 1000
learning_rate = 0.0001
early_stop = 0
early_stop_setting = 1000
restart = 0
fine = False
padd = 0
pad_setting = [0, 0]
val_save = False
```

For testing, run 'test.py' with weight in '/wgts' folder and weight path which is located in line 23 to 26 in 'test.py'
```shell
test_path = 'E:/dataset/dehazing/mydataset/Valid/2018/'
result_path = './result/MDN3/2018'
load_weight_path = "./wgts/MDN3/train/859.ckpt"
```

## Dataset

### Training set
Combination of I-HAZE, O-HAZE, Dense-Haze dataset
1. I-HAZE : 01_indoor_.jpg ~ 25_indoor_.jpg (25)
2. O-HAZE : 01_outdoor_.jpg ~ 40_indoor_.jpg (40)
3. Dense-Haze : 01.png ~ 50.png (50)

Total 115 images

### Validation set(Testing set)
each 5 images from I-HAZE, O-HAZE, Dense-Haze dataset
1. I-HAZE : 31_indoor_.jpg ~ 35_indoor_.jpg (5)
2. O-HAZE : 41_outdoor_.jpg ~ 45_outdoor_.jpg (5)
3. Dense-Haze : 51.png ~ 55.png (5)

Total 15 images

#### Pre-trained weight
Download link 
https://drive.google.com/open?id=1KqNAI1UKZdq-VtsGMq4fa9NL6Y4KGCq0
