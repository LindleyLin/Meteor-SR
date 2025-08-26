import torch
import torch.nn.functional as F
import numpy as np
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 加载原始数据
dataY = np.fromfile('dataset/trainy_8', dtype='float32')
dataY = dataY.reshape((-1, config["channels"], config["piy"] * 8, config["pix"] * 8))
dataY = torch.tensor(dataY, dtype=torch.float32)

# 定义下采样后的尺寸
heightx1 = config["piy"]
widthx1 = config["pix"]
heightx2 = config["piy"] * 2
widthx2 = config["pix"] * 2
heightx4 = config["piy"] * 4
widthx4 = config["pix"] * 4

# 将边长缩小为原来的八分之一
# mode='bicubic' 是一种常用的高质量下采样方法
dataYx1 = F.interpolate(dataY, size=(heightx1, widthx1), mode='bicubic', align_corners=False)
dataYx1 = torch.clamp(dataYx1, min=0)

# 将边长缩小为原来的二分之一
# mode='bicubic' 是一种常用的高质量下采样方法
dataYx2 = F.interpolate(dataY, size=(heightx2, widthx2), mode='bicubic', align_corners=False)
dataYx2 = torch.clamp(dataYx2, min=0)

# 将边长缩小为原来的四分之一
dataYx4 = F.interpolate(dataY, size=(heightx4, widthx4), mode='bicubic', align_corners=False)
dataYx4 = torch.clamp(dataYx4, min=0)

# 保存下采样后的数据到文件
# 使用 .numpy() 将张量转换回 NumPy 数组
# 然后使用 .tofile() 保存

dataYx1.numpy().tofile('dataset/trainx')
dataYx2.numpy().tofile('dataset/trainy_2')
dataYx4.numpy().tofile('dataset/trainy_4')