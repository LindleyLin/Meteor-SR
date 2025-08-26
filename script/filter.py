import numpy as np
import torch
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 确保你的数据文件存在
dataY = np.fromfile('dataset/trainy_8_original', dtype='float32')
dataY = dataY.reshape((-1, config["channels"], 35 * 8, 43 * 8))
dataY = torch.tensor(dataY, dtype=torch.float32)

# 获取总图片数量
num_images = dataY.shape[0]

# 创建一个列表来存储每张图片的方差
variances = []

# 遍历所有图片并计算方差
for i in range(num_images):
    # 选择第i张图片的所有通道数据
    image_tensor = dataY[i]
    # 使用 .flatten() 将所有通道的像素值拉平成一维，然后计算方差
    variance = torch.var(image_tensor.flatten()).item()
    variances.append(variance)

# 将方差列表转换为 NumPy 数组，方便排序
variances_array = np.array(variances)

# 使用 np.argsort() 获取排序后的索引
# argsort() 返回的是索引，从最小到最大
sorted_indices = np.argsort(variances_array)

# 因为 sorted_indices 是从小到大排列的，所以我们需要取末尾的1000个索引
top_indices = sorted_indices[-1000:]

# 现在你可以使用这些索引来创建一个新的数据集张量
top_images = dataY[top_indices]

# 确定裁剪的中心区域
target_height = 256
target_width = 256

# 计算裁剪的起始点
start_h = (35 * 8 - target_height) // 2
start_w = (43 * 8 - target_width) // 2

# 计算裁剪的结束点
end_h = start_h + target_height
end_w = start_w + target_width

# 使用 PyTorch 的切片操作来裁剪所有图片
cropped_images = top_images[:, :, start_h:end_h, start_w:end_w]

# 将这个新数据集保存到文件中
# 使用 .numpy() 将张量转换回 NumPy 数组
# 然后使用 .tofile() 保存
cropped_images.numpy().tofile('dataset/trainy_8')