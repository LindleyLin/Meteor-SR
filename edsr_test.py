import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from nn.edsr.net import Net
from utils import get_loader, get_data, normalize0

# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 实例化模型
model = Net(num_channels=config["channels"], upscale_factor=config["upscale"], 
            base_channel=config["basechannel"], num_residuals=config["num_resi"]).to(device)

# 加载模型
model.load_state_dict(torch.load('model/edsr/model.pth', weights_only=True, map_location=device))

criterion = torch.nn.L1Loss()

model.eval()

_, test_loader = get_loader(norm_option=0)
test_loss = 0

with torch.no_grad():
    for _, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)
        mae = criterion(prediction, target)
        test_loss += mae.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Loss : {avg_test_loss}")

dataX, dataY = get_data(config)
dataX, dataY = normalize0(dataX, dataY)

levels = np.arange(-0.05, 1.05, 0.05)

inputs = dataX[998:999, :, :, :].to(device)
yori = dataY[998:999, :, :, :]

# 模型预测
with torch.no_grad():
    outputs = model(inputs)

inputs = inputs.cpu().numpy()
outputs = outputs.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
step = 0 # 选择第0个样本进行绘图

# 绘制第一张图：输入
im1 = axes[0].contourf(inputs[step, 0, :, :], levels=levels)
axes[0].set_title('Normalized Input')
fig.colorbar(im1, ax=axes[0])

# 绘制第二张图：原始
im2 = axes[1].contourf(yori[step, 0, :, :], levels=levels)
axes[1].set_title('Normalized Ground Truth')
fig.colorbar(im2, ax=axes[1])

# 绘制第三张图：模型输出
im3 = axes[2].contourf(outputs[step, 0, :, :], levels=levels)
axes[2].set_title('Normalized Output')
fig.colorbar(im3, ax=axes[2])

plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.show()

print(outputs[step, 0, :, :].min())