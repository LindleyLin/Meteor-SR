import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from nn.ddpm.diffusion import GaussianDiffusionSampler
from nn.ddpm.net import UNet

from utils import get_data, normalize_1

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    model = UNet(ori_ch=config["channels"], ch=config["basechannel"], ch_mult=config["channel_mult"], 
        attn=config["attn"], num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
    model.load_state_dict(torch.load("model/ddpm/model.pth", weights_only=True, map_location=device))
    print("model load weight done.")
    model.eval()
    sampler = GaussianDiffusionSampler(model, config["T"], config["iT"], config["eta"]).to(device)

    dataX, dataY = get_data(config)
    dataX, dataY = normalize_1(dataX, dataY)

    levels = np.arange(-1.05, 5.05, 0.05)

    inputs = dataX[970:971, :, :, :].to(device)
    yori = dataY[970:971, :, :, :]

    # Sampled from standard normal distribution
    noisyImage = torch.randn(size=[inputs.size(0), config["channels"], config["piy"] * config["upscale"], config["pix"] * config["upscale"]], device=device)

    # 模型预测
    sampledImgs = sampler(noisyImage, inputs)

    inputs = inputs.cpu().numpy()
    outputs = sampledImgs.cpu().numpy()

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