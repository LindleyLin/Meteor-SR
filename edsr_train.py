import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from nn.edsr.net import Net
from utils import GradualWarmupScheduler, get_loader

# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 实例化模型、损失函数和优化器
model = Net(num_channels=config["channels"], upscale_factor=config["upscale"], 
            base_channel=config["basechannel"], num_residuals=config["num_resi"]).to(device)

# 加载模型
# model.load_state_dict(torch.load('model/edsr/model.pth', weights_only=True, map_location=device))

optimizer = optim.AdamW(params=model.parameters(), lr=config["lr"], weight_decay=1e-4)
cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"], eta_min=0, last_epoch=-1)
warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=config["multiplier"], warm_epoch=config["epochs"] // 10, after_scheduler=cosineScheduler)
criterion = nn.L1Loss()

model.train()

train_loss = 0
train_loader, _ = get_loader(norm_option=1)

for epoch in range(config["epochs"]):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["grad_clip"])
        optimizer.step()
    warmUpScheduler.step()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch : {epoch} Loss : {avg_train_loss} LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
    train_loss = 0
    if epoch % 5 == 0:
        torch.save(model.state_dict(), "model/edsr/model.pth")