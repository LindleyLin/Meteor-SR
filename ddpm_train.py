import torch
import torch.optim as optim
import yaml

from nn.ddpm.diffusion import GaussianDiffusionTrainer
from nn.ddpm.net import UNet
from utils import GradualWarmupScheduler, get_loader

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_loader, _ = get_loader(norm_option=1)

# model setup
net_model = UNet(ori_ch=config["channels"], ch=config["basechannel"], ch_mult=config["channel_mult"],
    attn=config["attn"], num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
net_model.train()
    
# net_model.load_state_dict(torch.load("model/ddpm/model.pth", weights_only=True, map_location=device))

optimizer = torch.optim.AdamW(net_model.parameters(), lr=config["lr"], weight_decay=1e-4)
cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"], eta_min=0, last_epoch=-1)
warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=config["multiplier"], warm_epoch=config["epochs"] // 10, after_scheduler=cosineScheduler)
trainer = GaussianDiffusionTrainer(net_model, config["T"]).to(device)
train_loss = 0

# start training
for e in range(config["epochs"]):
    for batch_idx, (data, target) in enumerate(train_loader):
        # train
        optimizer.zero_grad()
        x_r, x_0 = data.to(device), target.to(device)
        loss = trainer(x_0, x_r)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net_model.parameters(), config["grad_clip"])
        optimizer.step()
    warmUpScheduler.step()
    torch.save(net_model.state_dict(), "model/ddpm/model.pth")
    print(f"Epoch: {e}, Loss: {train_loss / len(train_loader):.4f}, LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
    train_loss = 0