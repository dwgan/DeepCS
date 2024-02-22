import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from model import DeepCS
import torch.nn.functional as F
from data_loader import generate_mat, generate_sample, CustomDataset, image_load
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
# matplotlib.use('Agg')  # 使用Agg后端，它不需要图形界面
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def tv_loss(x, dim):
    if x.dim() == 1:
        tv = torch.sum(torch.abs(x[:-1] - x[1:]))
    elif x.dim() == 2:
        x = x.reshape(dim)
        tv = torch.sum(torch.abs(x[:-1, :] - x[1:, :])) + torch.sum(torch.abs(x[:, :-1] - x[:, 1:]))
    else:
        raise ValueError("Unsupported input dimensions")
    return tv

torch.manual_seed(42)

epochs = 1000
batch_size = 16384
learning_rate = 2
lambda_reg = 0.0008
ratio = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

name = "{}/({}).png".format('./dataset/img', 1)
x, M, N, dim = image_load(name, ratio)

path_mat = './dataset/mat/A.pt'
generate_mat(M, N, path_mat)
A = torch.load(path_mat)

path_sample = './dataset/sample/b.pt'
generate_sample(A, x, path_sample)
b = torch.load(path_sample)

dataset = CustomDataset(path_mat, path_sample)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DeepCS(N).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[400], gamma=0.1)

total_samples = M
plt.ion()
fig, ax = plt.subplots()
for epoch in range(1, epochs + 1):
    model.train()

    for A_sample, b_sample in dataloader:
        A_sample, b_sample = A_sample.to(device), b_sample.to(device)

        A_sample, b_sample = A_sample.to(device), b_sample.to(device)
        optimizer.zero_grad()

        output = model(A_sample)

        mse_loss = F.mse_loss(output, b_sample)
        tv = tv_loss(model.x, dim)
        loss = mse_loss + lambda_reg * tv

        loss.backward()
        optimizer.step()
    scheduler.step()
    # with torch.no_grad():
    #     model.x.data.clamp_(0, 1)
    if epoch % 10 == 0:
        image = model.x.detach().cpu().numpy()
        ax.clear()
        ax.imshow(image.reshape(dim), cmap='gray')
        ax.axis('off')
        plt.draw()
        plt.pause(0.1)
        ssim_value = ssim(x.cpu().numpy().reshape(dim), image.reshape(dim), data_range=image.max() - image.min())
        psnr_value = psnr(x.cpu().numpy().reshape(dim), image.reshape(dim), data_range=image.max() - image.min())
        print(f"Epoch {epoch}: SSIM = {ssim_value:.4f}, PSNR = {psnr_value:.4f}, loss = {loss:.4f}, mse_loss = {mse_loss:.4f}, tv = {tv:.4f}")

plt.show()