import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def generate_mat(M, N, file_path):
    A = torch.randn(M, N)
    torch.save(A, file_path)
    print(f"save mat to {file_path}")


def generate_sample(A, x, file_path):
    b = A @ x
    torch.save(b, file_path)
    print(f"save sample to {file_path}")

def image_load(name, ratio):
    image = Image.open(name).convert('L')
    x = np.array(image)
    dim = x.shape
    N = dim[0] * dim[1]
    M = int(ratio * N)
    x = torch.from_numpy(x)
    x = x.float() / 255
    x = x.reshape(N, 1)
    return x, M, N, dim

class CustomDataset(Dataset):
    def __init__(self, A_path, b_path):
        self.A_path = A_path
        self.b_path = b_path
        self.A = torch.load(A_path)
        self.b = torch.load(b_path)

    def __len__(self):
        return len(self.b)

    def __getitem__(self, idx):
        A_sample = self.A[idx]
        b_sample = self.b[idx]
        return A_sample, b_sample

