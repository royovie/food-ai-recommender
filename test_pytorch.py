import torch

x = torch.rand(3, 3)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Tensor:")
print(x)