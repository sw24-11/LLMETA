import torch

# Check if PyTorch has CUDA support
cuda_version = torch.version.cuda

if cuda_version:
    print(f"CUDA version supported by PyTorch: {cuda_version}")
else:
    print("CUDA is not available in your current PyTorch installation.")

torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")