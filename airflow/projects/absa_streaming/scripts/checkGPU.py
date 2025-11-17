import torch

print("🧠 PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("🚀 GPU detected:", torch.cuda.get_device_name(0))
    print("🧮 Total GPUs:", torch.cuda.device_count())
    print("🔥 Current device:", torch.cuda.current_device())
else:
    print("❌ CUDA is NOT available — running on CPU.")
