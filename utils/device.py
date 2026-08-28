import torch


def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    print("[INFO] CUDA not available. Using CPU.")
    return "cpu"
