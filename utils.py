import torch


def get_device(use_accelerators: bool = True) -> str:
    if not use_accelerators:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"

    return "cpu"


def main():
    device = get_device()
    print(device)


if __name__ == "__main__":
    main()
