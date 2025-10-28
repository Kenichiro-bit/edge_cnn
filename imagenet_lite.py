import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

# ベンチマーク関数（推論時間＆Top-1精度を測定）
def benchmark(model_fn, dataloader, device="cpu"):
    model_fn = model_fn.to(device)
    model_fn.eval()
    total, correct = 0, 0
    times = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            start = time.perf_counter()
            outputs = model_fn(imgs)
            times.append(time.perf_counter() - start)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_latency = sum(times) / len(times) * 1000  # ms
    accuracy   = correct / total * 100
    print(f"Avg latency: {avg_latency:.1f} ms  |  Accuracy: {accuracy:.2f}%")

# 前処理パイプライン
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── A-1: Imagenette (10 クラス) ─────────────────────────
imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
download_and_extract_archive(imagenette_url, download_root="data")
train_ds_10 = ImageFolder("data/imagenette2-160/train", transform=transform)
val_ds_10   = ImageFolder("data/imagenette2-160/val",   transform=transform)
val_loader_10 = DataLoader(val_ds_10, batch_size=32, shuffle=False, num_workers=0)

# モデル例：TFLite ではなく PyTorch の MobileNetV3
# （TFLite を使う場合は model_fn で interpreter.invoke() ラップ）
from torchvision.models import mobilenet_v3_small
model_10 = mobilenet_v3_small(pretrained=True)

print("=== Imagenette (10 クラス) ===")
benchmark(model_10, val_loader_10, device="cpu")


# ── A-2: Tiny-ImageNet (200 クラス) ────────────────────
tiny_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
download_and_extract_archive(tiny_url, download_root="data", filename="tiny-imagenet-200.zip")
# 展開後フォルダ: data/tiny-imagenet-200/train, val
train_ds_200 = ImageFolder("data/tiny-imagenet-200/train", transform=transform)
val_dir = "data/tiny-imagenet-200/val/images"  # Tiny-ImageNet の検証は少し構造が特殊
#  val/val_annotations.txt を読んで ImageFolder 用に symlink かコピーで準備しておく必要あり
#  ここでは簡単化のため train データでベンチ
val_ds_200   = train_ds_200  # 実際は検証セット用意をお願いします
val_loader_200 = DataLoader(val_ds_200, batch_size=32, shuffle=False, num_workers=0)

model_200 = mobilenet_v3_small(pretrained=True)
print("\n=== Tiny-ImageNet (200 クラス) ===")
benchmark(model_200, val_loader_200, device="cpu")
