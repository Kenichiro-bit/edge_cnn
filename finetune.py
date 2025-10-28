import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models import mobilenet_v2

# ── 1. 前処理 パイプライン ────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── 2. データセット ダウンロード＆読み込み ───────────────
train_ds = CIFAR100(root="data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR100(root="data", train=False, download=True, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)

# ── 3. モデル準備（ImageNet pretrained → 100クラスヘッド） ──
model = mobilenet_v2(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ── 4. ヘッドのみファインチューニング ─────────────────────
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        print(f"[Head] Epoch {epoch}  Loss: {loss.item():.4f}  Acc: {correct/total:.3f}", end="\r")
    print(f"[Head] Epoch {epoch}  Acc: {correct/total:.3f}")

# ── 5. 全層微調整 ───────────────────────────────────
for p in model.parameters():
    p.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(5):
    model.train()
    total, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        print(f"[Fine-tune] Epoch {epoch}  Loss: {loss.item():.4f}  Acc: {correct/total:.3f}", end="\r")
    print(f"[Fine-tune] Epoch {epoch}  Acc: {correct/total:.3f}")

# ── 6. 検証 ───────────────────────────────────────────
model.eval()
total, correct = 0, 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Validation Acc: {correct/total:.3f}")
