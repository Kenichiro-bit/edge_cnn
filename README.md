# edge_cnn
some edge AI tool make us happy

エッジでAIを作成したいため、こちらを作成
段階としては
1. 画像認識のための基礎知識習得。
2. CNNとは？その細かい計算内容から量子化までっまとめる
3. CNNをラズパイで実装し速度を計算
4. CNNの計算領域をFPGAで高速化し実験
5. 新しいアーキテクチャの検討

こんな感じかなと思っている。

まずは、画像認識についてこれからまとめていく。


# まずは画像認識のための基礎知識

1. 前処理が必須
   通常の画像のままではDeepLearningModelに合わず読み込むことができない。画像を処理できるように入力された画像を事前に処理する工程
* 入力画像を224x224の形に変形
* テンソル形式[[[]]]に変更。(tensorは行列にさらに行列が埋め込まれているようなもの。グレイスケールでは1階、RGBでは3階テンソルになる。)詳細は→URL
テンソル形式にしないと計算させづらく、pytorchなどではテンソル形式で計算させるのが一般的
* 最後に平均と標準偏差を規定し、その値の中で画像データを正規化する。

以上の三工程(大きさ→Tensor→正規化)

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

2. データのダウンロード(データの用意)
今回はAmazonのデータセットを使用。image-netの軽量版をダウンロード

```python
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
download_and_extract_archive(url, download_root="data")
```

3. データセットを分割
データセットをトレーニング用とテスト用に分割。この時の変形方式をtransformで設定。
train_datasetとvalid_datasetに分割(今回はそもそもデータセットが分かれているので分割ではなく読み込みのみ。)
また、DataLoaderでシャッフル、バッチサイズを定義。バッチサイズは、データセットをバッチサイズ分に分割し、それぞれのバッチで学習を行うもの。
全体で学習するよりも重みの収束率が良い(ミニバッチ降下法)

<img width="619" height="636" alt="image" src="https://github.com/user-attachments/assets/bab9214b-2848-45b8-ab32-641b85cbd085" />
<img width="320" height="534" alt="image" src="https://github.com/user-attachments/assets/9303b74a-fea9-4c71-a758-b5ed0c82d8ba" />

[https://chefyushima.com/ai-ml_batch-online/2781/](url)

```python
train_ds = ImageFolder("data/imagenette2-160/train", transform=transform)
val_ds   = ImageFolder("data/imagenette2-160/val",   transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
```

4. CNNモデルの定義

CNNのモデルの定義を行う。今回はtorchを使用しているので、torchのnnモジュールの中から使用する処理を選択し、それを以下のように定義。
今回は、畳み込み→pooling→畳み込み→プーリング→全結合→全結合
そして、それらをforward()で順伝播させていく。
今回はself.pool(F.relu(self.conv1(x)))で畳み込み→プーリングを行い
x = x.view(x.size(0), -1)で以下の処理、F.relu(self.fc1(x))で全結合を行う。

🔍 分解して解説します
x = x.view(x.size(0), -1)  
```x.size(0)```

x の 0番目の次元のサイズ、つまり「バッチサイズ」を意味します。
例：もし x が [100, 3, 28, 28]（画像100枚、RGB3チャンネル、28×28ピクセル）なら
→ x.size(0) は 100 です。

```.view(...)```

NumPy の reshape() と同じ役割で、テンソルの形を変えます。

ただしメモリ上の配置を変えない（同じデータを違う形で見る）点が特徴です。

```-1```

PyTorchでは「残りの次元を自動的に計算してくれ」という意味です。
つまり「全要素数が一致するように、ここは自動で決めて」という指定です。
**
 
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 56 * 56, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # →112×112
        x = self.pool(F.relu(self.conv2(x)))  # →56×56
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

5．トレーニング
torchでデバイスを選択し、CPUかcudaを選択。
modelにCNNの先ほど定義したものを、classは分類先がいくつか(10個の画像に分類するなら10)を記載
optimazerは最適化関数。今回はAdamで学習率は1e-3。損失関数はクロスエントロピー誤差関数。

今回はepochを5にして、5回学習をさせる。まず、イメージをモデルに食わせ、その結果をlabelと比較。
その後、最適化関数を初期化させ、その損失を計算。その計算結果をもとに逆伝播させ重みの更新を行う。

```python
# 5) トレーニングループ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}  Loss: {total_loss/len(train_loader):.4f}")
```

6. 最後に推論をさせる
modelを推論モードにさせ、入力をサンプルイメージにし検証。
予測結果が自分のものと一致しているかを確認する。

```python
model.eval()
sample_imgs, _ = next(iter(val_loader))
with torch.no_grad():
    preds = model(sample_imgs.to(device))
    print("予測クラス：", torch.argmax(preds, dim=1)[:5].cpu().tolist())
```
