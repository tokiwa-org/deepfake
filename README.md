# Deepfake 教育プロジェクト

**目的**: Deepfake技術の仕組みを理解するための教育・研究プロジェクトです。

> ⚠️ **注意**: このプロジェクトは教育目的のみです。悪用は厳禁です。

---

## アーキテクチャ概要

このプロジェクトは、Deepfake技術の基礎となる**オートエンコーダベースの顔交換モデル**を実装しています。

### Deepfakeの仕組み

```
人物Aの顔 → 共有エンコーダ → 潜在空間 → デコーダB → 人物Bの顔（Aの表情）
人物Bの顔 → 共有エンコーダ → 潜在空間 → デコーダA → 人物Aの顔（Bの表情）
```

**主要コンポーネント**:
1. **共有エンコーダ**: すべての顔に共通する特徴（表情、ポーズ、照明）を学習
2. **人物別デコーダ**: 特定の人物の顔を再構築する方法を学習
3. **顔交換**: エンコーダ + 別人物のデコーダを組み合わせて実現

---

## モデル設定の比較

### 軽量版（学習用） vs 本番版（品質重視）

| 設定項目 | 軽量版 | 本番版 | 説明 |
|---------|-------|-------|------|
| **パラメータ数** | ~670万 | ~5,000万〜2億 | 本番は10〜30倍必要 |
| **画像サイズ** | 128×128 | 256×256〜512×512 | 高解像度で細部を表現 |
| **潜在次元** | 128 | 512〜1024 | 顔の特徴をより詳細に捉える |
| **エンコーダ深度** | 5層 | 8〜12層 | より複雑な特徴を抽出 |
| **損失関数** | L1のみ | L1 + Perceptual + GAN | 高品質な生成に必要 |
| **学習時間/epoch** | ~15秒 | ~10〜30分 | GPUスペックにも依存 |
| **推奨GPU** | M4 Mac / GTX 1060+ | RTX 3090 / A100 | VRAM 8GB以上推奨 |

### なぜ軽量版を使うのか？

本プロジェクトでは**軽量版をデフォルト**としています。理由は主に**ハードウェアスペックの制約**です：

1. **GPU/メモリの制約**: 本番品質のモデル（5,000万〜2億パラメータ）には RTX 3090 / A100 クラスのGPUと大量のVRAMが必要。M4 Macの16GB統合メモリでは軽量版が最適
2. **学習時間**: 本番設定では1エポック10〜30分 → 軽量版は約15秒で、実験を繰り返しやすい
3. **教育目的に十分**: アルゴリズムの理解と動作確認には670万パラメータで十分
4. **デバッグ容易**: 問題発見が速く、コードの理解に集中できる

> 💡 **品質を求める場合**: 高品質な顔交換が必要な場合は、DeepFaceLab や SimSwap などの商用レベルのフレームワークを推奨します。

### 本番品質が必要な場合

商用レベルの品質には以下が必要です：

```python
# 本番設定の例
model = DeepfakeModel(
    input_size=256,      # 高解像度
    latent_dim=512,      # 大きな潜在空間
)

trainer = DeepfakeTrainer(
    model=model,
    use_perceptual=True,  # VGG perceptual loss有効
    # + GAN discriminatorの追加を推奨
)
```

**参考: 有名なDeepfakeモデルのパラメータ数**

| モデル | パラメータ数 | 特徴 |
|-------|------------|------|
| DeepFaceLab | ~1億 | 業界標準、高品質 |
| SimSwap | ~8,000万 | リアルタイム対応 |
| FaceSwap | ~5,000万 | オープンソース |
| **本プロジェクト（軽量版）** | **670万** | 教育・学習用 |

---

## プロジェクト構造

```
deepfake/
├── src/
│   ├── models/           # ニューラルネットワーク
│   │   ├── encoder.py    # 共有エンコーダ
│   │   ├── decoder.py    # 人物別デコーダ
│   │   └── autoencoder.py # 完全なモデル
│   ├── data/             # データ処理
│   │   ├── dataset.py    # PyTorch Dataset
│   │   └── transforms.py # 画像拡張
│   ├── training/         # 学習ロジック
│   │   ├── trainer.py    # トレーナー
│   │   └── losses.py     # 損失関数
│   └── utils/            # ユーティリティ
│       └── face_utils.py # 顔検出・位置合わせ
├── data/                 # 学習データ（gitignore）
│   ├── person_a/         # 人物Aの顔画像
│   └── person_b/         # 人物Bの顔画像
├── checkpoints/          # モデル重み（gitignore）
├── runs/                 # TensorBoardログ
├── train.py              # 学習スクリプト
├── requirements.txt
└── README.md
```

---

## セットアップ（M4 Mac）

```bash
# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

---

## 使い方

### 1. データの準備

`data/person_a/` と `data/person_b/` に顔画像を配置します。

**画像の要件**:
- 各人物500枚以上推奨
- 顔がはっきり写った画像
- 様々な表情・角度があると良い

**データセットの取得方法**:

1. **Kaggle FFHQ Dataset（推奨）**
   - [Flickr-Faces-HQ (FFHQ)](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) から高品質な顔画像をダウンロード
   - ダウンロード後、2人分に分けて `data/person_a/` と `data/person_b/` に配置

2. **動画から抽出**
   ```bash
   # ffmpegで動画からフレームを抽出（1秒あたり5フレーム）
   ffmpeg -i video.mp4 -vf fps=5 data/person_a/frame_%04d.jpg
   ```

3. **自分で撮影**
   - 同意を得た人物の顔を様々な角度・表情で撮影
   - スマートフォンのカメラで十分

### 2. 学習の実行

```bash
# 軽量版（高速、教育用）
python train.py --data-a data/person_a --data-b data/person_b

# 品質重視版（低速、高品質）
python train.py --data-a data/person_a --data-b data/person_b \
    --perceptual --image-size 256 --batch-size 8
```

### 3. 学習の監視

TensorBoardで学習の進捗を可視化できます：

```bash
# TensorBoardのインストール（初回のみ）
pip install tensorboard

# TensorBoardの起動
tensorboard --logdir runs
# ブラウザで http://localhost:6006 を開く
```

### 4. 学習済みモデルの使用

```python
import torch
from src.models import DeepfakeModel

# モデルの読み込み
model = DeepfakeModel()
checkpoint = torch.load('checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 顔交換の実行
# face_a: 人物Aの顔画像テンソル [1, 3, 128, 128]
swapped = model.swap_face(face_a, target_person='b')
# → 人物Bの顔（人物Aの表情）
```

---

## コマンドラインオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--data-a` | (必須) | 人物Aの画像ディレクトリ |
| `--data-b` | (必須) | 人物Bの画像ディレクトリ |
| `--epochs` | 100 | エポック数 |
| `--batch-size` | 16 | バッチサイズ |
| `--image-size` | 128 | 画像サイズ（128 or 256） |
| `--lr` | 0.0001 | 学習率 |
| `--perceptual` | False | Perceptual Lossを有効化 |
| `--resume` | None | チェックポイントから再開 |

---

## Jupyter Notebook の使い方

`notebooks/01_understanding_deepfake.ipynb` でDeepfake技術を対話的に学べます。

### VS Code での実行

1. **Python環境の設定**
   ```bash
   # venvを Jupyter カーネルとして登録
   source venv/bin/activate
   pip install ipykernel
   python -m ipykernel install --user --name=deepfake --display-name "Python (deepfake)"
   ```

2. **ノートブックを開く**
   - VS Code で `notebooks/01_understanding_deepfake.ipynb` を開く

3. **カーネルの選択**
   - 右上のカーネル選択をクリック
   - 「Select Another Kernel...」→「Jupyter Kernel...」
   - 「**Python (deepfake)**」を選択

4. **実行**
   - 「Run All」で全セルを順番に実行
   - または Shift+Enter で1セルずつ実行

> ⚠️ **注意**: ノートブックは必ず**最初のセルから順番に実行**してください。途中のセルだけ実行すると、変数やインポートが定義されていないエラーになります。

### JupyterLab での実行

```bash
source venv/bin/activate
pip install jupyterlab
jupyter lab
# ブラウザで自動的に開きます
```

---

## 技術詳細

### オートエンコーダアーキテクチャ

**エンコーダ** (入力 → 潜在空間):
- 畳み込み層 + LeakyReLU
- ダウンサンプリング: 128×128 → 4×4×128

**デコーダ** (潜在空間 → 出力):
- PixelShuffle によるアップサンプリング
- 4×4×128 → 128×128×3

### 損失関数

| 損失 | 役割 | 軽量版 | 本番版 |
|-----|-----|-------|-------|
| **L1 Loss** | ピクセル単位の再構築 | ✓ | ✓ |
| **Perceptual Loss** | 構造的類似性（VGG特徴） | - | ✓ |
| **GAN Loss** | リアルな生成 | - | 推奨 |

### M4 Mac 最適化

- **MPS (Metal Performance Shaders)** バックエンドでGPU加速
- 統合メモリアーキテクチャに最適化されたバッチサイズ
- `num_workers=0` でマルチプロセス問題を回避

---

## 参考文献

- [FaceSwap GAN](https://arxiv.org/abs/1712.03451)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [DeepFaceLab](https://github.com/iperov/DeepFaceLab)

---

## ライセンス

教育・研究目的のみ。商用利用・悪用禁止。
