#!/usr/bin/env python3
"""
顔交換デモスクリプト

学習済みモデルを使用して顔交換を実行し、結果を可視化します。

使い方:
    python demo_swap.py

出力:
    - outputs/swap_demo.png: 顔交換結果の比較画像
"""

import os
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models import DeepfakeModel


def get_device() -> torch.device:
    """M4 Mac用のデバイスを取得"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> DeepfakeModel:
    """学習済みモデルを読み込む"""
    model = DeepfakeModel(input_size=128, latent_dim=128)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_image(image_path: str, size: int = 128) -> torch.Tensor:
    """画像を読み込んでテンソルに変換"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # バッチ次元を追加


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """テンソルをPIL画像に変換"""
    # [1, 3, H, W] -> [3, H, W] -> [H, W, 3]
    img = tensor.squeeze(0).cpu().detach()
    img = img.clamp(0, 1)  # 0-1にクリップ
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype("uint8"))


def get_random_images(data_dir: str, num_images: int = 4) -> list:
    """ディレクトリからランダムに画像を選択"""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        f for f in Path(data_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    return random.sample(images, min(num_images, len(images)))


def create_comparison_figure(
    images_a: list,
    images_b: list,
    model: DeepfakeModel,
    device: torch.device,
    output_path: str,
):
    """
    顔交換の比較図を作成

    表示内容:
    - Row 1: 人物Aの元画像
    - Row 2: 人物Aの再構築
    - Row 3: A→B 顔交換（Aの表情でBの顔）
    - Row 4: 人物Bの元画像
    - Row 5: 人物Bの再構築
    - Row 6: B→A 顔交換（Bの表情でAの顔）
    """
    num_samples = min(len(images_a), len(images_b), 4)

    fig, axes = plt.subplots(6, num_samples, figsize=(3 * num_samples, 18))
    fig.suptitle("Deepfake 顔交換デモ", fontsize=16, fontweight="bold")

    row_labels = [
        "人物A（元画像）",
        "人物A（再構築）",
        "A→B 顔交換",
        "人物B（元画像）",
        "人物B（再構築）",
        "B→A 顔交換",
    ]

    with torch.no_grad():
        for i in range(num_samples):
            # 画像を読み込み
            img_a = load_image(str(images_a[i])).to(device)
            img_b = load_image(str(images_b[i])).to(device)

            # モデルで処理
            outputs = model.get_training_outputs(img_a, img_b)

            # 各結果を表示
            results = [
                img_a,                    # 人物A元画像
                outputs["recon_a"],       # 人物A再構築
                outputs["swap_a_to_b"],   # A→B顔交換
                img_b,                    # 人物B元画像
                outputs["recon_b"],       # 人物B再構築
                outputs["swap_b_to_a"],   # B→A顔交換
            ]

            for row, tensor in enumerate(results):
                img = tensor_to_image(tensor)
                axes[row, i].imshow(img)
                axes[row, i].axis("off")

                # 左端の列にラベルを追加
                if i == 0:
                    axes[row, i].set_ylabel(row_labels[row], fontsize=10, rotation=0,
                                           labelpad=80, ha="right", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"結果を保存しました: {output_path}")
    plt.close()


def create_single_swap_demo(
    image_a_path: str,
    image_b_path: str,
    model: DeepfakeModel,
    device: torch.device,
    output_path: str,
):
    """
    単一画像ペアの顔交換デモ
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("顔交換デモ: 表情の転送", fontsize=14, fontweight="bold")

    with torch.no_grad():
        img_a = load_image(image_a_path).to(device)
        img_b = load_image(image_b_path).to(device)

        outputs = model.get_training_outputs(img_a, img_b)

        # 上段: 人物A関連
        axes[0, 0].imshow(tensor_to_image(img_a))
        axes[0, 0].set_title("人物A（入力）")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(tensor_to_image(outputs["recon_a"]))
        axes[0, 1].set_title("人物A（再構築）")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(tensor_to_image(outputs["swap_a_to_b"]))
        axes[0, 2].set_title("A→B（Aの表情でBの顔）")
        axes[0, 2].axis("off")

        # 下段: 人物B関連
        axes[1, 0].imshow(tensor_to_image(img_b))
        axes[1, 0].set_title("人物B（入力）")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(tensor_to_image(outputs["recon_b"]))
        axes[1, 1].set_title("人物B（再構築）")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(tensor_to_image(outputs["swap_b_to_a"]))
        axes[1, 2].set_title("B→A（Bの表情でAの顔）")
        axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"結果を保存しました: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Deepfake 顔交換デモ")
    print("=" * 60)

    # 設定
    checkpoint_path = "checkpoints/best.pt"
    data_a_dir = "data/person_a"
    data_b_dir = "data/person_b"
    output_dir = "outputs"

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # チェックポイントの確認
    if not os.path.exists(checkpoint_path):
        print(f"エラー: チェックポイントが見つかりません: {checkpoint_path}")
        print("先にモデルを学習してください: python train.py --data-a data/person_a --data-b data/person_b")
        return

    # デバイスとモデルの準備
    device = get_device()
    print(f"デバイス: {device}")

    print("モデルを読み込み中...")
    model = load_model(checkpoint_path, device)
    print("モデル読み込み完了")

    # ランダムな画像を選択
    print("\n画像を選択中...")
    images_a = get_random_images(data_a_dir, 4)
    images_b = get_random_images(data_b_dir, 4)

    print(f"人物A: {len(images_a)}枚")
    print(f"人物B: {len(images_b)}枚")

    # 比較図を作成
    print("\n顔交換を実行中...")

    # 1. 複数画像の比較
    create_comparison_figure(
        images_a, images_b, model, device,
        os.path.join(output_dir, "swap_comparison.png")
    )

    # 2. 単一ペアのデモ
    create_single_swap_demo(
        str(images_a[0]), str(images_b[0]), model, device,
        os.path.join(output_dir, "swap_single.png")
    )

    print("\n" + "=" * 60)
    print("完了！")
    print(f"結果は {output_dir}/ フォルダに保存されました")
    print("=" * 60)


if __name__ == "__main__":
    main()
