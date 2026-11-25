# -*- coding: utf-8 -*-
"""
QUY TRÌNH 3: XÂY DỰNG MÔ HÌNH CNN CƠ BẢN VÀ MOBILENETV2 (TRANSFER LEARNING)
- Mô hình 1: Custom CNN tự xây dựng với các khối Conv2D, BatchNormalization, MaxPooling2D, Dropout và Dense.
- Mô hình 2: MobileNetV2 sử dụng trọng số tiền huấn luyện trên ImageNet (ImageNet weights),
  thay thế lớp phân loại đầu ra cho 5 nhóm hoa và hỗ trợ Fine-tuning các lớp cuối.
"""
import os
import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras import layers, models

from config import (
    FIGURES_DIR, RESULTS_DIR, CLASS_NAMES,
    IMG_HEIGHT, IMG_WIDTH, RANDOM_SEED
)

tf.random.set_seed(RANDOM_SEED)


def build_data_augmentation():
    """Khối tăng cường dữ liệu (chỉ kích hoạt khi training=True)."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.08, 0.08),
    ], name="data_augmentation")


def build_custom_cnn(num_classes=len(CLASS_NAMES)):
    """
    Xây dựng Mô hình 1: Mạng CNN cơ bản (Custom CNN) gồm 4 khối tích chập - gộp
    và khối kết nối đầy đủ (Fully Connected / Dense).
    """
    aug = build_data_augmentation()

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_rgb_image")
    x = aug(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(x)

    # Khối tích chập 1 (160x160 -> 80x80)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_32")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Khối tích chập 2 (80x80 -> 40x40)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_64")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Khối tích chập 3 (40x40 -> 20x20)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_128")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Khối tích chập 4 (20x20 -> 10x10)
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4_256")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Khối phân loại (Classification Head)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout_04")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="flower_classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_CNN_Flowers")
    return model


def build_mobilenetv2_transfer(num_classes=len(CLASS_NAMES)):
    """
    Xây dựng Mô hình 2: MobileNetV2 Transfer Learning sử dụng trọng số ImageNet.
    Trả về (model, base_model) để phục vụ huấn luyện 2 pha:
      - Pha 1: Feature Extraction (base_model.trainable = False)
      - Pha 2: Fine-Tuning (mở khóa 30 lớp cuối của base_model)
    """
    aug = build_data_augmentation()

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_rgb_image")
    x = aug(inputs)
    # Chuẩn hóa pixel từ [0, 255] sang [-1, 1] đúng chuẩn MobileNetV2
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.35, name="head_dropout1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_128")(x)
    x = layers.Dropout(0.25, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="flower_classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer_Flowers")
    return model, base_model


def unfreeze_mobilenetv2_top_layers(base_model, num_unfreeze=30):
    """Mở khóa num_unfreeze lớp cuối của MobileNetV2 để Fine-tuning, giữ đóng băng BatchNormalization."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[-num_unfreeze:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def count_model_params(model):
    total_params = int(model.count_params())
    trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def get_model_summary_str(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda line: stream.write(line + "\n"))
    return stream.getvalue()


def plot_architecture_diagrams(cnn_params, mnet_p1_params, mnet_p2_params):
    """Vẽ sơ đồ so sánh kiến trúc trực quan giữa Custom CNN và MobileNetV2 Transfer Learning."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5), dpi=300)

    def draw_blocks(ax, title, blocks, footer_text):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(blocks) * 1.35 + 1.8)
        ax.axis("off")
        ax.text(5, len(blocks) * 1.35 + 1.2, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#2B4C7E", edgecolor="none"),
                color="white")

        top_y = len(blocks) * 1.35 + 0.1
        for i, (b_name, b_desc, b_color) in enumerate(blocks):
            y = top_y - (i + 1) * 1.25
            rect = patches.FancyBboxPatch(
                (1.0, y), 8.0, 0.80,
                boxstyle="round,pad=0.08,rounding_size=0.14",
                facecolor=b_color, edgecolor="#333333", linewidth=1.1
            )
            ax.add_patch(rect)
            ax.text(5.0, y + 0.52, b_name, ha="center", va="center", fontsize=10, fontweight="bold", color="#111111")
            ax.text(5.0, y + 0.22, b_desc, ha="center", va="center", fontsize=8.8, color="#222222")
            if i < len(blocks) - 1:
                y_next_top = (top_y - (i + 2) * 1.25) + 0.80 + 0.09
                ax.annotate(
                    "", xy=(5.0, y_next_top), xytext=(5.0, y - 0.09),
                    arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.7, shrinkA=0, shrinkB=0)
                )

        ax.text(5.0, 0.25, footer_text, ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F6F9", edgecolor="#888888"))

    cnn_blocks = [
        ("Input & Data Augmentation", "Ảnh RGB 160x160x3 | Flip, Rotation, Zoom, Shift | Rescale [0, 1]", "#D9E8F5"),
        ("Conv Block 1 (32 Filters)", "Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 80x80x32", "#B3D4FF"),
        ("Conv Block 2 (64 Filters)", "Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 40x40x64", "#B3D4FF"),
        ("Conv Block 3 (128 Filters)", "Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 20x20x128", "#94C2FF"),
        ("Conv Block 4 (256 Filters)", "Conv2D(256, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 10x10x256", "#75B0FF"),
        ("Global Average Pooling", "GlobalAveragePooling2D (10x10x256 -> Vector 256 chiều)", "#FFE0B2"),
        ("Fully Connected + Dropout", "Dense(256, ReLU) + Dropout(0.4)", "#FFCC80"),
        ("Output Softmax Layer", "Dense(5, Softmax) -> [daisy, dandelion, roses, sunflowers, tulips]", "#A5D6A7"),
    ]

    mnet_blocks = [
        ("Input & Data Augmentation", "Ảnh RGB 160x160x3 | Flip, Rotation, Zoom, Shift | Rescale [-1, 1]", "#D9E8F5"),
        ("MobileNetV2 Initial Conv", "Conv2D(32, stride=2) + BatchNorm + ReLU6 -> 80x80x32 (ImageNet)", "#E1BEE7"),
        ("17 Inverted Residual Blocks", "Depthwise Separable Conv + Linear Bottlenecks (154 layers)", "#CE93D8"),
        ("Feature Extractor Output", "Conv 1x1 (1280 channels) + BatchNorm + ReLU6 -> 5x5x1280", "#BA68C8"),
        ("Phase 2 Fine-Tuning Zone", "Mở khóa 30 lớp cuối (Trainable với lr=1e-4, giữ đóng băng BN)", "#F8BBD0"),
        ("Global Average Pooling", "GlobalAveragePooling2D (5x5x1280 -> Vector 1280 chiều)", "#FFE0B2"),
        ("Custom Classification Head", "BatchNorm + Dropout(0.35) + Dense(128, ReLU) + Dropout(0.25)", "#FFCC80"),
        ("Output Softmax Layer", "Dense(5, Softmax) -> [daisy, dandelion, roses, sunflowers, tulips]", "#A5D6A7"),
    ]

    draw_blocks(
        ax1,
        "MÔ HÌNH 1: CUSTOM CNN (HUẤN LUYỆN TỪ ĐẦU)",
        cnn_blocks,
        f"Tổng tham số: {cnn_params['total_params']:,} | Trainable: {cnn_params['trainable_params']:,}"
    )

    draw_blocks(
        ax2,
        "MÔ HÌNH 2: MOBILENETV2 (TRANSFER LEARNING + FINE-TUNING)",
        mnet_blocks,
        f"Tổng tham số: {mnet_p2_params['total_params']:,} | Pha 1 Trainable: {mnet_p1_params['trainable_params']:,} | Pha 2 Trainable: {mnet_p2_params['trainable_params']:,}"
    )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig8_model_architectures.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 3] Đã lưu sơ đồ kiến trúc 2 mô hình: {out_path}")


def run_quy_trinh_3():
    cnn_model = build_custom_cnn()
    cnn_params = count_model_params(cnn_model)
    cnn_summary = get_model_summary_str(cnn_model)

    mnet_model, base_model = build_mobilenetv2_transfer()
    mnet_p1_params = count_model_params(mnet_model)
    mnet_p1_summary = get_model_summary_str(mnet_model)

    unfreeze_mobilenetv2_top_layers(base_model, num_unfreeze=30)
    mnet_p2_params = count_model_params(mnet_model)
    mnet_p2_summary = get_model_summary_str(mnet_model)

    plot_architecture_diagrams(cnn_params, mnet_p1_params, mnet_p2_params)

    info = {
        "custom_cnn": {
            "name": cnn_model.name,
            "input_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
            "num_layers": len(cnn_model.layers),
            "params": cnn_params,
            "summary": cnn_summary,
        },
        "mobilenetv2": {
            "name": mnet_model.name,
            "input_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
            "base_model_layers": len(base_model.layers),
            "phase1_params": mnet_p1_params,
            "phase2_params": mnet_p2_params,
            "phase1_summary": mnet_p1_summary,
            "phase2_summary": mnet_p2_summary,
        },
    }

    out_json = os.path.join(RESULTS_DIR, "quy_trinh_3_models_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[Quy trình 3] Hoàn tất! Custom CNN params: {cnn_params['total_params']:,} | "
          f"MobileNetV2 params: {mnet_p2_params['total_params']:,}")
    return info


if __name__ == "__main__":
    run_quy_trinh_3()
