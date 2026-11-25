# -*- coding: utf-8 -*-
"""
QUY TRÌNH 2: ĐỌC ẢNH, THAY ĐỔI KÍCH THƯỚC, CHUẨN HÓA PIXEL, TĂNG CƯỜNG DỮ LIỆU VÀ CHIA TẬP
- Hiển thị lưới 25 ảnh mẫu đại diện cho 5 loài hoa
- Thay đổi kích thước ảnh (Resize về 160x160) và chuẩn hóa pixel ([0, 1] và [-1, 1])
- Chia tập dữ liệu theo phương pháp phân tầng (Stratified Split: 70% Train - 15% Val - 15% Test)
- Áp dụng kỹ thuật tăng cường dữ liệu (Data Augmentation) trên tập huấn luyện
"""
import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, FIGURES_DIR, RESULTS_DIR,
    CLASS_NAMES, CLASS_NAMES_VI, CLASS_COLORS,
    IMG_HEIGHT, IMG_WIDTH, IMG_SIZE, BATCH_SIZE, RANDOM_SEED
)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def load_or_create_metadata():
    meta_csv = os.path.join(RESULTS_DIR, "all_images_metadata.csv")
    if os.path.exists(meta_csv):
        return pd.read_csv(meta_csv)
    from quy_trinh_1_khao_sat_du_lieu import run_quy_trinh_1
    df, _ = run_quy_trinh_1()
    return df


def plot_sample_images_grid(df):
    """Vẽ lưới 5x5 hiển thị các hình ảnh mẫu của 5 loài hoa kèm kích thước gốc."""
    fig, axes = plt.subplots(5, 5, figsize=(13, 13), dpi=300)
    np.random.seed(RANDOM_SEED)

    for row_idx, cls_name in enumerate(CLASS_NAMES):
        sub = df[df["class_name"] == cls_name].sample(n=5, random_state=RANDOM_SEED)
        for col_idx, (_, row) in enumerate(sub.iterrows()):
            ax = axes[row_idx, col_idx]
            with Image.open(row["filepath"]) as img:
                img_rgb = img.convert("RGB")
                ax.imshow(img_rgb)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(
                    CLASS_NAMES_VI[cls_name].replace(" (", "\n("),
                    fontsize=10.5, fontweight="bold", rotation=90, labelpad=10
                )
            ax.set_title(f"{row['width']}x{row['height']} px", fontsize=8.5, color="#333333")

    plt.suptitle("Mẫu hình ảnh thực tế của 5 nhóm hoa trong bộ dữ liệu TF Flowers",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(FIGURES_DIR, "fig4_sample_images_grid.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 2] Đã lưu lưới ảnh mẫu: {out_path}")


def plot_resize_and_normalization(df):
    """Minh họa quá trình Resize ảnh về 160x160 và chuẩn hóa giá trị pixel."""
    sample_row = df[df["class_name"] == "roses"].iloc[2]
    with Image.open(sample_row["filepath"]) as img:
        orig_img = img.convert("RGB")
        resized_img = orig_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)

    orig_arr = np.asarray(orig_img, dtype=np.float32)
    resized_arr = np.asarray(resized_img, dtype=np.float32)
    norm_01 = resized_arr / 255.0
    norm_11 = (resized_arr / 127.5) - 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

    axes[0].imshow(orig_img)
    axes[0].set_title(f"1. Ảnh gốc\n({orig_arr.shape[1]}x{orig_arr.shape[0]}x3, [0..255])",
                      fontsize=10.5, fontweight="bold")

    axes[1].imshow(resized_img)
    axes[1].set_title(f"2. Ảnh sau Resize\n({IMG_WIDTH}x{IMG_HEIGHT}x3, [0..255])",
                      fontsize=10.5, fontweight="bold")

    # Phân phối pixel sau chuẩn hóa [0, 1] (Custom CNN)
    colors = [("#E15759", "Red"), ("#59A14F", "Green"), ("#4E79A7", "Blue")]
    for ch, (c, lbl) in enumerate(colors):
        axes[2].hist(norm_01[:, :, ch].ravel(), bins=40, alpha=0.5, color=c, label=lbl, density=True)
    axes[2].set_title("3. Chuẩn hóa [0, 1] (CNN)\nx' = x / 255.0", fontsize=10.5, fontweight="bold")
    axes[2].set_xlabel("Giá trị pixel chuẩn hóa")
    axes[2].set_ylabel("Mật độ")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    # Phân phối pixel sau chuẩn hóa [-1, 1] (MobileNetV2)
    for ch, (c, lbl) in enumerate(colors):
        axes[3].hist(norm_11[:, :, ch].ravel(), bins=40, alpha=0.5, color=c, label=lbl, density=True)
    axes[3].set_title("4. Chuẩn hóa [-1, 1] (MobileNetV2)\nx' = (x / 127.5) - 1.0",
                      fontsize=10.5, fontweight="bold")
    axes[3].set_xlabel("Giá trị pixel chuẩn hóa")
    axes[3].legend(fontsize=8.5)
    axes[3].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_resize_and_normalization.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 2] Đã lưu biểu đồ Resize và Chuẩn hóa: {out_path}")


def plot_data_augmentation_samples(df):
    """Minh họa các phép biến đổi tăng cường dữ liệu (Data Augmentation)."""
    sample_row = df[df["class_name"] == "sunflowers"].iloc[5]
    with Image.open(sample_row["filepath"]) as img:
        base_img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)

    w, h = IMG_SIZE
    # Tạo 8 biến thể tăng cường dữ liệu rõ ràng, trực quan
    flip_h = base_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    rot_pos = base_img.rotate(22, resample=Image.Resampling.BILINEAR, expand=False)
    rot_neg = base_img.rotate(-18, resample=Image.Resampling.BILINEAR, expand=False)

    # Zoom in 18%
    crop_margin = int(w * 0.09)
    zoom_in = base_img.crop((crop_margin, crop_margin, w - crop_margin, h - crop_margin)).resize(IMG_SIZE, Image.Resampling.BILINEAR)

    # Dịch chuyển (Translation)
    shift_x, shift_y = int(w * 0.12), int(h * 0.08)
    translated = base_img.transform(IMG_SIZE, Image.Transform.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y), resample=Image.Resampling.BILINEAR)

    # Tăng/giảm độ sáng và độ tương phản
    bright_img = ImageEnhance.Brightness(base_img).enhance(1.25)
    combined_img = ImageEnhance.Contrast(
        flip_h.rotate(15, resample=Image.Resampling.BILINEAR).crop(
            (int(w * 0.06), int(h * 0.06), w - int(w * 0.06), h - int(h * 0.06))
        ).resize(IMG_SIZE, Image.Resampling.BILINEAR)
    ).enhance(1.15)

    variants = [
        (base_img, "1. Ảnh gốc (160x160)"),
        (flip_h, "2. Lật ngang (Horizontal Flip)"),
        (rot_pos, "3. Xoay +22° (Rotation +22°)"),
        (rot_neg, "4. Xoay -18° (Rotation -18°)"),
        (zoom_in, "5. Phóng to 18% (Random Zoom)"),
        (translated, "6. Dịch chuyển (Translation)"),
        (bright_img, "7. Tăng độ sáng +25% (Brightness)"),
        (combined_img, "8. Tổ hợp đa phép biến đổi"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 8.0), dpi=300)
    for ax, (im, title) in zip(axes.ravel(), variants):
        ax.imshow(im)
        ax.set_title(title, fontsize=10.2, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Minh họa các kỹ thuật Tăng cường dữ liệu (Data Augmentation) trên tập Huấn luyện",
                 fontsize=13.5, fontweight="bold", y=0.98)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, hspace=0.26, wspace=0.12)
    out_path = os.path.join(FIGURES_DIR, "fig6_data_augmentation_samples.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 2] Đã lưu hình minh họa Data Augmentation: {out_path}")


def stratified_split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Chia tập dữ liệu theo phương pháp phân tầng (Stratified Split)
    70% Train - 15% Validation - 15% Test.
    Đảm bảo thực hiện TRƯỚC khi áp dụng Data Augmentation để tránh rò rỉ dữ liệu (Data Leakage).
    """
    np.random.seed(RANDOM_SEED)
    df = df.copy()
    splits = np.empty(len(df), dtype=object)

    for cls_name in CLASS_NAMES:
        cls_indices = df[df["class_name"] == cls_name].index.to_numpy()
        perm = np.random.permutation(cls_indices)
        n_total = len(perm)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        splits[train_idx] = "train"
        splits[val_idx] = "val"
        splits[test_idx] = "test"

    df["split"] = splits
    split_csv = os.path.join(RESULTS_DIR, "dataset_split.csv")
    df.to_csv(split_csv, index=False, encoding="utf-8-sig")

    # Thống kê số lượng từng lớp theo tập
    pivot = pd.crosstab(df["class_name"], df["split"])[["train", "val", "test"]].reindex(CLASS_NAMES)
    pivot["total"] = pivot.sum(axis=1)

    split_records = []
    for cls_name in CLASS_NAMES:
        row = pivot.loc[cls_name]
        split_records.append({
            "class_name": cls_name,
            "class_vi": CLASS_NAMES_VI[cls_name],
            "train": int(row["train"]),
            "val": int(row["val"]),
            "test": int(row["test"]),
            "total": int(row["total"]),
            "train_pct": round(row["train"] / row["total"] * 100, 1),
            "val_pct": round(row["val"] / row["total"] * 100, 1),
            "test_pct": round(row["test"] / row["total"] * 100, 1),
        })

    split_stats = {
        "train_total": int((df["split"] == "train").sum()),
        "val_total": int((df["split"] == "val").sum()),
        "test_total": int((df["split"] == "test").sum()),
        "grand_total": int(len(df)),
        "img_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "per_class": split_records,
    }

    split_json = os.path.join(RESULTS_DIR, "quy_trinh_2_split_stats.json")
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)

    print(f"[Quy trình 2] Chia tập hoàn tất: Train={split_stats['train_total']}, "
          f"Val={split_stats['val_total']}, Test={split_stats['test_total']}")
    return df, split_stats


def plot_train_val_test_split(split_stats):
    """Vẽ biểu đồ phân bố dữ liệu trên 3 tập Train (70%), Validation (15%), Test (15%)."""
    per_cls = pd.DataFrame(split_stats["per_class"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    x = np.arange(len(CLASS_NAMES))
    width = 0.26

    b1 = ax1.bar(x - width, per_cls["train"], width, label=f"Train ({split_stats['train_total']} ảnh - 70%)",
                 color="#4E79A7", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x, per_cls["val"], width, label=f"Validation ({split_stats['val_total']} ảnh - 15%)",
                 color="#F28E2B", edgecolor="black", linewidth=0.6)
    b3 = ax1.bar(x + width, per_cls["test"], width, label=f"Test ({split_stats['test_total']} ảnh - 15%)",
                 color="#59A14F", edgecolor="black", linewidth=0.6)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = int(bar.get_height())
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 6, str(h),
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Số lượng hình ảnh", fontsize=10.5, fontweight="bold")
    ax1.set_title("Phân bố số lượng ảnh Train / Val / Test theo từng lớp", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(per_cls["train"]) * 1.16)
    ax1.legend(fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Biểu đồ tròn tổng thể 3 tập
    totals = [split_stats["train_total"], split_stats["val_total"], split_stats["test_total"]]
    labels = [
        f"Tập Huấn luyện (Train)\n{totals[0]:,} ảnh (70.0%)".replace(",", "."),
        f"Tập Kiểm định (Validation)\n{totals[1]:,} ảnh (15.0%)".replace(",", "."),
        f"Tập Kiểm tra (Test)\n{totals[2]:,} ảnh (15.0%)".replace(",", "."),
    ]
    wedges, texts, autotexts = ax2.pie(
        totals, labels=labels, autopct="%1.1f%%", startangle=120,
        colors=["#4E79A7", "#F28E2B", "#59A14F"],
        explode=(0.03, 0.03, 0.03),
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for a in autotexts:
        a.set_color("white")
        a.set_fontweight("bold")
        a.set_fontsize(10)
    ax2.set_title("Tỷ lệ chia tập dữ liệu Stratified Split (Tổng: 3.670 ảnh)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig7_train_val_test_split.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 2] Đã lưu biểu đồ chia tập: {out_path}")


def run_quy_trinh_2():
    df = load_or_create_metadata()
    plot_sample_images_grid(df)
    plot_resize_and_normalization(df)
    plot_data_augmentation_samples(df)
    df_split, split_stats = stratified_split_dataset(df)
    plot_train_val_test_split(split_stats)
    return df_split, split_stats


if __name__ == "__main__":
    run_quy_trinh_2()
