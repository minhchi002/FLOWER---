# -*- coding: utf-8 -*-
"""
QUY TRÌNH 1: TẢI DỮ LIỆU TF FLOWERS, KIỂM TRA CẤU TRÚC THƯ MỤC VÀ PHÂN BỐ 5 LỚP
- Tải dữ liệu TF Flowers từ nguồn TensorFlow (nếu chưa tồn tại)
- Kiểm tra cấu trúc thư mục và quét ảnh lỗi / hỏng
- Thống kê số lượng và tỷ lệ phân bố 5 lớp hoa
- Phân tích phân bố kích thước ảnh (Width, Height, Aspect Ratio)
- Phân tích đặc trưng màu sắc RGB trung bình của 5 loài hoa
"""
import os
import json
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_DIR, FIGURES_DIR, RESULTS_DIR,
    CLASS_NAMES, CLASS_NAMES_VI, CLASS_COLORS
)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def download_tf_flowers_if_needed():
    """Tải và giải nén bộ dữ liệu TF Flowers nếu chưa có trên đĩa."""
    parent_dir = os.path.dirname(DATA_DIR)
    os.makedirs(parent_dir, exist_ok=True)
    tgz_path = os.path.join(parent_dir, "flower_photos.tgz")
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    if not os.path.exists(DATA_DIR):
        if not os.path.exists(tgz_path):
            print(f"[Quy trình 1] Đang tải dữ liệu TF Flowers từ: {url}")
            urllib.request.urlretrieve(url, tgz_path)
        print(f"[Quy trình 1] Đang giải nén {tgz_path}...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=parent_dir)
    print(f"[Quy trình 1] Thư mục dữ liệu sẵn sàng tại: {DATA_DIR}")


def inspect_dataset():
    """Kiểm tra cấu trúc thư mục, quét ảnh lỗi, thu thập metadata của toàn bộ ảnh."""
    download_tf_flowers_if_needed()

    records = []
    corrupted_files = []
    non_image_files = []

    for item in os.listdir(DATA_DIR):
        full_p = os.path.join(DATA_DIR, item)
        if os.path.isfile(full_p):
            non_image_files.append(item)

    for idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục lớp: {cls_dir}")

        files = sorted(os.listdir(cls_dir))
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                non_image_files.append(f"{cls_name}/{fname}")
                continue
            try:
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    w, h = img.size
                    mode = img.mode
                    img_rgb = img.convert("RGB")
                    # Thu nhỏ nhanh để tính thống kê màu RGB
                    thumb = np.asarray(img_rgb.resize((64, 64)), dtype=np.float32)
                    r_mean = float(thumb[:, :, 0].mean())
                    g_mean = float(thumb[:, :, 1].mean())
                    b_mean = float(thumb[:, :, 2].mean())
                    brightness = float(thumb.mean())
                file_kb = os.path.getsize(fpath) / 1024.0
                records.append({
                    "class_id": idx,
                    "class_name": cls_name,
                    "class_vi": CLASS_NAMES_VI[cls_name],
                    "filename": fname,
                    "filepath": fpath,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 4),
                    "mode": mode,
                    "file_kb": round(file_kb, 2),
                    "r_mean": round(r_mean, 2),
                    "g_mean": round(g_mean, 2),
                    "b_mean": round(b_mean, 2),
                    "brightness": round(brightness, 2),
                })
            except Exception as e:
                corrupted_files.append({"file": fpath, "error": str(e)})

    df = pd.DataFrame(records)
    print(f"[Quy trình 1] Tổng số ảnh hợp lệ: {len(df)} | Ảnh lỗi: {len(corrupted_files)}")
    return df, corrupted_files, non_image_files


def plot_class_distribution(df):
    """Vẽ biểu đồ cột và biểu đồ tròn thể hiện phân bố 5 lớp hoa."""
    counts = df["class_name"].value_counts().reindex(CLASS_NAMES)
    total = counts.sum()
    labels_vi = [CLASS_NAMES_VI[c] for c in CLASS_NAMES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Biểu đồ cột
    bars = ax1.bar(CLASS_NAMES, counts.values, color=CLASS_COLORS, edgecolor="black", linewidth=0.8, width=0.6)
    ax1.set_title("Số lượng hình ảnh theo từng lớp hoa (TF Flowers)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Tên lớp hoa (Class Name)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số lượng ảnh (Images)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(counts.values) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, cnt in zip(bars, counts.values):
        pct = cnt / total * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{cnt}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Biểu đồ tròn
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels_vi,
        autopct="%1.1f%%",
        startangle=140,
        colors=CLASS_COLORS,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
    for text in texts:
        text.set_fontsize(9.5)
    ax2.set_title(f"Tỷ lệ phân bố 5 lớp hoa (Tổng: {total:,} ảnh)", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig1_class_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 1] Đã lưu biểu đồ phân bố lớp: {out_path}")


def plot_image_dimensions(df):
    """Vẽ biểu đồ khảo sát kích thước ảnh (Chiều rộng, Chiều cao, Tỷ lệ khung hình)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. Scatter Width vs Height
    for cls_name, color in zip(CLASS_NAMES, CLASS_COLORS):
        sub = df[df["class_name"] == cls_name]
        axes[0].scatter(
            sub["width"], sub["height"],
            label=cls_name, color=color, alpha=0.55, s=18, edgecolors="none"
        )
    axes[0].set_title("Phân bố Chiều rộng vs Chiều cao ảnh", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Chiều rộng - Width (pixels)", fontsize=10.5)
    axes[0].set_ylabel("Chiều cao - Height (pixels)", fontsize=10.5)
    axes[0].legend(title="Lớp hoa", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # 2. Boxplot Width & Height theo lớp
    df_melt = df.melt(id_vars=["class_name"], value_vars=["width", "height"],
                      var_name="Dimension", value_name="Pixels")
    sns.boxplot(
        data=df_melt, x="class_name", y="Pixels", hue="Dimension",
        ax=axes[1], palette=["#4E79A7", "#F28E2B"], fliersize=2
    )
    axes[1].set_title("Phân vị kích thước ảnh theo từng lớp", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lớp hoa", fontsize=10.5)
    axes[1].set_ylabel("Kích thước (pixels)", fontsize=10.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Histogram Aspect Ratio
    for cls_name, color in zip(CLASS_NAMES, CLASS_COLORS):
        sub = df[df["class_name"] == cls_name]
        axes[2].hist(sub["aspect_ratio"], bins=25, alpha=0.45, label=cls_name, color=color)
    axes[2].axvline(df["aspect_ratio"].mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"TB: {df['aspect_ratio'].mean():.2f}")
    axes[2].set_title("Phân bố tỷ lệ khung hình (Width / Height)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Tỷ lệ khung hình (Aspect Ratio)", fontsize=10.5)
    axes[2].set_ylabel("Số lượng ảnh", fontsize=10.5)
    axes[2].legend(fontsize=8.5)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_image_dimensions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 1] Đã lưu biểu đồ kích thước ảnh: {out_path}")


def plot_color_characteristics(df):
    """Vẽ biểu đồ phân tích đặc trưng kênh màu RGB trung bình của 5 loài hoa."""
    rgb_means = df.groupby("class_name")[["r_mean", "g_mean", "b_mean", "brightness"]].mean().reindex(CLASS_NAMES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    ax1.bar(x - width, rgb_means["r_mean"], width, label="Kênh Đỏ (Red)", color="#E15759", edgecolor="black", linewidth=0.6)
    ax1.bar(x, rgb_means["g_mean"], width, label="Kênh Xanh lá (Green)", color="#59A14F", edgecolor="black", linewidth=0.6)
    ax1.bar(x + width, rgb_means["b_mean"], width, label="Kênh Xanh dương (Blue)", color="#4E79A7", edgecolor="black", linewidth=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Cường độ pixel trung bình (0 - 255)", fontsize=10.5)
    ax1.set_title("Cường độ màu R-G-B trung bình theo từng loài hoa", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 180)
    ax1.legend(fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Boxplot độ sáng
    sns.violinplot(
        data=df, x="class_name", y="brightness", hue="class_name", order=CLASS_NAMES,
        palette=CLASS_COLORS, ax=ax2, inner="quartile", legend=False
    )
    ax2.set_title("Phân bố độ sáng trung bình (Brightness) của 5 lớp hoa", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Lớp hoa", fontsize=10.5)
    ax2.set_ylabel("Độ sáng trung bình (0 - 255)", fontsize=10.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig3_color_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 1] Đã lưu biểu đồ phân tích màu sắc: {out_path}")


def run_quy_trinh_1():
    df, corrupted_files, non_image_files = inspect_dataset()
    plot_class_distribution(df)
    plot_image_dimensions(df)
    plot_color_characteristics(df)

    # Lưu bảng tổng hợp theo lớp
    summary_rows = []
    total_imgs = len(df)
    for cls_name in CLASS_NAMES:
        sub = df[df["class_name"] == cls_name]
        cnt = len(sub)
        summary_rows.append({
            "class_name": cls_name,
            "class_vi": CLASS_NAMES_VI[cls_name],
            "count": int(cnt),
            "percentage": round(cnt / total_imgs * 100, 2),
            "min_width": int(sub["width"].min()),
            "max_width": int(sub["width"].max()),
            "mean_width": round(float(sub["width"].mean()), 1),
            "min_height": int(sub["height"].min()),
            "max_height": int(sub["height"].max()),
            "mean_height": round(float(sub["height"].mean()), 1),
            "mean_aspect_ratio": round(float(sub["aspect_ratio"].mean()), 2),
            "mean_file_kb": round(float(sub["file_kb"].mean()), 2),
            "r_mean": round(float(sub["r_mean"].mean()), 2),
            "g_mean": round(float(sub["g_mean"].mean()), 2),
            "b_mean": round(float(sub["b_mean"].mean()), 2),
            "brightness_mean": round(float(sub["brightness"].mean()), 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_DIR, "class_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    metadata_csv = os.path.join(RESULTS_DIR, "all_images_metadata.csv")
    df.to_csv(metadata_csv, index=False, encoding="utf-8-sig")

    stats = {
        "total_images": int(total_imgs),
        "num_classes": len(CLASS_NAMES),
        "corrupted_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "non_image_files": non_image_files,
        "overall_width": {
            "min": int(df["width"].min()),
            "max": int(df["width"].max()),
            "mean": round(float(df["width"].mean()), 2),
            "std": round(float(df["width"].std()), 2),
            "median": float(df["width"].median()),
        },
        "overall_height": {
            "min": int(df["height"].min()),
            "max": int(df["height"].max()),
            "mean": round(float(df["height"].mean()), 2),
            "std": round(float(df["height"].std()), 2),
            "median": float(df["height"].median()),
        },
        "overall_file_kb": {
            "min": round(float(df["file_kb"].min()), 2),
            "max": round(float(df["file_kb"].max()), 2),
            "mean": round(float(df["file_kb"].mean()), 2),
            "total_mb": round(float(df["file_kb"].sum() / 1024.0), 2),
        },
        "class_summary": summary_rows,
    }

    stats_path = os.path.join(RESULTS_DIR, "quy_trinh_1_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[Quy trình 1] Hoàn tất! Đã lưu thống kê tại: {stats_path}")
    return df, stats


if __name__ == "__main__":
    run_quy_trinh_1()
