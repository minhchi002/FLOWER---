# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH ỨNG DỤNG NHẬN DIỆN VÀ PHÂN LOẠI HÌNH ẢNH CÁC LOÀI HOA
(Sử dụng đồng thời Mô hình 1: Custom CNN và Mô hình 2: MobileNetV2 Transfer Learning)

Cách sử dụng:
  1. Khởi chạy Giao diện Đồ họa (GUI Tkinter):
       python app_nhan_dien_hoa.py
  2. Dự đoán nhanh một ảnh qua dòng lệnh (CLI):
       python app_nhan_dien_hoa.py --image "data/flower_photos/roses/12240303_80d87f77a3_n.jpg"
  3. Xuất ảnh minh họa giao diện ứng dụng (phục vụ báo cáo):
       python app_nhan_dien_hoa.py --export-gui-preview
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import sys
import time
import argparse
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR,
    CLASS_NAMES, CLASS_NAMES_VI, CLASS_COLORS, IMG_SIZE
)


def load_trained_models(warmup=True):
    """Tải hai mô hình đã huấn luyện từ thư mục models/ và warm-up để đo chính xác thời gian suy luận."""
    print("[Ứng dụng] Đang khởi động TensorFlow và nạp 2 mô hình (Custom CNN & MobileNetV2)...", flush=True)
    import tensorflow as tf
    cnn_path = os.path.join(MODELS_DIR, "custom_cnn_best.keras")
    mnet_path = os.path.join(MODELS_DIR, "mobilenetv2_best.keras")
    if not os.path.exists(cnn_path) or not os.path.exists(mnet_path):
        raise FileNotFoundError(
            "Chưa tìm thấy file trọng số mô hình trong thư mục 'models/'. "
            "Vui lòng chạy 'python src/quy_trinh_4_huan_luyen_mo_hinh.py' trước."
        )
    cnn_model = tf.keras.models.load_model(cnn_path)
    mnet_model = tf.keras.models.load_model(mnet_path)
    if warmup:
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = cnn_model.predict(dummy, verbose=0)
        _ = mnet_model.predict(dummy, verbose=0)
    print("[Ứng dụng] Nạp mô hình hoàn tất!", flush=True)
    return cnn_model, mnet_model


def predict_flower(image_path, cnn_model, mnet_model):
    """Tiền xử lý ảnh hoa đầu vào và trả về kết quả dự đoán từ cả 2 mô hình."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy tệp hình ảnh: {image_path}")

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        resized = rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)

    batch = np.expand_dims(arr, axis=0)

    # Dự đoán bằng Custom CNN
    t0 = time.perf_counter()
    p_cnn = cnn_model.predict(batch, verbose=0)[0]
    t_cnn_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    idx_cnn = int(np.argmax(p_cnn))

    # Dự đoán bằng MobileNetV2
    t1 = time.perf_counter()
    p_mnet = mnet_model.predict(batch, verbose=0)[0]
    t_mnet_ms = round((time.perf_counter() - t1) * 1000.0, 2)
    idx_mnet = int(np.argmax(p_mnet))

    return {
        "image_path": image_path,
        "filename": os.path.basename(image_path),
        "original_size": (orig_w, orig_h),
        "custom_cnn": {
            "class_en": CLASS_NAMES[idx_cnn],
            "class_vi": CLASS_NAMES_VI[CLASS_NAMES[idx_cnn]],
            "confidence_pct": round(float(p_cnn[idx_cnn]) * 100.0, 2),
            "latency_ms": t_cnn_ms,
            "probs": {CLASS_NAMES[i]: round(float(p_cnn[i]) * 100.0, 2) for i in range(len(CLASS_NAMES))},
        },
        "mobilenetv2": {
            "class_en": CLASS_NAMES[idx_mnet],
            "class_vi": CLASS_NAMES_VI[CLASS_NAMES[idx_mnet]],
            "confidence_pct": round(float(p_mnet[idx_mnet]) * 100.0, 2),
            "latency_ms": t_mnet_ms,
            "probs": {CLASS_NAMES[i]: round(float(p_mnet[i]) * 100.0, 2) for i in range(len(CLASS_NAMES))},
        },
    }


def print_cli_report(result):
    """In kết quả dự đoán chi tiết ra màn hình console."""
    print("\n" + "=" * 78)
    print("  KẾT QUẢ NHẬN DIỆN VÀ PHÂN LOẠI HÌNH ẢNH HOA (CNN vs MOBILENETV2)")
    print("=" * 78)
    print(f"  - Tệp hình ảnh    : {result['image_path']}")
    print(f"  - Kích thước gốc  : {result['original_size'][0]} x {result['original_size'][1]} pixels")
    print(f"  - Kích thước chuẩn: {IMG_SIZE[0]} x {IMG_SIZE[1]} x 3 (RGB)")
    print("-" * 78)

    for key, title in [
        ("custom_cnn", "MÔ HÌNH 1: CUSTOM CNN (TỰ XÂY DỰNG)"),
        ("mobilenetv2", "MÔ HÌNH 2: MOBILENETV2 (TRANSFER LEARNING)"),
    ]:
        info = result[key]
        print(f"  [{title}]")
        print(f"    + Loài hoa dự đoán : {info['class_vi']}")
        print(f"    + Độ tin cậy       : {info['confidence_pct']:.2f}%")
        print(f"    + Thời gian xử lý  : {info['latency_ms']:.2f} ms")
        print("    + Phân bố xác suất 5 lớp:")
        for cname in CLASS_NAMES:
            p = info["probs"][cname]
            bar = "█" * int(round(p / 4))
            print(f"        * {CLASS_NAMES_VI[cname]:<30}: {p:6.2f}%  {bar}")
        print("-" * 78)


def export_gui_preview_image(cnn_model, mnet_model):
    """Xuất hình mô phỏng giao diện ứng dụng nhận diện hoa sắc nét, không chồng chữ phục vụ báo cáo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    # Warm-up 2 mô hình để đo chính xác độ trễ suy luận ổn định (loại bỏ cold-start tracing)
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    _ = cnn_model.predict(dummy, verbose=0)
    _ = mnet_model.predict(dummy, verbose=0)

    sample_dir = os.path.join(DATA_DIR, "sunflowers")
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".jpg")])
    sample_path = os.path.join(sample_dir, sample_files[3])
    res = predict_flower(sample_path, cnn_model, mnet_model)

    fig = plt.figure(figsize=(15.5, 8.6), dpi=300, facecolor="#EAEFF5")

    # Thanh tiêu đề cửa sổ ứng dụng
    header_ax = fig.add_axes([0.02, 0.89, 0.96, 0.09])
    header_ax.axis("off")
    header_ax.add_patch(patches.FancyBboxPatch(
        (0.005, 0.05), 0.99, 0.90, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#1E3A8A", edgecolor="#1E3A8A"
    ))
    header_ax.text(0.5, 0.64, "ỨNG DỤNG NHẬN DIỆN VÀ PHÂN LOẠI HÌNH ẢNH CÁC LOÀI HOA",
                   ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    header_ax.text(0.5, 0.24, "Đề tài 10 – So sánh trực tiếp Mô hình 1 (Custom CNN) & Mô hình 2 (MobileNetV2 Transfer Learning)",
                   ha="center", va="center", fontsize=10, color="#DBEAFE")

    # Khung bên trái: Ảnh hoa đầu vào (cách rõ ràng khỏi nhãn cột phải)
    left_ax = fig.add_axes([0.025, 0.14, 0.30, 0.68])
    with Image.open(sample_path) as im:
        left_ax.imshow(im.convert("RGB"))
    left_ax.set_title(
        f"Ảnh đầu vào: {res['filename']}\nGốc: {res['original_size'][0]}×{res['original_size'][1]} px → 160×160×3",
        fontsize=10, fontweight="bold", pad=10, color="#111827"
    )
    left_ax.set_xticks([])
    left_ax.set_yticks([])

    labels_vi_short = [CLASS_NAMES_VI[c] for c in CLASS_NAMES]

    # Khung bên phải trên: Kết quả Mô hình 1 (Custom CNN) — đặt x0 = 0.54 để nhãn y-axis nằm hoàn toàn trong [0.34..0.53]
    ax_cnn = fig.add_axes([0.54, 0.53, 0.43, 0.30])
    c_info = res["custom_cnn"]
    c_probs = [c_info["probs"][c] for c in CLASS_NAMES]
    bars1 = ax_cnn.barh(labels_vi_short, c_probs, color=CLASS_COLORS, edgecolor="black", height=0.58)
    ax_cnn.set_xlim(0, 120)
    ax_cnn.invert_yaxis()
    ax_cnn.tick_params(axis="y", labelsize=9.5)
    ax_cnn.set_title(
        f"MÔ HÌNH 1: CUSTOM CNN | Dự đoán: {c_info['class_vi']} ({c_info['confidence_pct']:.2f}%) | {c_info['latency_ms']:.1f} ms",
        fontsize=9.8, fontweight="bold", color="#1E3A8A", pad=8
    )
    for b, v in zip(bars1, c_probs):
        ax_cnn.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")
    ax_cnn.grid(axis="x", linestyle="--", alpha=0.45)

    # Khung bên phải dưới: Kết quả Mô hình 2 (MobileNetV2)
    ax_mnet = fig.add_axes([0.54, 0.14, 0.43, 0.30])
    m_info = res["mobilenetv2"]
    m_probs = [m_info["probs"][c] for c in CLASS_NAMES]
    bars2 = ax_mnet.barh(labels_vi_short, m_probs, color=CLASS_COLORS, edgecolor="black", height=0.58)
    ax_mnet.set_xlim(0, 120)
    ax_mnet.invert_yaxis()
    ax_mnet.tick_params(axis="y", labelsize=9.5)
    ax_mnet.set_title(
        f"MÔ HÌNH 2: MOBILENETV2 | Dự đoán: {m_info['class_vi']} ({m_info['confidence_pct']:.2f}%) | {m_info['latency_ms']:.1f} ms",
        fontsize=9.8, fontweight="bold", color="#065F46", pad=8
    )
    for b, v in zip(bars2, m_probs):
        ax_mnet.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")
    ax_mnet.grid(axis="x", linestyle="--", alpha=0.45)

    # Thanh nút điều khiển bên dưới (căn khoảng cách tách bạch, không tràn chữ)
    footer_ax = fig.add_axes([0.02, 0.02, 0.96, 0.075])
    footer_ax.axis("off")
    footer_ax.set_xlim(0, 1)
    footer_ax.set_ylim(0, 1)
    btn_specs = [
        (0.01, 0.24, "[+] Tải ảnh hoa từ máy tính...", "#2563EB"),
        (0.28, 0.28, "[*] Chọn ảnh mẫu ngẫu nhiên (TF Flowers)", "#059669"),
        (0.59, 0.40, f"Trạng thái: Nhận diện thành công – {m_info['class_vi']}", "#374151"),
    ]
    for bx, bw, btxt, bcol in btn_specs:
        footer_ax.add_patch(patches.FancyBboxPatch(
            (bx, 0.12), bw, 0.76, boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor=bcol, edgecolor="none"
        ))
        footer_ax.text(bx + bw / 2, 0.50, btxt, ha="center", va="center", fontsize=9.2, fontweight="bold", color="white")

    out_path = os.path.join(FIGURES_DIR, "fig18_gui_app_screenshot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Ứng dụng] Đã xuất ảnh giao diện chương trình tại: {out_path}")
    return res


def launch_tkinter_gui(cnn_model, mnet_model):
    """Khởi chạy giao diện đồ họa Tkinter trên Windows."""
    import random
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Ứng dụng Nhận diện & Phân loại Hình ảnh Hoa - Custom CNN vs MobileNetV2 (Đề tài 10)")
    root.geometry("1080x680")
    root.configure(bg="#F3F4F6")

    header = tk.Frame(root, bg="#1E3A8A", height=65)
    header.pack(fill=tk.X)
    tk.Label(
        header,
        text="CHƯƠNG TRÌNH NHẬN DIỆN VÀ PHÂN LOẠI HÌNH ẢNH CÁC LOÀI HOA",
        font=("Segoe UI", 15, "bold"), fg="white", bg="#1E3A8A"
    ).pack(pady=(8, 2))
    tk.Label(
        header,
        text="So sánh trực quan Mô hình 1 (Custom CNN) và Mô hình 2 (MobileNetV2 Transfer Learning)",
        font=("Segoe UI", 10), fg="#DBEAFE", bg="#1E3A8A"
    ).pack(pady=(0, 8))

    main_frame = tk.Frame(root, bg="#F3F4F6")
    main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

    # Cột trái: hiển thị ảnh
    left_card = tk.LabelFrame(main_frame, text=" Hình ảnh hoa đầu vào ", font=("Segoe UI", 11, "bold"),
                              bg="white", fg="#111827", padx=12, pady=12)
    left_card.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 12))

    img_label = tk.Label(left_card, text="Chưa chọn ảnh", width=38, height=18, bg="#E5E7EB", fg="#4B5563",
                         font=("Segoe UI", 10))
    img_label.pack(pady=6)

    info_label = tk.Label(left_card, text="Kích thước gốc: --", font=("Segoe UI", 10), bg="white", fg="#374151")
    info_label.pack(pady=4)

    # Cột phải: kết quả 2 mô hình
    right_frame = tk.Frame(main_frame, bg="#F3F4F6")
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    cards = {}
    for key, title, color in [
        ("custom_cnn", "MÔ HÌNH 1: CUSTOM CNN (TỰ XÂY DỰNG)", "#1E3A8A"),
        ("mobilenetv2", "MÔ HÌNH 2: MOBILENETV2 (TRANSFER LEARNING)", "#065F46"),
    ]:
        card = tk.LabelFrame(right_frame, text=f" {title} ", font=("Segoe UI", 10, "bold"),
                             bg="white", fg=color, padx=14, pady=8)
        card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        pred_lbl = tk.Label(card, text="Kết quả dự đoán: --", font=("Segoe UI", 12, "bold"), bg="white", fg=color)
        pred_lbl.pack(anchor="w", pady=(2, 6))

        bars_dict = {}
        for cname in CLASS_NAMES:
            row_f = tk.Frame(card, bg="white")
            row_f.pack(fill=tk.X, pady=2)
            tk.Label(row_f, text=f"{CLASS_NAMES_VI[cname]:<30}", width=28, anchor="w",
                     font=("Segoe UI", 10), bg="white").pack(side=tk.LEFT)
            pb = ttk.Progressbar(row_f, orient="horizontal", length=260, mode="determinate", maximum=100.0)
            pb.pack(side=tk.LEFT, padx=8)
            val_lbl = tk.Label(row_f, text="0.00%", width=8, anchor="e", font=("Segoe UI", 10, "bold"), bg="white")
            val_lbl.pack(side=tk.LEFT)
            bars_dict[cname] = (pb, val_lbl)

        cards[key] = (pred_lbl, bars_dict)

    def update_with_image(fpath):
        try:
            res = predict_flower(fpath, cnn_model, mnet_model)
            with Image.open(fpath) as im:
                disp = im.convert("RGB").resize((290, 290), Image.Resampling.BILINEAR)
            tk_im = ImageTk.PhotoImage(disp)
            img_label.configure(image=tk_im, text="", width=290, height=290)
            img_label.image = tk_im
            info_label.configure(
                text=f"File: {res['filename']} | Gốc: {res['original_size'][0]}x{res['original_size'][1]} px"
            )

            for key in ["custom_cnn", "mobilenetv2"]:
                pred_lbl, bars_dict = cards[key]
                inf = res[key]
                pred_lbl.configure(
                    text=f"Dự đoán: {inf['class_vi']}  |  Độ tin cậy: {inf['confidence_pct']:.2f}%  |  {inf['latency_ms']:.1f} ms"
                )
                for cname in CLASS_NAMES:
                    p = inf["probs"][cname]
                    pb, val_lbl = bars_dict[cname]
                    pb["value"] = p
                    val_lbl.configure(text=f"{p:.2f}%")
        except Exception as e:
            messagebox.showerror("Lỗi dự đoán", str(e))

    def choose_file():
        fpath = filedialog.askopenfilename(
            title="Chọn hình ảnh hoa cần nhận diện",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if fpath:
            update_with_image(fpath)

    def pick_random_sample():
        c = random.choice(CLASS_NAMES)
        cdir = os.path.join(DATA_DIR, c)
        files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            update_with_image(os.path.join(cdir, random.choice(files)))

    btn_frame = tk.Frame(left_card, bg="white")
    btn_frame.pack(fill=tk.X, pady=(8, 2))
    tk.Button(btn_frame, text="Chọn ảnh từ máy tính...", command=choose_file,
              bg="#2563EB", fg="white", font=("Segoe UI", 10, "bold"), padx=10, pady=6, relief=tk.FLAT).pack(fill=tk.X, pady=3)
    tk.Button(btn_frame, text="Thử ảnh mẫu ngẫu nhiên", command=pick_random_sample,
              bg="#059669", fg="white", font=("Segoe UI", 10, "bold"), padx=10, pady=6, relief=tk.FLAT).pack(fill=tk.X, pady=3)

    pick_random_sample()
    print("[Ứng dụng] Đang hiển thị cửa sổ giao diện đồ họa (GUI)...", flush=True)
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Ứng dụng nhận diện và phân loại hình ảnh hoa (CNN & MobileNetV2)")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn tới hình ảnh hoa cần dự đoán (CLI mode)")
    parser.add_argument("--export-gui-preview", action="store_true", help="Xuất hình ảnh minh họa giao diện GUI")
    args = parser.parse_args()

    cnn_model, mnet_model = load_trained_models()

    if args.export_gui_preview:
        res = export_gui_preview_image(cnn_model, mnet_model)
        print_cli_report(res)
    elif args.image:
        res = predict_flower(args.image, cnn_model, mnet_model)
        print_cli_report(res)
    else:
        launch_tkinter_gui(cnn_model, mnet_model)


if __name__ == "__main__":
    main()
