# -*- coding: utf-8 -*-
"""
QUY TRÌNH 4: HUẤN LUYỆN HAI MÔ HÌNH VÀ THEO DÕI ACCURACY, LOSS
- Nạp dữ liệu theo đúng phân tập Stratified Split (Train: 2.569, Val: 551, Test: 550)
- Huấn luyện Mô hình 1: Custom CNN từ đầu (20 Epochs)
- Huấn luyện Mô hình 2: MobileNetV2 Transfer Learning
    + Giai đoạn 1 (Feature Extraction): Đóng băng Base Model (8 Epochs, lr=1e-3)
    + Giai đoạn 2 (Fine-Tuning): Mở khóa 30 lớp cuối của MobileNetV2 (7 Epochs, lr=1e-4)
- Ghi nhận lịch sử Accuracy, Loss, thời gian huấn luyện mỗi Epoch và lưu trọng số tốt nhất
- Vẽ biểu đồ quá trình huấn luyện của từng mô hình và biểu đồ so sánh tốc độ hội tụ
"""
import os
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from config import (
    FIGURES_DIR, RESULTS_DIR, MODELS_DIR,
    CLASS_NAMES, IMG_SIZE, BATCH_SIZE, RANDOM_SEED
)
from quy_trinh_3_xay_dung_mo_hinh import (
    build_custom_cnn,
    build_mobilenetv2_transfer,
    unfreeze_mobilenetv2_top_layers,
    count_model_params,
)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class EpochTimerCallback(callbacks.Callback):
    """Ghi nhận thời gian thực thi (giây) của từng Epoch."""
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._start
        self.epoch_times.append(round(elapsed, 2))
        if logs is not None:
            logs["epoch_time_sec"] = round(elapsed, 2)


def load_split_arrays():
    """Đọc toàn bộ ảnh theo tập Train / Val / Test vào bộ nhớ RAM để tối ưu tốc độ huấn luyện."""
    split_csv = os.path.join(RESULTS_DIR, "dataset_split.csv")
    if not os.path.exists(split_csv):
        from quy_trinh_2_tien_xu_ly_tang_cuong import run_quy_trinh_2
        df, _ = run_quy_trinh_2()
    else:
        df = pd.read_csv(split_csv)

    cache_npz = os.path.join(RESULTS_DIR, "cached_flowers_160.npz")
    if os.path.exists(cache_npz):
        print(f"[Quy trình 4] Đang nạp dữ liệu ảnh đã cache từ: {cache_npz}")
        data = np.load(cache_npz)
        return (
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
            df
        )

    print("[Quy trình 4] Đang đọc và resize 3.670 ảnh về 160x160x3...")
    splits = {}
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        X_list = []
        y_list = []
        for _, row in sub.iterrows():
            with Image.open(row["filepath"]) as img:
                arr = np.asarray(img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)
            X_list.append(arr)
            y_list.append(int(row["class_id"]))
        splits[sp] = (np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32))

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    np.savez_compressed(
        cache_npz,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"[Quy trình 4] Đã lưu cache dữ liệu ảnh tại: {cache_npz}")
    return X_train, y_train, X_val, y_val, X_test, y_test, df


def make_tf_datasets(X_train, y_train, X_val, y_val):
    train_ds = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train_custom_cnn(train_ds, val_ds, epochs=20):
    print("\n" + "=" * 75)
    print(f"[Quy trình 4] BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH 1: CUSTOM CNN ({epochs} EPOCHS)")
    print("=" * 75)

    model = build_custom_cnn(num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt_path = os.path.join(MODELS_DIR, "custom_cnn_best.keras")
    timer_cb = EpochTimerCallback()
    cbs = [
        timer_cb,
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
    total_time = round(time.perf_counter() - t0, 2)

    # Đảm bảo lưu model tốt nhất
    if not os.path.exists(ckpt_path):
        model.save(ckpt_path)

    file_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2)
    params = count_model_params(model)

    history_dict = {
        "epoch": list(range(1, len(hist.history["loss"]) + 1)),
        "loss": [round(float(x), 4) for x in hist.history["loss"]],
        "accuracy": [round(float(x), 4) for x in hist.history["accuracy"]],
        "val_loss": [round(float(x), 4) for x in hist.history["val_loss"]],
        "val_accuracy": [round(float(x), 4) for x in hist.history["val_accuracy"]],
        "epoch_time_sec": timer_cb.epoch_times,
        "total_time_sec": total_time,
        "avg_epoch_sec": round(total_time / len(timer_cb.epoch_times), 2),
        "best_val_accuracy": round(float(max(hist.history["val_accuracy"])), 4),
        "best_epoch": int(np.argmax(hist.history["val_accuracy"]) + 1),
        "model_path": ckpt_path,
        "model_size_mb": file_mb,
        "params": params,
    }
    print(f"[Quy trình 4] Hoàn tất Custom CNN trong {total_time}s | "
          f"Best Val Acc: {history_dict['best_val_accuracy']*100:.2f}% (Epoch {history_dict['best_epoch']})")
    return history_dict


def train_mobilenetv2(train_ds, val_ds, phase1_epochs=8, phase2_epochs=7):
    print("\n" + "=" * 75)
    print(f"[Quy trình 4] BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH 2: MOBILENETV2 "
          f"(Pha 1: {phase1_epochs} Epochs + Pha 2 Fine-tuning: {phase2_epochs} Epochs)")
    print("=" * 75)

    model, base_model = build_mobilenetv2_transfer(num_classes=len(CLASS_NAMES))
    ckpt_path = os.path.join(MODELS_DIR, "mobilenetv2_best.keras")

    # PHA 1: Feature Extraction (đóng băng base_model)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    p1_params = count_model_params(model)
    timer_p1 = EpochTimerCallback()
    cbs_p1 = [
        timer_p1,
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    ]

    t0_p1 = time.perf_counter()
    hist_p1 = model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs, callbacks=cbs_p1, verbose=2)
    time_p1 = round(time.perf_counter() - t0_p1, 2)

    # PHA 2: Fine-Tuning 30 lớp cuối của MobileNetV2 với lr=1e-4
    print("\n--- Chuyển sang Giai đoạn 2: Mở khóa 30 lớp cuối của MobileNetV2 để Tinh chỉnh (Fine-Tuning) ---")
    unfreeze_mobilenetv2_top_layers(base_model, num_unfreeze=30)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    p2_params = count_model_params(model)
    best_p1_val_acc = float(max(hist_p1.history["val_accuracy"]))
    timer_p2 = EpochTimerCallback()
    cbs_p2 = [
        timer_p2,
        callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True,
            mode="max", initial_value_threshold=best_p1_val_acc, verbose=1
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    t0_p2 = time.perf_counter()
    hist_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=phase1_epochs,
        epochs=phase1_epochs + phase2_epochs,
        callbacks=cbs_p2,
        verbose=2
    )
    time_p2 = round(time.perf_counter() - t0_p2, 2)
    total_time = round(time_p1 + time_p2, 2)

    if not os.path.exists(ckpt_path):
        model.save(ckpt_path)

    file_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2)

    loss_all = [round(float(x), 4) for x in (hist_p1.history["loss"] + hist_p2.history["loss"])]
    acc_all = [round(float(x), 4) for x in (hist_p1.history["accuracy"] + hist_p2.history["accuracy"])]
    val_loss_all = [round(float(x), 4) for x in (hist_p1.history["val_loss"] + hist_p2.history["val_loss"])]
    val_acc_all = [round(float(x), 4) for x in (hist_p1.history["val_accuracy"] + hist_p2.history["val_accuracy"])]
    times_all = timer_p1.epoch_times + timer_p2.epoch_times

    history_dict = {
        "epoch": list(range(1, len(loss_all) + 1)),
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "loss": loss_all,
        "accuracy": acc_all,
        "val_loss": val_loss_all,
        "val_accuracy": val_acc_all,
        "epoch_time_sec": times_all,
        "phase1_time_sec": time_p1,
        "phase2_time_sec": time_p2,
        "total_time_sec": total_time,
        "avg_epoch_sec": round(total_time / len(times_all), 2),
        "best_val_accuracy": round(float(max(val_acc_all)), 4),
        "best_epoch": int(np.argmax(val_acc_all) + 1),
        "model_path": ckpt_path,
        "model_size_mb": file_mb,
        "phase1_params": p1_params,
        "phase2_params": p2_params,
    }
    print(f"[Quy trình 4] Hoàn tất MobileNetV2 trong {total_time}s | "
          f"Best Val Acc: {history_dict['best_val_accuracy']*100:.2f}% (Epoch {history_dict['best_epoch']})")
    return history_dict


def plot_training_curves(cnn_hist, mnet_hist):
    """Vẽ biểu đồ quá trình học của Custom CNN, MobileNetV2 và biểu đồ so sánh."""
    # 1. Biểu đồ Custom CNN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ep_cnn = cnn_hist["epoch"]
    ax1.plot(ep_cnn, [x * 100 for x in cnn_hist["accuracy"]], "o-", color="#4E79A7", lw=2, label="Train Accuracy")
    ax1.plot(ep_cnn, [x * 100 for x in cnn_hist["val_accuracy"]], "s-", color="#E15759", lw=2, label="Validation Accuracy")
    best_ep_cnn = cnn_hist["best_epoch"]
    best_acc_cnn = cnn_hist["best_val_accuracy"] * 100
    ax1.scatter([best_ep_cnn], [best_acc_cnn], color="gold", edgecolor="black", s=110, zorder=5,
                label=f"Đỉnh Val Acc: {best_acc_cnn:.2f}% (Ep {best_ep_cnn})")
    ax1.set_title("Biểu đồ Độ chính xác (Accuracy) - Mô hình 1: Custom CNN", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=10.5)
    ax1.legend(fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.45)

    ax2.plot(ep_cnn, cnn_hist["loss"], "o-", color="#4E79A7", lw=2, label="Train Loss")
    ax2.plot(ep_cnn, cnn_hist["val_loss"], "s-", color="#E15759", lw=2, label="Validation Loss")
    ax2.set_title("Biểu đồ Hàm mất mát (Loss) - Mô hình 1: Custom CNN", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10.5)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=10.5)
    ax2.legend(fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    p_cnn = os.path.join(FIGURES_DIR, "fig9_cnn_training_curves.png")
    plt.savefig(p_cnn, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Biểu đồ MobileNetV2 (có đường phân cách Pha 1 và Pha 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ep_mnet = mnet_hist["epoch"]
    p1_end = mnet_hist["phase1_epochs"]

    ax1.plot(ep_mnet, [x * 100 for x in mnet_hist["accuracy"]], "o-", color="#59A14F", lw=2, label="Train Accuracy")
    ax1.plot(ep_mnet, [x * 100 for x in mnet_hist["val_accuracy"]], "s-", color="#B07AA1", lw=2, label="Validation Accuracy")
    ax1.axvline(p1_end + 0.5, color="#333333", linestyle="--", lw=1.5, label="Bắt đầu Fine-Tuning (Pha 2)")
    best_ep_m = mnet_hist["best_epoch"]
    best_acc_m = mnet_hist["best_val_accuracy"] * 100
    ax1.scatter([best_ep_m], [best_acc_m], color="gold", edgecolor="black", s=110, zorder=5,
                label=f"Đỉnh Val Acc: {best_acc_m:.2f}% (Ep {best_ep_m})")
    ax1.set_title("Biểu đồ Độ chính xác (Accuracy) - Mô hình 2: MobileNetV2", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.45)

    ax2.plot(ep_mnet, mnet_hist["loss"], "o-", color="#59A14F", lw=2, label="Train Loss")
    ax2.plot(ep_mnet, mnet_hist["val_loss"], "s-", color="#B07AA1", lw=2, label="Validation Loss")
    ax2.axvline(p1_end + 0.5, color="#333333", linestyle="--", lw=1.5, label="Bắt đầu Fine-Tuning (Pha 2)")
    ax2.set_title("Biểu đồ Hàm mất mát (Loss) - Mô hình 2: MobileNetV2", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10.5)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=10.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    p_mnet = os.path.join(FIGURES_DIR, "fig10_mobilenetv2_training_curves.png")
    plt.savefig(p_mnet, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Biểu đồ đặt cạnh nhau so sánh trực tiếp Custom CNN vs MobileNetV2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ax1.plot(ep_cnn, [x * 100 for x in cnn_hist["val_accuracy"]], "o--", color="#4E79A7", lw=2, label="Custom CNN - Val Accuracy")
    ax1.plot(ep_cnn, [x * 100 for x in cnn_hist["accuracy"]], ":", color="#4E79A7", lw=1.4, label="Custom CNN - Train Accuracy")
    ax1.plot(ep_mnet, [x * 100 for x in mnet_hist["val_accuracy"]], "s-", color="#E15759", lw=2.2, label="MobileNetV2 - Val Accuracy")
    ax1.plot(ep_mnet, [x * 100 for x in mnet_hist["accuracy"]], "-.", color="#E15759", lw=1.4, label="MobileNetV2 - Train Accuracy")
    ax1.set_title("So sánh tốc độ hội tụ Accuracy: Custom CNN vs MobileNetV2", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.45)

    ax2.plot(ep_cnn, cnn_hist["val_loss"], "o--", color="#4E79A7", lw=2, label="Custom CNN - Val Loss")
    ax2.plot(ep_mnet, mnet_hist["val_loss"], "s-", color="#E15759", lw=2.2, label="MobileNetV2 - Val Loss")
    ax2.set_title("So sánh hàm mất mát Validation Loss: Custom CNN vs MobileNetV2", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10.5)
    ax2.set_ylabel("Validation Loss", fontsize=10.5)
    ax2.legend(fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    p_cmp = os.path.join(FIGURES_DIR, "fig11_training_comparison_curves.png")
    plt.savefig(p_cmp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 4] Đã lưu 3 biểu đồ quá trình huấn luyện:\n  - {p_cnn}\n  - {p_mnet}\n  - {p_cmp}")


def run_quy_trinh_4():
    X_train, y_train, X_val, y_val, X_test, y_test, df = load_split_arrays()
    train_ds, val_ds = make_tf_datasets(X_train, y_train, X_val, y_val)

    cnn_hist = train_custom_cnn(train_ds, val_ds, epochs=20)
    mnet_hist = train_mobilenetv2(train_ds, val_ds, phase1_epochs=8, phase2_epochs=7)

    plot_training_curves(cnn_hist, mnet_hist)

    out = {
        "custom_cnn": cnn_hist,
        "mobilenetv2": mnet_hist,
    }
    out_json = os.path.join(RESULTS_DIR, "quy_trinh_4_training_history.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Lưu bảng lịch sử epoch ra CSV để dễ tra cứu và đưa vào báo cáo
    pd.DataFrame({
        "epoch": cnn_hist["epoch"],
        "train_loss": cnn_hist["loss"],
        "train_acc": cnn_hist["accuracy"],
        "val_loss": cnn_hist["val_loss"],
        "val_acc": cnn_hist["val_accuracy"],
        "time_sec": cnn_hist["epoch_time_sec"],
    }).to_csv(os.path.join(RESULTS_DIR, "cnn_epoch_history.csv"), index=False)

    pd.DataFrame({
        "epoch": mnet_hist["epoch"],
        "phase": ["Feature Extraction"] * mnet_hist["phase1_epochs"] + ["Fine-Tuning"] * mnet_hist["phase2_epochs"],
        "train_loss": mnet_hist["loss"],
        "train_acc": mnet_hist["accuracy"],
        "val_loss": mnet_hist["val_loss"],
        "val_acc": mnet_hist["val_accuracy"],
        "time_sec": mnet_hist["epoch_time_sec"],
    }).to_csv(os.path.join(RESULTS_DIR, "mobilenetv2_epoch_history.csv"), index=False)

    print(f"[Quy trình 4] Đã lưu toàn bộ lịch sử huấn luyện tại: {out_json}")
    return out


if __name__ == "__main__":
    run_quy_trinh_4()
