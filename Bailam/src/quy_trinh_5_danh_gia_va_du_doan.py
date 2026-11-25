# -*- coding: utf-8 -*-
"""
QUY TRÌNH 5: ĐÁNH GIÁ MÔ HÌNH BẰNG ACCURACY, PRECISION, RECALL, F1-SCORE,
CONFUSION MATRIX, ROC-AUC, PHÂN TÍCH LỖI VÀ XÂY DỰNG CHỨC NĂNG DỰ ĐOÁN LOÀI HOA
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
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    log_loss,
)
from sklearn.preprocessing import label_binarize

from config import (
    FIGURES_DIR, RESULTS_DIR, MODELS_DIR,
    CLASS_NAMES, CLASS_NAMES_VI, CLASS_COLORS,
    IMG_SIZE, BATCH_SIZE, RANDOM_SEED
)
from quy_trinh_4_huan_luyen_mo_hinh import load_split_arrays

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=RANDOM_SEED):
    """Tính khoảng tin cậy 95% (Bootstrap 95% CI) cho Accuracy và Macro F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        _, _, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        f1s.append(f1)
    return {
        "acc_ci95_low": round(float(np.percentile(accs, 2.5)), 4),
        "acc_ci95_high": round(float(np.percentile(accs, 97.5)), 4),
        "f1_ci95_low": round(float(np.percentile(f1s, 2.5)), 4),
        "f1_ci95_high": round(float(np.percentile(f1s, 97.5)), 4),
    }


def evaluate_single_model(model, X_test, y_test, model_label):
    """Đánh giá toàn diện một mô hình trên tập Test độc lập."""
    X_test_f = X_test.astype(np.float32)

    # Warm-up 1 batch
    _ = model.predict(X_test_f[:BATCH_SIZE], verbose=0)

    t0 = time.perf_counter()
    y_prob = model.predict(X_test_f, batch_size=BATCH_SIZE, verbose=0)
    infer_total_sec = time.perf_counter() - t0
    infer_ms_per_img = round((infer_total_sec / len(X_test)) * 1000.0, 2)
    fps = round(len(X_test) / infer_total_sec, 1)

    y_pred = np.argmax(y_prob, axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    test_loss = float(log_loss(y_test, y_prob, labels=list(range(len(CLASS_NAMES)))))

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))

    # ROC-AUC từng lớp và Macro
    y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
    roc_auc_per_class = {}
    fpr_dict, tpr_dict = {}, {}
    for i, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc_per_class[cname] = round(float(auc(fpr, tpr)), 4)
        fpr_dict[cname] = fpr
        tpr_dict[cname] = tpr
    macro_auc = round(float(np.mean(list(roc_auc_per_class.values()))), 4)

    ci95 = bootstrap_ci(y_test, y_pred)

    per_class_rows = []
    for i, cname in enumerate(CLASS_NAMES):
        per_class_rows.append({
            "class_id": i,
            "class_name": cname,
            "class_vi": CLASS_NAMES_VI[cname],
            "precision": round(float(prec_c[i]), 4),
            "recall": round(float(rec_c[i]), 4),
            "f1_score": round(float(f1_c[i]), 4),
            "roc_auc": roc_auc_per_class[cname],
            "support": int(sup_c[i]),
            "correct": int(cm[i, i]),
            "misclassified": int(sup_c[i] - cm[i, i]),
        })

    print(f"[Quy trình 5] {model_label}: Test Acc = {acc*100:.2f}% | "
          f"Macro F1 = {f1_m*100:.2f}% | Macro AUC = {macro_auc:.4f} | "
          f"Latency = {infer_ms_per_img} ms/ảnh")

    return {
        "model_label": model_label,
        "accuracy": round(acc, 4),
        "test_loss": round(test_loss, 4),
        "macro_precision": round(float(prec_m), 4),
        "macro_recall": round(float(rec_m), 4),
        "macro_f1": round(float(f1_m), 4),
        "weighted_precision": round(float(prec_w), 4),
        "weighted_recall": round(float(rec_w), 4),
        "weighted_f1": round(float(f1_w), 4),
        "macro_roc_auc": macro_auc,
        "ci95": ci95,
        "infer_total_sec": round(infer_total_sec, 3),
        "infer_ms_per_img": infer_ms_per_img,
        "infer_fps": fps,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_rows,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "fpr_dict": fpr_dict,
        "tpr_dict": tpr_dict,
    }


def plot_confusion_matrices(cnn_eval, mnet_eval):
    """Vẽ Ma trận nhầm lẫn (Confusion Matrix) của Custom CNN và MobileNetV2 đặt cạnh nhau."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300)

    for ax, ev, title, cmap in [
        (ax1, cnn_eval, f"Ma trận nhầm lẫn - Mô hình 1: Custom CNN\n(Test Accuracy: {cnn_eval['accuracy']*100:.2f}%)", "Blues"),
        (ax2, mnet_eval, f"Ma trận nhầm lẫn - Mô hình 2: MobileNetV2\n(Test Accuracy: {mnet_eval['accuracy']*100:.2f}%)", "Greens"),
    ]:
        cm = np.array(ev["confusion_matrix"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

        sns.heatmap(
            cm, annot=annot, fmt="", cmap=cmap, cbar=True, ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.8, linecolor="white", annot_kws={"size": 9.5, "weight": "bold"}
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Nhãn thực tế (True Label)", fontsize=10.5, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig12_confusion_matrices.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 5] Đã lưu Ma trận nhầm lẫn: {out_path}")


def plot_per_class_metrics(cnn_eval, mnet_eval):
    """Vẽ biểu đồ so sánh Precision, Recall, F1-Score trên từng lớp hoa giữa 2 mô hình."""
    cnn_df = pd.DataFrame(cnn_eval["per_class"])
    mnet_df = pd.DataFrame(mnet_eval["per_class"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
    metrics_info = [
        ("precision", "Precision (Độ chuẩn xác)"),
        ("recall", "Recall (Độ phủ / Độ nhạy)"),
        ("f1_score", "F1-Score (Trung bình điều hòa)"),
    ]

    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    for ax, (m_key, m_title) in zip(axes, metrics_info):
        v_cnn = cnn_df[m_key] * 100
        v_mnet = mnet_df[m_key] * 100

        b1 = ax.bar(x - width / 2, v_cnn, width, label="Custom CNN", color="#4E79A7", edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x + width / 2, v_mnet, width, label="MobileNetV2", color="#E15759", edgecolor="black", linewidth=0.6)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 112)
        ax.set_ylabel("Tỷ lệ (%)", fontsize=10)
        ax.set_title(m_title, fontsize=11.5, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.45)

    plt.suptitle("So sánh chi tiết Precision, Recall và F1-Score theo từng loài hoa trên tập Test",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig13_per_class_metrics_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 5] Đã lưu biểu đồ so sánh chỉ số từng lớp: {out_path}")


def plot_roc_curves(cnn_eval, mnet_eval):
    """Vẽ đường cong ROC đa lớp (One-vs-Rest) cho cả 2 mô hình."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for ax, ev, title in [
        (ax1, cnn_eval, f"Đường cong ROC đa lớp - Custom CNN (Macro AUC = {cnn_eval['macro_roc_auc']:.4f})"),
        (ax2, mnet_eval, f"Đường cong ROC đa lớp - MobileNetV2 (Macro AUC = {mnet_eval['macro_roc_auc']:.4f})"),
    ]:
        for cname, color, row in zip(CLASS_NAMES, CLASS_COLORS, ev["per_class"]):
            ax.plot(
                ev["fpr_dict"][cname],
                ev["tpr_dict"][cname],
                color=color, lw=2,
                label=f"{cname} (AUC = {row['roc_auc']:.4f})"
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ngẫu nhiên (AUC = 0.5000)")
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([0.0, 1.03])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig14_roc_curves_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 5] Đã lưu biểu đồ ROC-AUC: {out_path}")


def plot_misclassified_analysis(X_test, y_test, cnn_eval, mnet_eval):
    """
    Phân tích trực quan các trường hợp nhận diện sai (Error Analysis) ĐA DẠNG LỚP HOA:
    - Hàng 1: 4 ảnh thuộc 4 loài hoa khác nhau mà Custom CNN dự đoán SAI nhưng MobileNetV2 dự đoán ĐÚNG
    - Hàng 2: 4 ảnh thách thức thuộc 4 loài hoa khác nhau mà MobileNetV2 dự đoán SAI
    """
    y_pred_cnn = cnn_eval["y_pred"]
    y_prob_cnn = cnn_eval["y_prob"]
    y_pred_mnet = mnet_eval["y_pred"]
    y_prob_mnet = mnet_eval["y_prob"]

    def pick_diverse_indices(candidate_indices, count=4):
        picked = []
        used_classes = set()
        # Ưu tiên mỗi ảnh một lớp hoa khác nhau (tulips, roses, dandelion, sunflowers, daisy)
        for cls_pref in [4, 2, 1, 3, 0]:
            for idx in candidate_indices:
                if y_test[idx] == cls_pref and cls_pref not in used_classes:
                    picked.append(int(idx))
                    used_classes.add(cls_pref)
                    break
            if len(picked) == count:
                return picked
        for idx in candidate_indices:
            if int(idx) not in picked:
                picked.append(int(idx))
                if len(picked) == count:
                    break
        return picked

    cnn_wrong_mnet_right = np.where((y_pred_cnn != y_test) & (y_pred_mnet == y_test))[0]
    mnet_wrong = np.where(y_pred_mnet != y_test)[0]

    sel_row1 = pick_diverse_indices(cnn_wrong_mnet_right, 4)
    sel_row2 = pick_diverse_indices(mnet_wrong, 4)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.2), dpi=300)

    for col, idx in enumerate(sel_row1):
        ax = axes[0, col]
        ax.imshow(X_test[idx])
        true_c = CLASS_NAMES[y_test[idx]]
        cnn_c = CLASS_NAMES[y_pred_cnn[idx]]
        cnn_p = y_prob_cnn[idx, y_pred_cnn[idx]] * 100
        mnet_c = CLASS_NAMES[y_pred_mnet[idx]]
        mnet_p = y_prob_mnet[idx, y_pred_mnet[idx]] * 100

        ax.set_title(
            f"Thực tế: {true_c}\nCNN: {cnn_c} ({cnn_p:.1f}%) [SAI]\nMobileNetV2: {mnet_c} ({mnet_p:.1f}%) [ĐÚNG]",
            fontsize=9.2, fontweight="bold", color="#1B5E20", pad=8
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for col, idx in enumerate(sel_row2):
        ax = axes[1, col]
        ax.imshow(X_test[idx])
        true_c = CLASS_NAMES[y_test[idx]]
        cnn_c = CLASS_NAMES[y_pred_cnn[idx]]
        cnn_p = y_prob_cnn[idx, y_pred_cnn[idx]] * 100
        mnet_c = CLASS_NAMES[y_pred_mnet[idx]]
        mnet_p = y_prob_mnet[idx, y_pred_mnet[idx]] * 100

        ax.set_title(
            f"Thực tế: {true_c}\nCNN: {cnn_c} ({cnn_p:.1f}%)\nMobileNetV2: {mnet_c} ({mnet_p:.1f}%) [NHẦM]",
            fontsize=9.2, fontweight="bold", color="#B71C1C", pad=8
        )
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("CNN Sai -> MobileNetV2 Đúng", fontsize=10.5, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Các mẫu khó / Nhầm lẫn", fontsize=10.5, fontweight="bold", labelpad=10)

    plt.suptitle("Phân tích các trường hợp nhận diện sai đa dạng lớp hoa (Error Analysis) trên tập Test",
                 fontsize=13.5, fontweight="bold", y=0.98)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.87, bottom=0.03, hspace=0.42, wspace=0.14)
    out_path = os.path.join(FIGURES_DIR, "fig15_error_analysis_misclassified.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 5] Đã lưu hình phân tích lỗi nhận diện: {out_path}")


def predict_single_flower_image(image_path, cnn_model, mnet_model):
    """
    Chức năng nhập 1 hình ảnh hoa mới, tiền xử lý (Resize 160x160x3)
    và trả về tên nhóm hoa dự đoán cùng phân phối xác suất của cả 2 mô hình.
    """
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        orig_size = rgb_img.size
        resized = rgb_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)

    batch = np.expand_dims(arr, axis=0)

    t0 = time.perf_counter()
    prob_cnn = cnn_model.predict(batch, verbose=0)[0]
    ms_cnn = round((time.perf_counter() - t0) * 1000.0, 2)
    idx_cnn = int(np.argmax(prob_cnn))

    t1 = time.perf_counter()
    prob_mnet = mnet_model.predict(batch, verbose=0)[0]
    ms_mnet = round((time.perf_counter() - t1) * 1000.0, 2)
    idx_mnet = int(np.argmax(prob_mnet))

    return {
        "image_path": image_path,
        "original_size": orig_size,
        "custom_cnn": {
            "predicted_class": CLASS_NAMES[idx_cnn],
            "predicted_vi": CLASS_NAMES_VI[CLASS_NAMES[idx_cnn]],
            "confidence": round(float(prob_cnn[idx_cnn]) * 100, 2),
            "latency_ms": ms_cnn,
            "probabilities": {CLASS_NAMES[i]: round(float(prob_cnn[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
        },
        "mobilenetv2": {
            "predicted_class": CLASS_NAMES[idx_mnet],
            "predicted_vi": CLASS_NAMES_VI[CLASS_NAMES[idx_mnet]],
            "confidence": round(float(prob_mnet[idx_mnet]) * 100, 2),
            "latency_ms": ms_mnet,
            "probabilities": {CLASS_NAMES[i]: round(float(prob_mnet[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
        },
    }


def plot_prediction_demo_grid(df, cnn_model, mnet_model):
    """Minh họa kết quả dự đoán trên 5 ảnh hoa thực tế thuộc 5 loài khác nhau."""
    test_df = df[df["split"] == "test"]
    fig, axes = plt.subplots(5, 3, figsize=(14, 14.5), dpi=300,
                             gridspec_kw={"width_ratios": [1.1, 1.45, 1.45]})

    demo_results = []
    for row_idx, cname in enumerate(CLASS_NAMES):
        sub = test_df[test_df["class_name"] == cname]
        sample_row = sub.iloc[0]
        res = predict_single_flower_image(sample_row["filepath"], cnn_model, mnet_model)
        res["true_class"] = cname
        res["true_vi"] = CLASS_NAMES_VI[cname]
        demo_results.append(res)

        # Cột 1: Hình ảnh hoa đầu vào
        ax_img = axes[row_idx, 0]
        with Image.open(sample_row["filepath"]) as im:
            ax_img.imshow(im.convert("RGB"))
        ax_img.set_title(f"Ảnh đầu vào: {CLASS_NAMES_VI[cname]}\n({sample_row['filename']})",
                         fontsize=9.5, fontweight="bold")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Cột 2: Xác suất dự đoán của Custom CNN
        ax_cnn = axes[row_idx, 1]
        probs_cnn = [res["custom_cnn"]["probabilities"][c] for c in CLASS_NAMES]
        bars_c = ax_cnn.barh(CLASS_NAMES, probs_cnn, color=CLASS_COLORS, edgecolor="black", height=0.6)
        ax_cnn.set_xlim(0, 122)
        ax_cnn.invert_yaxis()
        pred_c = res["custom_cnn"]["predicted_class"]
        conf_c = res["custom_cnn"]["confidence"]
        ax_cnn.set_title(f"Custom CNN -> {pred_c} ({conf_c:.1f}%)", fontsize=9.5, fontweight="bold",
                         color="#1B5E20" if pred_c == cname else "#B71C1C")
        for b, v in zip(bars_c, probs_cnn):
            ax_cnn.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")
        ax_cnn.grid(axis="x", linestyle="--", alpha=0.4)

        # Cột 3: Xác suất dự đoán của MobileNetV2
        ax_m = axes[row_idx, 2]
        probs_m = [res["mobilenetv2"]["probabilities"][c] for c in CLASS_NAMES]
        bars_m = ax_m.barh(CLASS_NAMES, probs_m, color=CLASS_COLORS, edgecolor="black", height=0.6)
        ax_m.set_xlim(0, 122)
        ax_m.invert_yaxis()
        pred_m = res["mobilenetv2"]["predicted_class"]
        conf_m = res["mobilenetv2"]["confidence"]
        ax_m.set_title(f"MobileNetV2 -> {pred_m} ({conf_m:.1f}%)", fontsize=9.5, fontweight="bold",
                       color="#1B5E20" if pred_m == cname else "#B71C1C")
        for b, v in zip(bars_m, probs_m):
            ax_m.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")
        ax_m.grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle("Minh họa chức năng tải ảnh hoa mới và dự đoán xác suất trên 5 lớp (Custom CNN vs MobileNetV2)",
                 fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(FIGURES_DIR, "fig16_prediction_demo_grid.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 5] Đã lưu hình minh họa dự đoán ảnh mẫu: {out_path}")
    return demo_results


def plot_overall_model_comparison(cnn_eval, mnet_eval, train_hist):
    """Vẽ biểu đồ 2x2 tổng hợp so sánh 2 mô hình về Accuracy/F1, Thời gian Train, Tốc độ Suy luận và Dung lượng."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.4), dpi=300)
    w = 0.36

    # 1. Các chỉ số hiệu năng phân loại (%)
    ax1 = axes[0, 0]
    metric_labels = ["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro ROC-AUC"]
    cnn_vals = [
        cnn_eval["accuracy"] * 100,
        cnn_eval["macro_precision"] * 100,
        cnn_eval["macro_recall"] * 100,
        cnn_eval["macro_f1"] * 100,
        cnn_eval["macro_roc_auc"] * 100,
    ]
    mnet_vals = [
        mnet_eval["accuracy"] * 100,
        mnet_eval["macro_precision"] * 100,
        mnet_eval["macro_recall"] * 100,
        mnet_eval["macro_f1"] * 100,
        mnet_eval["macro_roc_auc"] * 100,
    ]
    x = np.arange(len(metric_labels))
    b1 = ax1.bar(x - w / 2, cnn_vals, w, label="Custom CNN", color="#4E79A7", edgecolor="black", linewidth=0.7)
    b2 = ax1.bar(x + w / 2, mnet_vals, w, label="MobileNetV2", color="#E15759", edgecolor="black", linewidth=0.7)
    for b in b1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=8.5)
    for b in b2:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991B1B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 114)
    ax1.set_ylabel("Tỷ lệ (%)", fontsize=10, fontweight="bold")
    ax1.set_title("1. So sánh hiệu năng phân loại trên tập Test (550 ảnh)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(fontsize=9.5, loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    # 2. Thời gian huấn luyện (phút & giây/epoch tách rõ đơn vị)
    ax2 = axes[0, 1]
    t_cnn_min = round(train_hist["custom_cnn"]["total_time_sec"] / 60.0, 2)
    t_mnet_min = round(train_hist["mobilenetv2"]["total_time_sec"] / 60.0, 2)
    ep_cnn_s = train_hist["custom_cnn"]["avg_epoch_sec"]
    ep_mnet_s = train_hist["mobilenetv2"]["avg_epoch_sec"]
    cats2 = ["Tổng thời gian Train (phút)", "Thời gian TB mỗi Epoch (giây)"]
    v_c2 = [t_cnn_min, ep_cnn_s]
    v_m2 = [t_mnet_min, ep_mnet_s]
    x2 = np.arange(len(cats2))
    b21 = ax2.bar(x2 - w / 2, v_c2, w, label="Custom CNN (20 Epochs)", color="#4E79A7", edgecolor="black", linewidth=0.7)
    b22 = ax2.bar(x2 + w / 2, v_m2, w, label="MobileNetV2 (15 Epochs)", color="#E15759", edgecolor="black", linewidth=0.7)
    for b, unit in zip(b21, ["phút", "giây"]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, f"{b.get_height():.2f}\n{unit}",
                 ha="center", va="bottom", fontsize=8.8)
    for b, unit in zip(b22, ["phút", "giây"]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, f"{b.get_height():.2f}\n{unit}",
                 ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#991B1B")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats2, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(v_c2 + v_m2) * 1.25)
    ax2.set_ylabel("Giá trị thời gian", fontsize=10, fontweight="bold")
    ax2.set_title("2. So sánh thời gian huấn luyện (MobileNetV2 nhanh hơn 3.27 lần)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    # 3. Tốc độ suy luận (Latency ms/ảnh & Thông lượng FPS)
    ax3 = axes[1, 0]
    cats3 = ["Độ trễ suy luận (ms / ảnh)", "Thông lượng xử lý (FPS - ảnh/giây)"]
    v_c3 = [cnn_eval["infer_ms_per_img"], cnn_eval["infer_fps"]]
    v_m3 = [mnet_eval["infer_ms_per_img"], mnet_eval["infer_fps"]]
    x3 = np.arange(len(cats3))
    b31 = ax3.bar(x3 - w / 2, v_c3, w, label="Custom CNN", color="#4E79A7", edgecolor="black", linewidth=0.7)
    b32 = ax3.bar(x3 + w / 2, v_m3, w, label="MobileNetV2", color="#E15759", edgecolor="black", linewidth=0.7)
    for b, unit in zip(b31, ["ms/ảnh", "FPS"]):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8, f"{b.get_height():.2f}\n{unit}",
                 ha="center", va="bottom", fontsize=8.8)
    for b, unit in zip(b32, ["ms/ảnh", "FPS"]):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8, f"{b.get_height():.2f}\n{unit}",
                 ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#991B1B")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats3, fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(v_c3 + v_m3) * 1.25)
    ax3.set_ylabel("Độ trễ (ms) / Thông lượng (FPS)", fontsize=10, fontweight="bold")
    ax3.set_title("3. So sánh tốc độ suy luận thời gian thực trên CPU", fontsize=11.5, fontweight="bold", pad=10)
    ax3.legend(fontsize=9.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.45)

    # 4. Dung lượng lưu trữ (MB) & Số lượng tham số (Triệu)
    ax4 = axes[1, 1]
    cats4 = ["Dung lượng tệp\n.keras (MB)", "Tổng số tham số\n(Triệu tham số)", "Tham số huấn luyện\n(Triệu tham số)"]
    v_c4 = [
        train_hist["custom_cnn"]["model_size_mb"],
        round(train_hist["custom_cnn"]["params"]["total_params"] / 1e6, 2),
        round(train_hist["custom_cnn"]["params"]["trainable_params"] / 1e6, 2),
    ]
    v_m4 = [
        train_hist["mobilenetv2"]["model_size_mb"],
        round(train_hist["mobilenetv2"]["phase2_params"]["total_params"] / 1e6, 2),
        round(train_hist["mobilenetv2"]["phase2_params"]["trainable_params"] / 1e6, 2),
    ]
    x4 = np.arange(len(cats4))
    b41 = ax4.bar(x4 - w / 2, v_c4, w, label="Custom CNN", color="#4E79A7", edgecolor="black", linewidth=0.7)
    b42 = ax4.bar(x4 + w / 2, v_m4, w, label="MobileNetV2", color="#E15759", edgecolor="black", linewidth=0.7)
    for b, unit in zip(b41, ["MB", "M", "M"]):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.2f} {unit}",
                 ha="center", va="bottom", fontsize=8.8)
    for b, unit in zip(b42, ["MB", "M", "M"]):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.2f} {unit}",
                 ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#991B1B")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(cats4, fontsize=9.8, fontweight="bold")
    ax4.set_ylim(0, max(v_c4 + v_m4) * 1.22)
    ax4.set_ylabel("Dung lượng (MB) / Số tham số (Triệu)", fontsize=10, fontweight="bold")
    ax4.set_title("4. So sánh dung lượng lưu trữ và độ phức tạp tham số", fontsize=11.5, fontweight="bold", pad=10)
    ax4.legend(fontsize=9.5)
    ax4.grid(axis="y", linestyle="--", alpha=0.45)

    plt.suptitle("Tổng hợp so sánh toàn diện Mô hình 1 (Custom CNN) và Mô hình 2 (MobileNetV2 Transfer Learning)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(FIGURES_DIR, "fig17_comprehensive_model_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Quy trình 5] Đã lưu biểu đồ tổng hợp so sánh 2 mô hình: {out_path}")


def run_quy_trinh_5():
    X_train, y_train, X_val, y_val, X_test, y_test, df = load_split_arrays()

    hist_json = os.path.join(RESULTS_DIR, "quy_trinh_4_training_history.json")
    with open(hist_json, "r", encoding="utf-8") as f:
        train_hist = json.load(f)

    cnn_path = os.path.join(MODELS_DIR, "custom_cnn_best.keras")
    mnet_path = os.path.join(MODELS_DIR, "mobilenetv2_best.keras")

    print("[Quy trình 5] Đang tải trọng số tốt nhất của 2 mô hình để đánh giá trên tập Test (550 ảnh)...")
    cnn_model = tf.keras.models.load_model(cnn_path)
    mnet_model = tf.keras.models.load_model(mnet_path)

    cnn_eval = evaluate_single_model(cnn_model, X_test, y_test, "Mô hình 1: Custom CNN")
    mnet_eval = evaluate_single_model(mnet_model, X_test, y_test, "Mô hình 2: MobileNetV2 (Transfer Learning)")

    plot_confusion_matrices(cnn_eval, mnet_eval)
    plot_per_class_metrics(cnn_eval, mnet_eval)
    plot_roc_curves(cnn_eval, mnet_eval)
    plot_misclassified_analysis(X_test, y_test, cnn_eval, mnet_eval)
    demo_results = plot_prediction_demo_grid(df, cnn_model, mnet_model)
    plot_overall_model_comparison(cnn_eval, mnet_eval, train_hist)

    # Xóa mảng numpy tạm thời trước khi ghi ra JSON
    for ev in [cnn_eval, mnet_eval]:
        for k in ["y_prob", "y_pred", "fpr_dict", "tpr_dict"]:
            ev.pop(k, None)

    out_data = {
        "test_set_size": int(len(X_test)),
        "custom_cnn_eval": cnn_eval,
        "mobilenetv2_eval": mnet_eval,
        "demo_predictions": demo_results,
    }

    eval_json = os.path.join(RESULTS_DIR, "quy_trinh_5_evaluation.json")
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # Xuất bảng CSV so sánh từng lớp
    rows_cls = []
    for c_row, m_row in zip(cnn_eval["per_class"], mnet_eval["per_class"]):
        rows_cls.append({
            "class_name": c_row["class_name"],
            "class_vi": c_row["class_vi"],
            "support": c_row["support"],
            "cnn_precision": c_row["precision"],
            "cnn_recall": c_row["recall"],
            "cnn_f1": c_row["f1_score"],
            "cnn_auc": c_row["roc_auc"],
            "mnet_precision": m_row["precision"],
            "mnet_recall": m_row["recall"],
            "mnet_f1": m_row["f1_score"],
            "mnet_auc": m_row["roc_auc"],
            "f1_gain_pct": round((m_row["f1_score"] - c_row["f1_score"]) * 100, 2),
        })
    pd.DataFrame(rows_cls).to_csv(os.path.join(RESULTS_DIR, "per_class_metrics_comparison.csv"), index=False, encoding="utf-8-sig")

    print(f"[Quy trình 5] Hoàn tất! Kết quả đánh giá đã được lưu tại: {eval_json}")
    return out_data


if __name__ == "__main__":
    run_quy_trinh_5()
