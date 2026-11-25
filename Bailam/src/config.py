# -*- coding: utf-8 -*-
"""
Cấu hình chung cho Đề tài 10:
Xây dựng và đánh giá mô hình CNN và MobileNetV2 trong bài toán nhận diện và phân loại hình ảnh các loài hoa
Bộ dữ liệu: TF Flowers (3.670 ảnh, 5 lớp)
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "flower_photos")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

CLASS_NAMES_VI = {
    "daisy": "Hoa Cúc (Daisy)",
    "dandelion": "Hoa Bồ Công Anh (Dandelion)",
    "roses": "Hoa Hồng (Roses)",
    "sunflowers": "Hoa Hướng Dương (Sunflowers)",
    "tulips": "Hoa Tulip (Tulips)",
}

CLASS_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#EDC948", "#B07AA1"]

IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
RANDOM_SEED = 42
