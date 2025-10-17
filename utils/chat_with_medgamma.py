#!/usr/bin/env python3
import sys
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np
import os
import SimpleITK as sitk
import pydicom

# 引入医学影像库
try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    import pydicom
except ImportError:
    pydicom = None

def load_dicom_series(folder_path):
    """
    使用 SimpleITK 自动读取一个 DICOM 序列文件夹（CT、MRI 等）
    返回中间层切片（PIL.Image），用于可视化
    """
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(folder_path)

    if not series_IDs:
        raise ValueError(f"❌ 文件夹中未检测到有效的 DICOM 序列: {folder_path}")

    # 取第一个序列（通常一个病人/一次扫描只有一个序列）
    series_file_names = reader.GetGDCMSeriesFileNames(folder_path, series_IDs[0])
    reader.SetFileNames(series_file_names)

    # 读取影像序列
    image = reader.Execute()

    # 转为 numpy 数组，形状：[slices, height, width]
    array = sitk.GetArrayFromImage(image)

    # 选取中间一层用于显示
    mid_slice = array[array.shape[0] // 2]

    # 归一化到 0~255 灰度范围
    img = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice)) * 255.0
    img = img.astype(np.uint8)

    # 转为 PIL 图像
    return Image.fromarray(img).convert("L")


def load_medical_image(path):
    """加载各种医学图像格式（包括文件夹形式的 DICOM 序列）"""
    if os.path.isdir(path):
        # 文件夹模式 —— 尝试作为 DICOM 序列加载
        return load_dicom_series(path)

    ext = os.path.splitext(path)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return Image.open(path).convert("RGB")

    elif ext == ".dcm":
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        return Image.fromarray(img).convert("L")

    elif ext in [".nii", ".gz"]:
        nii = nib.load(path)
        data = nii.get_fdata()
        slice_idx = data.shape[-1] // 2
        slice_data = data[..., slice_idx]
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255.0
        slice_data = slice_data.astype(np.uint8)
        return Image.fromarray(slice_data).convert("L")

    else:
        raise ValueError(f"Unsupported image format or path: {path}")


def main():
    if len(sys.argv) == 2:
        question = sys.argv[1]
        image = None
    else:
        question = sys.argv[1]
        image_path = sys.argv[2]
        image = load_medical_image(image_path)

    model_path = "/data/huggingface_model"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    if image is None:

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist. Provide detailed and accurate answers to questions about medical images."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist. Provide detailed and accurate answers to questions about medical images."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image}
                ]
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        generation = generation[0][input_len:]

    answer = processor.decode(generation, skip_special_tokens=True)

    print(f"\nquestion: {question}")
    if image is not None:
        print(f"image: {image_path}")
    print(f"answer:\n{answer}\n")


if __name__ == "__main__":
    main()
