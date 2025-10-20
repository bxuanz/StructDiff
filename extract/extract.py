import sys
import cv2
import numpy as np
from PIL import Image
import os
import torch

# 检查 controlnet_aux 是否已安装
try:
    from controlnet_aux import (
        HEDdetector, MLSDdetector, MidasDetector, LineartDetector,
        NormalBaeDetector, PidiNetDetector, ContentShuffleDetector
    )
except ImportError as e:
    print("错误：无法导入 controlnet_aux 的某个模块。")
    print(f"具体错误: {e}")
    print("请确保您已在新环境中运行 'pip install -U controlnet-aux opencv-python-headless'")
    sys.exit(1)

def generate_readme(output_dir, file_descriptions):
    """
    生成一个 readme.txt 文件来描述生成的控制图像。

    参数:
    - output_dir (str): 输出目录。
    - file_descriptions (dict): 包含文件名后缀和其描述的字典。
    """
    readme_path = os.path.join(output_dir, "readme.txt")
    try:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("生成的控制图像说明\n")
            f.write("=" * 30 + "\n\n")
            for key, description in file_descriptions.items():
                f.write(f"文件名后缀: *_{key}.png\n")
                f.write(f"类型: {description['name']}\n")
                f.write(f"简介: {description['desc']}\n\n")
        print(f"\n[信息] 描述文件 'readme.txt' 已成功生成于 '{output_dir}' 目录。")
    except Exception as e:
        print(f"[错误] 无法生成描述文件: {e}")


def extract_remote_sensing_controls(input_image_path, output_dir="control_images_rs"):
    """
    对一张遥感图像提取多种不同类型的控制图像。

    参数:
    - input_image_path (str): 输入图片的路径。
    - output_dir (str): 保存输出图像的目录。
    """
    print("-" * 50)
    print(f"脚本正在使用的Python解释器是: {sys.executable}")
    print(f"开始处理遥感图像: {input_image_path}")

    # 检查输入图片是否存在
    if not os.path.exists(input_image_path):
        print(f"[错误] 输入图片 '{input_image_path}' 不存在。")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有生成的控制图将保存在 '{output_dir}' 目录下。")

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 读取输入图片
    input_image_pil = Image.open(input_image_path).convert("RGB")
    input_image_np = np.array(input_image_pil)
    
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    generated_files = {} # 用于存储生成的文件信息以创建readme

    def flush_gpu():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 1. Canny 边缘检测 (OpenCV) ---
    print("\n1/8: 正在提取 Canny 边缘图...")
    try:
        canny_image_np = cv2.Canny(input_image_np, 100, 200)
        canny_image_pil = Image.fromarray(canny_image_np, mode='L').convert("RGB")
        canny_image_pil.save(os.path.join(output_dir, f"{base_name}_canny.png"))
        generated_files['canny'] = {
            "name": "Canny 边缘图",
            "desc": "一种经典的边缘检测算法，能够快速提取图像中物体的轮廓线，适用于识别道路、河流边界、建筑轮廓等。"
        }
        print("  - Canny 图已保存。")
    except Exception as e:
        print(f"  - Canny 处理失败: {e}")

    # --- 2. HED 边缘检测 ---
    print("\n2/8: 正在提取 HED 边缘图...")
    try:
        hed_processor = HEDdetector.from_pretrained('lllyasviel/Annotators').to(device)
        hed_image_pil = hed_processor(input_image_pil)
        hed_image_pil.save(os.path.join(output_dir, f"{base_name}_hed.png"))
        generated_files['hed'] = {
            "name": "HED 边缘图 (Holistically-Nested Edge Detection)",
            "desc": "基于深度学习的边缘检测，能提取出更丰富、更符合人类感知的多层次边缘，对于复杂的地面纹理和地物边界有更好的效果。"
        }
        del hed_processor
        flush_gpu()
        print("  - HED 图已保存。")
    except Exception as e:
        print(f"  - HED 处理失败: {e}")

    # --- 3. MLSD 直线段检测 ---
    print("\n3/8: 正在提取 MLSD 直线段图...")
    try:
        mlsd_processor = MLSDdetector.from_pretrained('lllyasviel/Annotators').to(device)
        mlsd_image_pil = mlsd_processor(input_image_pil)
        mlsd_image_pil.save(os.path.join(output_dir, f"{base_name}_mlsd.png"))
        generated_files['mlsd'] = {
            "name": "MLSD 直线图 (Mobile Line Segment Detection)",
            "desc": "专门用于检测图像中的直线段。在遥感图像中，非常适合提取规整的建筑轮廓、道路、农田边界等人工地物的直线特征。"
        }
        del mlsd_processor
        flush_gpu()
        print("  - MLSD 图已保存。")
    except Exception as e:
        print(f"  - MLSD 处理失败: {e}")

    # --- 4. Depth Map (深度图) ---
    print("\n4/8: 正在提取深度图 (Midas)...")
    try:
        midas_processor = MidasDetector.from_pretrained('lllyasviel/Annotators').to(device)
        depth_image_pil = midas_processor(input_image_pil)
        depth_image_pil.save(os.path.join(output_dir, f"{base_name}_depth.png"))
        generated_files['depth'] = {
            "name": "深度图 (Midas)",
            "desc": "估算图像中各点到视角的距离。在遥感图像中，它可以转化为一种伪高程信息，帮助模型理解地形的起伏、建筑物的高度关系等空间结构。"
        }
        del midas_processor
        flush_gpu()
        print("  - 深度图已保存。")
    except Exception as e:
        print(f"  - 深度图处理失败: {e}")

    # --- 5. Normal Map (法线图) ---
    print("\n5/8: 正在提取法线图 (NormalBae)...")
    try:
        normal_processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators').to(device)
        normal_image_pil = normal_processor(input_image_pil)
        normal_image_pil.save(os.path.join(output_dir, f"{base_name}_normal.png"))
        generated_files['normal'] = {
            "name": "法线图 (NormalBae)",
            "desc": "表示物体表面的朝向信息。它可以详细地刻画地物的表面纹理和细节，例如山脉的褶皱、建筑物的立面细节等，对光影和质感有强烈的控制作用。"
        }
        del normal_processor
        flush_gpu()
        print("  - 法线图已保存。")
    except Exception as e:
        print(f"  - 法线图处理失败: {e}")

    # --- 6. Lineart (线稿) ---
    print("\n6/8: 正在提取线稿图 (Lineart)...")
    try:
        lineart_processor = LineartDetector.from_pretrained('lllyasviel/Annotators').to(device)
        lineart_image_pil = lineart_processor(input_image_pil)
        lineart_image_pil.save(os.path.join(output_dir, f"{base_name}_lineart.png"))
        generated_files['lineart'] = {
            "name": "艺术线稿图 (Lineart)",
            "desc": "生成一种干净、精细的线稿图，能捕捉到地物的主要轮廓和内部结构线条，风格上比Canny更偏向艺术性和概括性。"
        }
        del lineart_processor
        flush_gpu()
        print("  - 线稿图已保存。")
    except Exception as e:
        print(f"  - 线稿图处理失败: {e}")

    # --- 7. Scribble (手绘风线稿) ---
    print("\n7/8: 正在提取手绘风线稿 (PidiNet)...")
    try:
        pidi_processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators').to(device)
        pidi_image_pil = pidi_processor(input_image_pil, safe=True) # 使用 safe 模式
        pidi_image_pil.save(os.path.join(output_dir, f"{base_name}_scribble_pidi.png"))
        generated_files['scribble_pidi'] = {
            "name": "手绘涂鸦图 (Scribble/PidiNet)",
            "desc": "提取一种类似手绘或涂鸦风格的线条，线条更粗犷、不规则。这种控制图可以引导生成具有手绘感或者更加抽象风格的遥感图像。"
        }
        del pidi_processor
        flush_gpu()
        print("  - 手绘风线稿已保存。")
    except Exception as e:
        print(f"  - 手绘风线稿处理失败: {e}")
        
    # --- 8. Content Shuffle (内容随机重组) ---
    print("\n8/8: 正在提取内容重组图 (Shuffle)...")
    try:
        shuffle_processor = ContentShuffleDetector() # 通常在CPU上运行
        shuffle_image_pil = shuffle_processor(input_image_pil)
        shuffle_image_pil.save(os.path.join(output_dir, f"{base_name}_shuffle.png"))
        generated_files['shuffle'] = {
            "name": "内容重组图 (Content Shuffle)",
            "desc": "将图像分割成小块并随机打乱重组。这个控制条件保留了原始图像的色彩和基本纹理，但破坏了其宏观结构，常用于风格迁移或生成抽象纹理的场景。"
        }
        del shuffle_processor
        flush_gpu()
        print("  - 内容重组图已保存。")
    except Exception as e:
        print(f"  - 内容重组图处理失败: {e}")

    print("\n" + "=" * 50)
    print(f"所有 {len(generated_files)} 种适用于遥感图像的控制图提取完成！")
    print("=" * 50)

    # --- 生成描述文件 ---
    generate_readme(output_dir, generated_files)




if __name__ == '__main__':
    # --- 使用说明 ---
    # 1. 确保已在新环境中安装了所有依赖库。
    # 2. 将你的图片路径替换下面的 "input_file"。
    
    input_file = "/data01/zhaobingxuan/training_free/RichControl/extract/23372.jpg"
    output_folder = "stru_23372"
    
    extract_remote_sensing_controls(input_image_path=input_file, output_dir=output_folder)