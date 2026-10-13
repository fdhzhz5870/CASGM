#!/usr/bin/env python
import os
import glob
from pathlib import Path
def parse_image_name(image_name):
    base_name = os.path.basename(image_name)
    if not base_name.endswith('.png'):
        return None, None
    name_without_ext = base_name[:-4]
    if 'pair' not in name_without_ext:
        return None, None
    try:
        parts = name_without_ext.split('pair', 1)
        if len(parts) != 2:
            return None, None
        pair_part = parts[1]
        if '_' in pair_part:
            pair_num, index = pair_part.split('_', 1)
            return f"pair{pair_num}", index
        else:
            return f"pair{pair_part}", None
    except:
        return None, None
def get_all_image_pairs(opt_folder, sar_folder):
    image_info = {}
    opt_images = glob.glob(os.path.join(opt_folder, "*.png"))
    for opt_img in opt_images:
        img_name = os.path.basename(opt_img)
        pair_num, index = parse_image_name(img_name)
        if pair_num:
            image_info[img_name] = (pair_num, index)
    return image_info
def count_images_per_pair(opt_folder):
    pair_counts = {}
    opt_images = glob.glob(os.path.join(opt_folder, "*.png"))
    for opt_img in opt_images:
        img_name = os.path.basename(opt_img)
        pair_num, index = parse_image_name(img_name)
        if pair_num:
            if pair_num not in pair_counts:
                pair_counts[pair_num] = []
            pair_counts[pair_num].append(img_name)
    for pair_num in pair_counts:
        pair_counts[pair_num].sort()
    return pair_counts
def filter_dataset(data_folder, image_names_to_delete, backup=True):
    opt_folder = os.path.join(data_folder, "opt")
    sar_folder = os.path.join(data_folder, "sar")
    gt_file = os.path.join(data_folder, "gt_homography.txt")
    if not os.path.exists(opt_folder):
        print(f"错误: opt文件夹不存在: {opt_folder}")
        return False
    if not os.path.exists(sar_folder):
        print(f"错误: sar文件夹不存在: {sar_folder}")
        return False
    if not os.path.exists(gt_file):
        print(f"错误: gt_homography.txt不存在: {gt_file}")
        return False
    if backup:
        backup_file = gt_file + ".backup"
        if os.path.exists(gt_file):
            import shutil
            shutil.copy2(gt_file, backup_file)
            print(f"已备份gt_homography.txt到: {backup_file}")
    print("正在分析图片结构...")
    pair_counts = count_images_per_pair(opt_folder)
    print(f"找到 {len(pair_counts)} 个原始图片对")
    total_images = sum(len(images) for images in pair_counts.values())
    print(f"总共 {total_images} 张图片")
    images_to_delete = set()
    for img_name in image_names_to_delete:
        img_name = os.path.basename(img_name)
        if img_name.endswith('.png'):
            images_to_delete.add(img_name)
        else:
            images_to_delete.add(img_name + '.png')
    print(f"\n准备删除 {len(images_to_delete)} 张图片")
    all_images = []
    for pair_num in sorted(pair_counts.keys()):
        all_images.extend(pair_counts[pair_num])
    indices_to_delete = []
    for idx, img_name in enumerate(all_images):
        if img_name in images_to_delete:
            indices_to_delete.append(idx)
    print(f"将在gt_homography.txt中删除 {len(indices_to_delete)} 行")
    print("\n正在读取gt_homography.txt...")
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    if len(lines) != len(all_images):
        print(f"警告: gt_homography.txt有 {len(lines)} 行，但图片有 {len(all_images)} 张")
        print("将按照图片顺序删除对应的行")
    indices_to_delete_sorted = sorted(indices_to_delete, reverse=True)
    for idx in indices_to_delete_sorted:
        if idx < len(lines):
            del lines[idx]
    print("正在更新gt_homography.txt...")
    with open(gt_file, 'w') as f:
        f.writelines(lines)
    deleted_opt = 0
    deleted_sar = 0
    print("\n正在删除图片文件...")
    for img_name in images_to_delete:
        opt_path = os.path.join(opt_folder, img_name)
        sar_path = os.path.join(sar_folder, img_name)
        if os.path.exists(opt_path):
            os.remove(opt_path)
            deleted_opt += 1
            print(f"  已删除: {opt_path}")
        if os.path.exists(sar_path):
            os.remove(sar_path)
            deleted_sar += 1
            print(f"  已删除: {sar_path}")
    print(f"\n完成！")
    print(f"  删除opt图片: {deleted_opt} 张")
    print(f"  删除sar图片: {deleted_sar} 张")
    print(f"  删除gt_homography.txt行数: {len(indices_to_delete)} 行")
    print(f"  剩余图片: {len(all_images) - len(images_to_delete)} 张")
    print(f"  剩余gt_homography.txt行数: {len(lines)} 行")
    return True
def filter_dataset_from_file(data_folder, image_list_file, backup=True):
    if not os.path.exists(image_list_file):
        print(f"错误: 文件不存在: {image_list_file}")
        return False
    print(f"正在读取图片列表文件: {image_list_file}")
    with open(image_list_file, 'r', encoding='utf-8') as f:
        image_names = [line.strip() for line in f if line.strip()]
    print(f"从文件中读取到 {len(image_names)} 个图片名称")
    return filter_dataset(data_folder, image_names, backup)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Delete listed image pairs and their homography records."
    )
    parser.add_argument("data_folder", help="Dataset root containing opt/, sar/, and gt_homography.txt.")
    parser.add_argument("image_list", help="Text file containing one image name per line.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up gt_homography.txt.")
    args = parser.parse_args()
    filter_dataset_from_file(args.data_folder, args.image_list, backup=not args.no_backup)
