import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
def move_and_rename_png_files(source_folder, target_folder, overwrite=False):
    source_path = Path(source_folder)
    target_path = Path(target_folder)
    if not source_path.exists():
        print(f"错误：源文件夹不存在：{source_folder}")
        return 0, 0, 0
    target_path.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    skipped_count = 0
    error_count = 0
    subfolder_files = defaultdict(list)
    print(f"正在扫描源文件夹：{source_folder}")
    subfolder_count = 0
    for subfolder in source_path.iterdir():
        if subfolder.is_dir():
            subfolder_count += 1
            png_files = list(subfolder.glob("*.png"))
            png_files.extend(subfolder.glob("*.PNG"))
            png_files = list(set(png_files))
            if len(png_files) > 0:
                png_files.sort(key=lambda x: x.name)
                subfolder_files[subfolder.name] = png_files
                print(f"  子文件夹 {subfolder.name}: 找到 {len(png_files)} 个PNG文件")
                for idx, f in enumerate(png_files):
                    print(f"    [{idx+1}] {f.name}")
            else:
                print(f"  子文件夹 {subfolder.name}: 没有找到PNG文件")
    print(f"总共找到 {subfolder_count} 个子文件夹")
    def get_folder_sort_key(folder_name):
        numbers = re.findall(r'\d+', folder_name)
        if numbers:
            return (0, int(numbers[0]), folder_name)
        else:
            return (1, 0, folder_name)
    sorted_subfolders = sorted(subfolder_files.keys(), key=get_folder_sort_key)
    print(f"有效子文件夹（包含PNG文件）：{len(sorted_subfolders)} 个，共 {sum(len(files) for files in subfolder_files.values())} 个PNG文件")
    if len(sorted_subfolders) == 0:
        print("  警告：没有找到包含PNG文件的子文件夹！")
        return 0, 0, 0
    print(f"\n开始复制文件...")
    pair_index = 1
    for subfolder_name in sorted_subfolders:
        png_files = subfolder_files[subfolder_name]
        if len(png_files) == 0:
            continue
        if len(png_files) < 2:
            print(f"  警告：子文件夹 {subfolder_name} 只有 {len(png_files)} 个文件，需要至少2个文件，跳过")
            error_count += len(png_files)
            continue
        file1 = png_files[0]
        file2 = png_files[1]
        if file1.resolve() == file2.resolve():
            print(f"  错误：子文件夹 {subfolder_name} 中的两个文件指向同一个文件！")
            print(f"    文件1: {file1}")
            print(f"    文件2: {file2}")
            error_count += 2
            continue
        pair_name = f"pair{pair_index:03d}"
        target_file1 = target_path / f"{pair_name}_1.png"
        target_file2 = target_path / f"{pair_name}_2.png"
        try:
            if (target_file1.exists() or target_file2.exists()) and not overwrite:
                print(f"  跳过（已存在）：{pair_name}_1.png 或 {pair_name}_2.png")
                skipped_count += 2
                pair_index += 1
                continue
            print(f"  [{pair_index:03d}] 正在复制：{subfolder_name}")
            print(f"    源文件1: {file1} -> 目标: {target_file1.name}")
            print(f"    源文件2: {file2} -> 目标: {target_file2.name}")
            shutil.copy2(str(file1), str(target_file1))
            shutil.copy2(str(file2), str(target_file2))
            if target_file1.stat().st_size == 0 or target_file2.stat().st_size == 0:
                print(f"    警告：复制后的文件大小为0！")
            copied_count += 2
            pair_index += 1
        except Exception as e:
            print(f"  错误：复制文件失败 {subfolder_name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 2
    return copied_count, skipped_count, error_count
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect and rename paired PNG files.")
    parser.add_argument("source_folder", help="Directory containing one subdirectory per pair.")
    parser.add_argument("target_folder", help="Destination directory for renamed PNG files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files.")
    args = parser.parse_args()
    print(f"Source directory: {args.source_folder}")
    print(f"Target directory: {args.target_folder}")
    print(f"Overwrite existing files: {args.overwrite}")
    print("-" * 60)
    copied, skipped, errors = move_and_rename_png_files(
        args.source_folder,
        args.target_folder,
        overwrite=args.overwrite,
    )
    print("-" * 60)
    print(f"\n复制完成！")
    print(f"  成功复制：{copied} 个文件")
    print(f"  跳过（已存在）：{skipped} 个文件")
    print(f"  错误：{errors} 个文件")
if __name__ == "__main__":
    main()
