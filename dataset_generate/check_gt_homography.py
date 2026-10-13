import os
import numpy as np
import glob
import math
def calculate_scale_from_homography(H):
    if H.shape != (3, 3):
        return None, None, None
    sx = math.sqrt(H[0, 0]**2 + H[1, 0]**2 + H[2, 0]**2)
    sy = math.sqrt(H[0, 1]**2 + H[1, 1]**2 + H[2, 1]**2)
    scale = math.sqrt(sx * sy)
    return sx, sy, scale
def check_gt_homography_file(save_folder, dataset_name="WHU-SEN-City", stype=None):
    file_path = os.path.join(save_folder, dataset_name, 'gt_homography.txt')
    opt_folder = os.path.join(save_folder, dataset_name, 'opt')
    print("=" * 60)
    print(f"检查 gt_homography.txt 文件")
    print("=" * 60)
    print(f"文件路径: {file_path}")
    print(f"图片文件夹: {opt_folder}")
    print()
    if not os.path.exists(file_path):
        print("❌ 错误: gt_homography.txt 文件不存在！")
        return False
    if os.path.exists(opt_folder):
        img_list = glob.glob(os.path.join(opt_folder, "*.png"))
        img_list.sort()
        expected_lines = len(img_list)
        print(f"✓ 找到 {expected_lines} 张图片")
    else:
        print("⚠ 警告: opt 文件夹不存在，无法统计图片数量")
        expected_lines = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        actual_lines = len([line for line in lines if line.strip()])
        print(f"✓ gt_homography.txt 有 {actual_lines} 行（非空）")
        if expected_lines is not None:
            if actual_lines == expected_lines:
                print(f"✓ 行数匹配: {actual_lines} == {expected_lines}")
            else:
                print(f"❌ 行数不匹配: {actual_lines} != {expected_lines}")
                print(f"   差异: {abs(actual_lines - expected_lines)} 行")
    except Exception as e:
        print(f"❌ 错误: 读取文件失败: {e}")
        return False
    print("\n检查文件格式...")
    invalid_lines = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        nums = line.split()
        if len(nums) != 9:
            invalid_lines.append((i, len(nums)))
    if invalid_lines:
        print(f"❌ 发现 {len(invalid_lines)} 行格式错误（不是9个数字）:")
        for line_num, num_count in invalid_lines[:10]:
            print(f"   第 {line_num} 行: {num_count} 个数字（应该是9个）")
        if len(invalid_lines) > 10:
            print(f"   ... 还有 {len(invalid_lines) - 10} 行格式错误")
    else:
        print("✓ 所有行格式正确（每行9个数字）")
    print("\n检查数据有效性...")
    try:
        H_list = np.loadtxt(file_path)
        if len(H_list.shape) == 1:
            H_list = H_list.reshape(1, -1)
        if H_list.shape[1] != 9:
            print(f"❌ 错误: 数据形状错误: {H_list.shape}，应该是 (N, 9)")
            return False
        print(f"✓ 成功加载 {H_list.shape[0]} 个单应矩阵")
        has_nan = np.isnan(H_list).any()
        has_inf = np.isinf(H_list).any()
        if has_nan:
            nan_count = np.isnan(H_list).sum()
            print(f"❌ 发现 {nan_count} 个 NaN 值")
        else:
            print("✓ 没有 NaN 值")
        if has_inf:
            inf_count = np.isinf(H_list).sum()
            print(f"❌ 发现 {inf_count} 个 Inf 值")
        else:
            print("✓ 没有 Inf 值")
        zero_rows = np.all(H_list == 0, axis=1).sum()
        if zero_rows > 0:
            print(f"⚠ 发现 {zero_rows} 行全为0")
        else:
            print("✓ 没有全0的行")
        if stype is not None:
            print(f"\n检查缩放范围 (stype={stype})...")
            invalid_scales = []
            scale_ranges = []
            for i, H_flat in enumerate(H_list):
                H = H_flat.reshape(3, 3)
                sx, sy, scale = calculate_scale_from_homography(H)
                if scale is None:
                    invalid_scales.append((i+1, "计算失败"))
                    continue
                scale_ranges.append(scale)
                if stype == 0:
                    if scale < 0.5 - 1e-6 or scale > 2.0 + 1e-6:
                        invalid_scales.append((i+1, f"{scale:.6f}"))
                elif stype == 1:
                    in_range1 = (0.25 - 1e-6 <= scale <= 0.5 + 1e-6)
                    in_range2 = (2.0 - 1e-6 <= scale <= 4.0 + 1e-6)
                    if not (in_range1 or in_range2):
                        invalid_scales.append((i+1, f"{scale:.6f}"))
                elif stype == 3:
                    in_range1 = (0.25 - 1e-6 <= scale <= 1.0 + 1e-6)
                    in_range2 = (1.0 - 1e-6 <= scale <= 4.0 + 1e-6)
                    if not (in_range1 or in_range2):
                        invalid_scales.append((i+1, f"{scale:.6f}"))
                elif stype == 4:
                    in_range_small = (1.0/6.0 - 1e-6 <= scale <= 1.0/4.0 + 1e-6)
                    in_range_large = (4.0 - 1e-6 <= scale <= 6.0 + 1e-6)
                    if not (in_range_small or in_range_large):
                        invalid_scales.append((i+1, f"{scale:.6f}"))
            if scale_ranges:
                min_scale = min(scale_ranges)
                max_scale = max(scale_ranges)
                avg_scale = np.mean(scale_ranges)
                print(f"  缩放范围: [{min_scale:.6f}, {max_scale:.6f}]")
                print(f"  平均缩放: {avg_scale:.6f}")
            if invalid_scales:
                print(f"❌ 发现 {len(invalid_scales)} 个单应矩阵的缩放不在正确范围内:")
                for line_num, scale_val in invalid_scales[:20]:
                    print(f"   第 {line_num} 行: scale={scale_val}")
                if len(invalid_scales) > 20:
                    print(f"   ... 还有 {len(invalid_scales) - 20} 个不符合要求")
            else:
                print("✓ 所有单应矩阵的缩放都在正确范围内")
        print("\n前5行数据示例:")
        for i in range(min(5, len(H_list))):
            print(f"  第 {i+1} 行: {H_list[i]}")
        print("\n" + "=" * 60)
        if invalid_lines or (stype is not None and invalid_scales):
            print("❌ 检查未完全通过，请查看上述错误信息")
            return False
        else:
            print("✓ 所有检查通过！")
            return True
    except Exception as e:
        print(f"❌ 错误: 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate a dataset homography file.")
    parser.add_argument("save_folder", help="Root directory containing the named dataset.")
    parser.add_argument("dataset_name", help="Dataset subdirectory name.")
    parser.add_argument("--stype", type=int, default=None, help="Optional scale-range preset to validate.")
    args = parser.parse_args()
    success = check_gt_homography_file(args.save_folder, args.dataset_name, args.stype)
    raise SystemExit(0 if success else 1)
