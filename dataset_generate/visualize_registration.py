import os
import glob
import numpy as np
import cv2
def overlay_images(opt_img, sar_img, homography, alpha=0.5):
    if len(opt_img.shape) == 3:
        opt_img = cv2.cvtColor(opt_img, cv2.COLOR_BGR2GRAY)
    if len(sar_img.shape) == 3:
        sar_img = cv2.cvtColor(sar_img, cv2.COLOR_BGR2GRAY)
    opt_h, opt_w = opt_img.shape[:2]
    sar_h, sar_w = sar_img.shape[:2]
    opt_size = opt_h * opt_w
    sar_size = sar_h * sar_w
    if homography.shape == (9,):
        H = homography.reshape(3, 3)
    else:
        H = homography
    if H[2, 2] != 0:
        H = H / H[2, 2]
    if opt_size >= sar_size:
        base_img = opt_img
        warp_img = sar_img
        H_warp = np.linalg.inv(H)
        base_h, base_w = opt_h, opt_w
        warp_h, warp_w = sar_h, sar_w
    else:
        base_img = sar_img
        warp_img = opt_img
        H_warp = H
        base_h, base_w = sar_h, sar_w
        warp_h, warp_w = opt_h, opt_w
    corners = np.array([
        [0, 0, 1],
        [warp_w, 0, 1],
        [warp_w, warp_h, 1],
        [0, warp_h, 1]
    ], dtype=np.float32).T
    transformed_corners = H_warp @ corners
    transformed_corners = transformed_corners / transformed_corners[2, :]
    min_x = min(0, np.min(transformed_corners[0, :]))
    max_x = max(base_w, np.max(transformed_corners[0, :]))
    min_y = min(0, np.min(transformed_corners[1, :]))
    max_y = max(base_h, np.max(transformed_corners[1, :]))
    margin = 10
    output_w = int(np.ceil(max_x - min_x)) + 2 * margin
    output_h = int(np.ceil(max_y - min_y)) + 2 * margin
    offset_x = -min_x + margin
    offset_y = -min_y + margin
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)
    H_warp_adjusted = T @ H_warp
    output_img = np.zeros((output_h, output_w), dtype=np.uint8)
    base_x1 = int(offset_x)
    base_y1 = int(offset_y)
    base_x2 = base_x1 + base_w
    base_y2 = base_y1 + base_h
    if base_x1 >= 0 and base_y1 >= 0 and base_x2 <= output_w and base_y2 <= output_h:
        output_img[base_y1:base_y2, base_x1:base_x2] = base_img
    else:
        src_y1 = max(0, -base_y1)
        src_x1 = max(0, -base_x1)
        src_y2 = min(base_h, output_h - base_y1)
        src_x2 = min(base_w, output_w - base_x1)
        dst_y1 = max(0, base_y1)
        dst_x1 = max(0, base_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        if src_y2 > src_y1 and src_x2 > src_x1:
            output_img[dst_y1:dst_y2, dst_x1:dst_x2] = base_img[src_y1:src_y2, src_x1:src_x2]
    warp_img_transformed = cv2.warpPerspective(warp_img, H_warp_adjusted, (output_w, output_h), 
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=0)
    mask = warp_img_transformed > 0
    output_img = output_img.astype(np.float32)
    warp_img_transformed = warp_img_transformed.astype(np.float32)
    output_img[mask] = output_img[mask] * (1 - alpha) + warp_img_transformed[mask] * alpha
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    return output_img
def visualize_dataset(dataset_path, output_path, dataset_names=None, alpha=0.5, split_names=None):
    if dataset_names is None:
        dataset_names = []
        if os.path.exists(dataset_path):
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    has_split = False
                    for split in ["train", "val", "test"]:
                        split_path = os.path.join(item_path, split)
                        if os.path.exists(split_path) and os.path.exists(os.path.join(split_path, "opt")):
                            has_split = True
                            break
                    if has_split:
                        dataset_names.append(item)
    if split_names is None:
        split_names = []
        if len(dataset_names) > 0:
            first_dataset_path = os.path.join(dataset_path, dataset_names[0])
            if os.path.exists(first_dataset_path):
                for item in os.listdir(first_dataset_path):
                    item_path = os.path.join(first_dataset_path, item)
                    if os.path.isdir(item_path) and item in ["train", "val", "test"]:
                        if os.path.exists(os.path.join(item_path, "opt")):
                            split_names.append(item)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"创建输出目录: {output_path}")
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")
        dataset_dir = os.path.join(dataset_path, dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"  警告: 数据集目录不存在: {dataset_dir}")
            continue
        for split_name in split_names:
            print(f"\n处理分割: {split_name}")
            split_path = os.path.join(dataset_dir, split_name)
            split_output_path = os.path.join(output_path, dataset_name, split_name)
            if not os.path.exists(split_path):
                print(f"  警告: 分割目录不存在: {split_path}")
                continue
            opt_dir = os.path.join(split_path, "opt")
            sar_dir = os.path.join(split_path, "sar")
            H_dir = os.path.join(split_path, "H")
            if not os.path.exists(opt_dir) or not os.path.exists(sar_dir):
                print(f"  警告: opt或sar目录不存在")
                continue
            if not os.path.exists(H_dir):
                print(f"  警告: H目录不存在: {H_dir}")
                continue
            img_list = sorted(
                glob.glob(os.path.join(opt_dir, "*.png")) +
                glob.glob(os.path.join(opt_dir, "*.tif")) +
                glob.glob(os.path.join(opt_dir, "*.jpg")) +
                glob.glob(os.path.join(opt_dir, "*.jpeg")) +
                glob.glob(os.path.join(opt_dir, "*.bmp"))
            )
            if len(img_list) == 0:
                print(f"  警告: 未找到图片文件")
                continue
            if not os.path.exists(split_output_path):
                os.makedirs(split_output_path)
            success_count = 0
            for i, opt_path in enumerate(img_list):
                try:
                    img_name = os.path.basename(opt_path)
                    stem = os.path.splitext(img_name)[0]
                    sar_path = os.path.join(sar_dir, img_name)
                    h_path = os.path.join(H_dir, stem + ".txt")
                    if not os.path.exists(sar_path):
                        print(f"  警告: SAR图片不存在: {sar_path}")
                        continue
                    if not os.path.exists(h_path):
                        print(f"  警告: H txt不存在: {h_path}")
                        continue
                    opt_img = cv2.imread(opt_path, cv2.IMREAD_GRAYSCALE)
                    sar_img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
                    if opt_img is None or sar_img is None:
                        print(f"  警告: 无法读取图片: {img_name}")
                        continue
                    try:
                        H = np.loadtxt(h_path)
                    except Exception as e:
                        print(f"  警告: 无法读取H txt {h_path}: {e}")
                        continue
                    overlay_img = overlay_images(opt_img, sar_img, H, alpha=alpha)
                    if len(overlay_img.shape) == 3:
                        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
                    output_img_path = os.path.join(split_output_path, img_name)
                    cv2.imwrite(output_img_path, overlay_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    success_count += 1
                    if (i + 1) % 100 == 0:
                        print(f"  已处理: {i + 1}/{len(img_list)}")
                except Exception as e:
                    print(f"  错误: 处理图片 {img_name} 时出错: {e}")
                    continue
            print(f"  完成: 成功处理 {success_count}/{len(img_list)} 对图片")
            print(f"  输出目录: {split_output_path}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize registration results.")
    parser.add_argument("dataset_path", help="Dataset root to visualize.")
    parser.add_argument("output_path", help="Directory for visualization images.")
    parser.add_argument("--dataset", action="append", dest="dataset_names", help="Dataset name; repeat as needed.")
    parser.add_argument("--split", action="append", dest="split_names", help="Split name; repeat as needed.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity from 0 to 1.")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    visualize_dataset(args.dataset_path, args.output_path, args.dataset_names, args.alpha, args.split_names)
