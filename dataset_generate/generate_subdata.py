import gc
import glob
import math
import os
import cv2
import numpy as np
import torch
import yaml
from utils.homographic_augmentation import sample_homography
def calculate_scale_from_homography(H):
    if isinstance(H, torch.Tensor):
        H = H.squeeze().cpu().numpy()
    if H.shape == (9,):
        H = H.reshape(3, 3)
    if H.shape != (3, 3):
        return None, None, None
    sx = math.sqrt(H[0, 0]**2 + H[1, 0]**2 + H[2, 0]**2)
    sy = math.sqrt(H[0, 1]**2 + H[1, 1]**2 + H[2, 1]**2)
    scale = math.sqrt(sx * sy)
    return sx, sy, scale
def check_homography_valid(H, config_params):
    if isinstance(H, torch.Tensor):
        H = H.squeeze().cpu().numpy()
    if H.shape != (3, 3):
        return False, f"单应矩阵形状错误: {H.shape}"
    import math
    sx = math.sqrt(H[0, 0]**2 + H[1, 0]**2 + H[2, 0]**2)
    sy = math.sqrt(H[0, 1]**2 + H[1, 1]**2 + H[2, 1]**2)
    scale = math.sqrt(sx * sy)
    R = H[:2, :2]
    try:
        U, s, Vt = np.linalg.svd(R)
        rotation_matrix = U @ Vt
        angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        angle_abs = abs(angle)
    except:
        angle_abs = 0
    if config_params.get('rotation', True):
        max_angle = config_params.get('max_angle', math.pi)
        if angle_abs > max_angle + 1e-6:
            return False, f"角度 {angle_abs:.4f} > max_angle {max_angle:.4f}"
    if config_params.get('scaling', True):
        stype = config_params.get('stype', 0)
        if stype == 0:
            if scale < 1.0 - 1e-6 or scale > 2.0 + 1e-6:
                return False, f"缩放 {scale:.4f} 不在 [1, 2] 范围内 (stype=0)"
        elif stype == 1:
            if scale < 2.0 - 1e-6 or scale > 4.0 + 1e-6:
                return False, f"缩放 {scale:.4f} 不在 [2, 4] 范围内 (stype=1)"
        elif stype == 3:
            if scale < 1.0 - 1e-6 or scale > 4.0 + 1e-6:
                return False, f"缩放 {scale:.4f} 不在 [1, 4] 范围内 (stype=3)"
        elif stype == 4:
            in_small_range = (scale >= 1.0/6.0 - 1e-6 and scale <= 1.0/4.0 + 1e-6)
            in_large_range = (scale >= 4.0 - 1e-6 and scale <= 6.0 + 1e-6)
            if not (in_small_range or in_large_range):
                return False, f"缩放 {scale:.4f} 不在 [1/6, 1/4]=[0.167, 0.25] 或 [4, 6] 范围内 (stype=4)"
    if sx > 0 and sy > 0:
        aniso = max(sx, sy) / min(sx, sy)
        if aniso > 2.5:
            return False, f"各向异性 sx/sy={sx:.4f}/{sy:.4f} 比值 {aniso:.4f} > 2.5"
    return True, None
def save_vis_image(opt_img, sar_img, save_path, vis_size):
    if len(opt_img.shape) == 2:
        opt_img_vis = cv2.cvtColor(opt_img, cv2.COLOR_GRAY2BGR)
    elif len(opt_img.shape) == 3 and opt_img.shape[2] == 4:
        opt_img_vis = cv2.cvtColor(opt_img, cv2.COLOR_BGRA2BGR)
    else:
        opt_img_vis = opt_img.copy()
    if len(sar_img.shape) == 2:
        sar_img_vis = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    elif len(sar_img.shape) == 3 and sar_img.shape[2] == 4:
        sar_img_vis = cv2.cvtColor(sar_img, cv2.COLOR_BGRA2BGR)
    else:
        sar_img_vis = sar_img.copy()
    h, w = opt_img_vis.shape[:2]
    if sar_img_vis.shape[:2] != (h, w):
        sar_img_vis = cv2.resize(sar_img_vis, (w, h))
    vis_img = np.concatenate([opt_img_vis, sar_img_vis], axis=1)
    vis_img_resized = cv2.resize(vis_img, (vis_size[1] * 2, vis_size[0]))
    cv2.imwrite(save_path, vis_img_resized)
def crop_with_padding(img, top, bottom, left, right, offset_y, offset_x):
    src_top = int(top - offset_y)
    src_bottom = int(bottom - offset_y)
    src_left = int(left - offset_x)
    src_right = int(right - offset_x)
    out_h = int(bottom - top)
    out_w = int(right - left)
    if len(img.shape) == 3:
        out = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((out_h, out_w), dtype=img.dtype)
    img_h, img_w = img.shape[:2]
    valid_top = max(0, src_top)
    valid_bottom = min(img_h, src_bottom)
    valid_left = max(0, src_left)
    valid_right = min(img_w, src_right)
    if valid_top < valid_bottom and valid_left < valid_right:
        dst_top = valid_top - src_top
        dst_bottom = dst_top + (valid_bottom - valid_top)
        dst_left = valid_left - src_left
        dst_right = dst_left + (valid_right - valid_left)
        out[dst_top:dst_bottom, dst_left:dst_right] = img[valid_top:valid_bottom, valid_left:valid_right]
    return out
def generate_subdata(psimg, ptimg, num_sample=5, cropnum=4, ssave_path=None, tsave_path=None, stype=None):
    gt_homo = []
    saved_opt_paths = []
    simg = cv2.imread(psimg, -1)
    timg = cv2.imread(ptimg, -1)
    if simg is None or timg is None:
        return gt_homo, saved_opt_paths
    Hs, Ws = simg.shape[:2]
    Ht, Wt = timg.shape[:2]
    H = max(Hs, Ht)
    W = max(Ws, Wt)
    maskp = glob.glob(os.path.join(os.path.dirname(psimg), "vmask*.png"))
    if len(maskp) != 0 and os.path.exists(maskp[0]):
        vmask = cv2.imread(maskp[0], 0) / 255.0
        if vmask.shape[:2] != (Hs, Ws):
            vmask = cv2.resize(vmask, (Ws, Hs))
    else:
        vmask = np.ones((Hs, Ws), dtype=np.float32)
    if len(simg.shape) == 3:
        img_valid = (~np.all(simg[:, :, :3] <= 5, axis=-1)).astype(np.float32)
    else:
        img_valid = (simg > 5).astype(np.float32)
    vmask = np.minimum(vmask.astype(np.float32), img_valid)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transparam_path = os.path.join(current_dir, "gen_data.yaml")
    with open(transparam_path, "r", encoding='utf-8') as fc:
        config = yaml.safe_load(fc)
    subsize = tuple(config['data']['preprocessing']['resize'])
    tratio = config['data']['preprocessing']['tratio']
    resize = (2*H, 2*W)
    params = config['data']['augmentation']['homographic']['params']
    if stype is not None:
        params['stype'] = stype
    H_folder = None
    if ssave_path is not None:
        base_dir = os.path.dirname(os.path.dirname(ssave_path))
        H_folder = os.path.join(base_dir, 'H')
        if not os.path.exists(H_folder):
            os.makedirs(H_folder)
    if min(H, W) / min(subsize) < 2.0:
        if params.get('rotation', False):
            params['max_angle'] = 0.5
        if params.get('scaling', False):
            params['scaling'] = False
        params['patch_ratio'] = 0.8
        if params.get('translation', False):
            tratio = 0.2
        else:
            tratio = 0.0
    subh, subw = subsize
    T_offset_s = np.array([[1, 0, W//2], [0, 1, H//2], [0, 0, 1]], dtype=np.float64)
    T_offset_t = np.array([[1, 0, W//2], [0, 1, H//2], [0, 0, 1]], dtype=np.float64)
    def check_image_valid(img, max_ratio=0.02):
        if img is None: return False
        if len(img.shape) == 3:
            img_check = img[:, :, :3] if img.shape[2] == 4 else img
            black_pixels = np.sum(np.all(img_check <= 5, axis=-1))
            white_pixels = np.sum(np.all(img_check >= 250, axis=-1))
        else:
            black_pixels = np.sum(img <= 5)
            white_pixels = np.sum(img >= 250)
        total = img.shape[0] * img.shape[1]
        if (black_pixels / total) > max_ratio or (white_pixels / total) > max_ratio:
            return False
        return True
    def is_scrop_valid(cx, cy):
        v_left = int(cx - subw//2 - W//2)
        v_right = int(cx + subw//2 - W//2)
        v_top = int(cy - subh//2 - H//2)
        v_bottom = int(cy + subh//2 - H//2)
        if v_left < 0 or v_right > Ws or v_top < 0 or v_bottom > Hs:
            return False
        return np.mean(vmask[v_top:v_bottom, v_left:v_right]) >= 0.90
    min_x = W//2 + subw//2
    max_x = W//2 + max(Ws, subw) - subw//2
    if min_x > max_x: min_x = max_x
    min_y = H//2 + subh//2
    max_y = H//2 + max(Hs, subh) - subh//2
    if min_y > max_y: min_y = max_y
    vmask_f32 = np.ascontiguousarray(vmask, dtype=np.float32)
    for i in range(num_sample):
        retry_count = 0
        max_retries = 20
        scenters_found = []
        tcenters_found = []
        subhomo_found = []
        while len(scenters_found) < cropnum and retry_count < max_retries:
            retry_count += 1
            H0 = sample_homography(resize, params)
            H0 = H0 / H0[:, -1, -1]
            is_valid, error_msg = check_homography_valid(H0, params)
            if not is_valid:
                continue
            H0_np = H0.squeeze(dim=0).numpy().astype(np.float64)
            scenters_temp = []
            tcenters_temp = []
            subhomo_temp = []
            for _ in range(500):
                if len(scenters_temp) == cropnum: break
                cx = np.random.randint(min_x, max_x + 1)
                cy = np.random.randint(min_y, max_y + 1)
                if not is_scrop_valid(cx, cy):
                    continue
                pt = H0_np @ np.array([cx, cy, 1.0])
                tcx, tcy = pt[0]/pt[2], pt[1]/pt[2]
                found_shift = False
                n_shifts = 1 if tratio == 0 else 100
                for _ in range(n_shifts):
                    dx = 0 if tratio == 0 else np.random.uniform(-subw*tratio, subw*tratio)
                    dy = 0 if tratio == 0 else np.random.uniform(-subh*tratio, subh*tratio)
                    ntcx, ntcy = int(tcx + dx), int(tcy + dy)
                    tl, tr = ntcx - subw//2, ntcx + subw//2
                    tu, td = ntcy - subh//2, ntcy + subh//2
                    if tl < 0 or tr > 2*W or tu < 0 or td > 2*H:
                        continue
                    T_dst_inv = np.array([[1, 0, -tl], [0, 1, -tu], [0, 0, 1]], dtype=np.float64)
                    M_local_s = T_dst_inv @ H0_np @ T_offset_s
                    warped_vmask = cv2.warpPerspective(vmask_f32, M_local_s, (subw, subh), flags=cv2.INTER_NEAREST)
                    if np.mean(warped_vmask) >= 0.99:
                        found_shift = True
                        break
                if found_shift:
                    sl, sr = cx - subw//2, cx + subw//2
                    su, sd = cy - subh//2, cy + subh//2
                    simg_ij = crop_with_padding(simg, su, sd, sl, sr, H//2, W//2)
                    M_local_t = T_dst_inv @ H0_np @ T_offset_t
                    timg_ij = cv2.warpPerspective(timg, M_local_t, (subw, subh), flags=cv2.INTER_LINEAR)
                    if not check_image_valid(simg_ij):
                        found_shift = False
                if found_shift:
                    scenters_temp.append((cx, cy))
                    tcenters_temp.append((ntcx, ntcy))
                    T_src = np.array([[1, 0, cx - subw//2], [0, 1, cy - subh//2], [0, 0, 1]], dtype=np.float64)
                    subhomo = T_dst_inv @ H0_np @ T_src
                    subhomo_temp.append(subhomo)
            if len(scenters_temp) == cropnum:
                scenters_found = scenters_temp
                tcenters_found = tcenters_temp
                subhomo_found = subhomo_temp
                sx_h0, sy_h0, scale_h0 = calculate_scale_from_homography(H0_np)
                if scale_h0 is not None:
                    print(f"  Base homography H0 [{i:02d}]: sx={sx_h0:.6f}, sy={sy_h0:.6f}, scale={scale_h0:.6f}")
                break
        if len(scenters_found) < cropnum:
            print(f"  [ERROR] Could not generate a valid homography after {retry_count} attempts; stopping this image pair")
            gc.collect()
            return gt_homo, saved_opt_paths
        for j in range(cropnum):
            cx, cy = scenters_found[j]
            ntcx, ntcy = tcenters_found[j]
            sl, sr = cx - subw//2, cx + subw//2
            su, sd = cy - subh//2, cy + subh//2
            simg_ij = crop_with_padding(simg, su, sd, sl, sr, H//2, W//2)
            tl, tr = ntcx - subw//2, ntcx + subw//2
            tu, td = ntcy - subh//2, ntcy + subh//2
            T_dst_inv = np.array([[1, 0, -tl], [0, 1, -tu], [0, 0, 1]], dtype=np.float64)
            M_local_t = T_dst_inv @ H0_np @ T_offset_t
            timg_ij = cv2.warpPerspective(timg, M_local_t, (subw, subh), flags=cv2.INTER_LINEAR)
            homo_ij = subhomo_found[j].flatten()
            homo_ij = homo_ij / homo_ij[-1]
            if ssave_path is not None:
                dir_s, name_s = os.path.split(ssave_path)
                base_s, ext_s = os.path.splitext(name_s)
                dir_t, name_t = os.path.split(tsave_path)
                base_t, ext_t = os.path.splitext(name_t)
                suffix = f"_0{i:02d}{j:02d}"
                simg_sp = os.path.join(dir_s, f"{base_s}{suffix}{ext_s}")
                timg_sp = os.path.join(dir_t, f"{base_t}{suffix}{ext_t}")
                simg_save = simg_ij
                if len(simg_save.shape) == 3 and simg_save.shape[2] == 4:
                    simg_save = cv2.cvtColor(simg_save, cv2.COLOR_BGRA2BGR)
                timg_save = timg_ij
                if len(timg_save.shape) == 3 and timg_save.shape[2] == 4:
                    timg_save = cv2.cvtColor(timg_save, cv2.COLOR_BGRA2BGR)
                ok_s = cv2.imwrite(simg_sp, simg_save)
                ok_t = cv2.imwrite(timg_sp, timg_save)
                if not ok_s or not ok_t:
                    print(f"  [WARNING] Failed to write images [{i:02d}-{j:02d}]; skipping this homography")
                    for p in [simg_sp, timg_sp]:
                        if os.path.exists(p):
                            os.remove(p)
                    continue
                if not os.path.exists(simg_sp) or not os.path.exists(timg_sp):
                    print(f"  [WARNING] Output files are missing after write [{i:02d}-{j:02d}]; skipping this homography")
                    for p in [simg_sp, timg_sp]:
                        if os.path.exists(p):
                            os.remove(p)
                    continue
                h_sp = os.path.join(H_folder, f"{base_s}{suffix}.txt")
                try:
                    with open(h_sp, 'w') as hf:
                        hf.write(' '.join(map(str, homo_ij)) + '\n')
                except Exception as e:
                    print(f"  [WARNING] Failed to write homography file [{i:02d}-{j:02d}]: {e}; skipping")
                    for p in [simg_sp, timg_sp, h_sp]:
                        if os.path.exists(p):
                            os.remove(p)
                    continue
            else:
                simg_sp = None
            sx, sy, scale = calculate_scale_from_homography(homo_ij)
            if scale is not None:
                print(f"  Homography [{i:02d}-{j:02d}]: sx={sx:.6f}, sy={sy:.6f}, scale={scale:.6f}")
            gt_homo.append(homo_ij)
            saved_opt_paths.append(simg_sp)
        gc.collect()
    for i in range(num_sample):
        retry_count = 0
        max_retries = 20
        tcenters_found = []
        scenters_found = []
        subhomo_found = []
        while len(tcenters_found) < cropnum and retry_count < max_retries:
            retry_count += 1
            H1 = sample_homography(resize, params)
            H1 = H1 / H1[:, -1, -1]
            is_valid, error_msg = check_homography_valid(H1, params)
            if not is_valid:
                continue
            H1_np = H1.squeeze(dim=0).numpy().astype(np.float64)
            tcenters_temp = []
            scenters_temp = []
            subhomo_temp = []
            for _ in range(500):
                if len(tcenters_temp) == cropnum: break
                tcx = np.random.randint(min_x, max_x + 1)
                tcy = np.random.randint(min_y, max_y + 1)
                if not is_scrop_valid(tcx, tcy):
                    continue
                pt = H1_np @ np.array([tcx, tcy, 1.0])
                scx, scy = pt[0]/pt[2], pt[1]/pt[2]
                found_shift = False
                n_shifts = 1 if tratio == 0 else 100
                for _ in range(n_shifts):
                    dx = 0 if tratio == 0 else np.random.uniform(-subw*tratio, subw*tratio)
                    dy = 0 if tratio == 0 else np.random.uniform(-subh*tratio, subh*tratio)
                    nscx, nscy = int(scx + dx), int(scy + dy)
                    sl, sr = nscx - subw//2, nscx + subw//2
                    su, sd = nscy - subh//2, nscy + subh//2
                    if sl < 0 or sr > 2*W or su < 0 or sd > 2*H:
                        continue
                    T_dst_inv = np.array([[1, 0, -sl], [0, 1, -su], [0, 0, 1]], dtype=np.float64)
                    M_local_s = T_dst_inv @ H1_np @ T_offset_s
                    warped_vmask = cv2.warpPerspective(vmask_f32, M_local_s, (subw, subh), flags=cv2.INTER_NEAREST)
                    if np.mean(warped_vmask) >= 0.90:
                        found_shift = True
                        break
                if found_shift:
                    tl, tr = tcx - subw//2, tcx + subw//2
                    tu, td = tcy - subh//2, tcy + subh//2
                    timg_ij = crop_with_padding(timg, tu, td, tl, tr, H//2, W//2)
                    M_local_s = T_dst_inv @ H1_np @ T_offset_s
                    simg_ij = cv2.warpPerspective(simg, M_local_s, (subw, subh), flags=cv2.INTER_LINEAR)
                    if not check_image_valid(simg_ij):
                        found_shift = False
                if found_shift:
                    tcenters_temp.append((tcx, tcy))
                    scenters_temp.append((nscx, nscy))
                    T_src = np.array([[1, 0, tcx - subw//2], [0, 1, tcy - subh//2], [0, 0, 1]], dtype=np.float64)
                    subhomo = T_dst_inv @ H1_np @ T_src
                    subhomo_inv = np.linalg.inv(subhomo)
                    subhomo_temp.append(subhomo_inv)
            if len(tcenters_temp) == cropnum:
                tcenters_found = tcenters_temp
                scenters_found = scenters_temp
                subhomo_found = subhomo_temp
                sx_h1, sy_h1, scale_h1 = calculate_scale_from_homography(H1_np)
                if scale_h1 is not None:
                    print(f"  Base homography H1 [{i:02d}]: sx={sx_h1:.6f}, sy={sy_h1:.6f}, scale={scale_h1:.6f}")
                break
        if len(tcenters_found) < cropnum:
            print(f"  [ERROR] Could not generate a valid homography after {retry_count} attempts; stopping this image pair")
            gc.collect()
            return gt_homo, saved_opt_paths
        for j in range(cropnum):
            tcx, tcy = tcenters_found[j]
            nscx, nscy = scenters_found[j]
            tl, tr = tcx - subw//2, tcx + subw//2
            tu, td = tcy - subh//2, tcy + subh//2
            timg_ij = crop_with_padding(timg, tu, td, tl, tr, H//2, W//2)
            sl, sr = nscx - subw//2, nscx + subw//2
            su, sd = nscy - subh//2, nscy + subh//2
            T_dst_inv = np.array([[1, 0, -sl], [0, 1, -su], [0, 0, 1]], dtype=np.float64)
            M_local_simg = T_dst_inv @ H1_np @ T_offset_s
            simg_ij = cv2.warpPerspective(simg, M_local_simg, (subw, subh), flags=cv2.INTER_LINEAR)
            homo_ij = subhomo_found[j].flatten()
            homo_ij = homo_ij / homo_ij[-1]
            if ssave_path is not None:
                dir_s, name_s = os.path.split(ssave_path)
                base_s, ext_s = os.path.splitext(name_s)
                dir_t, name_t = os.path.split(tsave_path)
                base_t, ext_t = os.path.splitext(name_t)
                suffix = f"_1{i:02d}{j:02d}"
                simg_sp = os.path.join(dir_s, f"{base_s}{suffix}{ext_s}")
                timg_sp = os.path.join(dir_t, f"{base_t}{suffix}{ext_t}")
                simg_save = simg_ij
                if len(simg_save.shape) == 3 and simg_save.shape[2] == 4:
                    simg_save = cv2.cvtColor(simg_save, cv2.COLOR_BGRA2BGR)
                timg_save = timg_ij
                if len(timg_save.shape) == 3 and timg_save.shape[2] == 4:
                    timg_save = cv2.cvtColor(timg_save, cv2.COLOR_BGRA2BGR)
                ok_s = cv2.imwrite(simg_sp, simg_save)
                ok_t = cv2.imwrite(timg_sp, timg_save)
                if not ok_s or not ok_t:
                    print(f"  [WARNING] Failed to write images [{i:02d}-{j:02d}]; skipping this homography")
                    for p in [simg_sp, timg_sp]:
                        if os.path.exists(p):
                            os.remove(p)
                    continue
                if not os.path.exists(simg_sp) or not os.path.exists(timg_sp):
                    print(f"  [WARNING] Output files are missing after write [{i:02d}-{j:02d}]; skipping this homography")
                    for p in [simg_sp, timg_sp]:
                        if os.path.exists(p):
                            os.remove(p)
                    continue
                h_sp = os.path.join(H_folder, f"{base_s}{suffix}.txt")
                try:
                    with open(h_sp, 'w') as hf:
                        hf.write(' '.join(map(str, homo_ij)) + '\n')
                except Exception as e:
                    print(f"  [WARNING] Failed to write homography file [{i:02d}-{j:02d}]: {e}; skipping")
                    for p in [simg_sp, timg_sp, h_sp]:
                        if os.path.exists(p):
                            os.remove(p)
                    continue
            else:
                simg_sp = None
            sx, sy, scale = calculate_scale_from_homography(homo_ij)
            if scale is not None:
                print(f"  Homography [{i:02d}-{j:02d}]: sx={sx:.6f}, sy={sy:.6f}, scale={scale:.6f}")
            gt_homo.append(homo_ij)
            saved_opt_paths.append(simg_sp)
        gc.collect()
    return gt_homo, saved_opt_paths
