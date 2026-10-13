import gc
import os
import glob
from generate_subdata import generate_subdata
def _cleanup_and_count(save_folder, foldername, expected_per_city):
    total_valid = 0
    for fname in foldername:
        existing_opt = sorted(glob.glob(os.path.join(save_folder, 'opt', fname + '_*.png')))
        count = len(existing_opt)
        if count == 0:
            continue
        valid_opt = []
        for opt_p in existing_opt:
            basename = os.path.basename(opt_p)
            stem = os.path.splitext(basename)[0]
            sar_p = os.path.join(save_folder, 'sar', basename)
            h_p = os.path.join(save_folder, 'H', stem + '.txt')
            if os.path.exists(opt_p) and os.path.exists(sar_p) and os.path.exists(h_p):
                valid_opt.append(opt_p)
            else:
                for p in [opt_p, sar_p, h_p]:
                    if os.path.exists(p):
                        os.remove(p)
                print(f"  [REPAIR] {fname}: removed orphaned file {basename}")
        valid_count = len(valid_opt)
        if valid_count == expected_per_city:
            total_valid += valid_count
            print(f"  [SKIP] {fname}: already complete ({valid_count} files)")
        else:
            for p in valid_opt:
                os.remove(p)
            for p in glob.glob(os.path.join(save_folder, 'sar', fname + '_*.png')):
                os.remove(p)
            for p in glob.glob(os.path.join(save_folder, 'H', fname + '_*.txt')):
                os.remove(p)
            print(f"  [CLEANUP] {fname}: incomplete ({valid_count}/{expected_per_city}); removed for regeneration")
    print(f"  Validation complete: {total_valid} valid file pairs")
    return total_valid
def main_whu_opt_sar(base_folder, save_folder, num_sample=2, cropnum=3, max_pairs=2000, stype=None):
    for d in [save_folder,
              os.path.join(save_folder, 'opt'),
              os.path.join(save_folder, 'sar'),
              os.path.join(save_folder, 'H')]:
        os.makedirs(d, exist_ok=True)
    simgpath = []
    timgpath = []
    foldername = []
    opt_folder = os.path.join(base_folder, "optical")
    sar_folder = os.path.join(base_folder, "sar")
    opt_files = []
    for ext in ("*.tif", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
        opt_files.extend(glob.glob(os.path.join(opt_folder, ext)))
    opt_files.sort()
    for opt_file in opt_files:
        file_name = os.path.basename(opt_file)
        pair_name = os.path.splitext(file_name)[0]
        sar_file = os.path.join(sar_folder, file_name)
        if os.path.exists(sar_file):
            timgpath.append(opt_file)
            simgpath.append(sar_file)
            foldername.append(pair_name)
    expected_per_city = 2 * num_sample * cropnum  
    print("=== Checking existing output ===")
    total_valid = _cleanup_and_count(save_folder, foldername, expected_per_city)
    print(f"=== Found {total_valid} completed pairs; processing the remainder ===\n")
    total_pairs = len(simgpath)
    skipped_count = 0
    written_count = 0
    total_rows_written = total_valid
    for i in range(len(simgpath)):
        if total_rows_written >= max_pairs:
            print(f"  Maximum of {max_pairs} pairs reached; stopping")
            break
        fname = foldername[i]
        existing = len(glob.glob(os.path.join(save_folder, 'opt', fname + '_*.png')))
        if existing == expected_per_city:
            continue
        ssp = os.path.join(save_folder, "opt", fname + ".png")
        tsp = os.path.join(save_folder, "sar", fname + ".png")
        print(f"Processing image pair {i + 1}/{total_pairs}: {fname}")
        gt_homo, saved_opt_paths = generate_subdata(timgpath[i], simgpath[i], num_sample=num_sample, cropnum=cropnum, ssave_path=ssp, tsave_path=tsp, stype=stype)
        if gt_homo is None or len(gt_homo) == 0:
            print("  [WARNING] generate_subdata returned no results; skipping")
            skipped_count += 1
            continue
        verified_count = 0
        for k, (_, opt_p) in enumerate(zip(gt_homo, saved_opt_paths)):
            if opt_p is None:
                verified_count += 1
                continue
            stem = os.path.splitext(os.path.basename(opt_p))[0]
            sar_p = opt_p.replace(os.sep + 'opt' + os.sep, os.sep + 'sar' + os.sep)
            h_p = os.path.join(save_folder, 'H', stem + '.txt')
            if os.path.exists(opt_p) and os.path.exists(sar_p) and os.path.exists(h_p):
                verified_count += 1
            else:
                print(f"  [WARNING] Validation failed for matrix {k}: corresponding files are missing; discarding")
                for p in [opt_p, sar_p, h_p]:
                    if os.path.exists(p):
                        os.remove(p)
        if verified_count == 0:
            print("  [WARNING] No valid data remained after validation; skipping")
            skipped_count += 1
            continue
        remaining = max_pairs - total_rows_written
        if remaining < verified_count:
            new_opt = sorted(glob.glob(os.path.join(save_folder, 'opt', fname + '_*.png')))
            for opt_p in new_opt[remaining:]:
                stem = os.path.splitext(os.path.basename(opt_p))[0]
                sar_p = opt_p.replace(os.sep + 'opt' + os.sep, os.sep + 'sar' + os.sep)
                h_p = os.path.join(save_folder, 'H', stem + '.txt')
                for p in [opt_p, sar_p, h_p]:
                    if os.path.exists(p):
                        os.remove(p)
            verified_count = remaining
        written_count += 1
        total_rows_written += verified_count
        print(f"  [+] Completed {verified_count} pairs (total: {total_rows_written})")
    print("\n=== Generation summary ===")
    print(f"Input image pairs: {total_pairs}")
    print(f"Successfully processed: {written_count}")
    print(f"Skipped (no results): {skipped_count}")
    print(f"Total completed pairs: {total_rows_written}")
    h_count = len(glob.glob(os.path.join(save_folder, 'H', '*.txt')))
    opt_count = len(glob.glob(os.path.join(save_folder, 'opt', '*.png')))
    print(f"Homography files: {h_count}, optical images: {opt_count}")
    if h_count != opt_count:
        print("  [WARNING] Homography and optical-image counts do not match")
    print("=" * 30)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate paired optical/SAR image crops.")
    parser.add_argument("input_dir", help="Dataset root containing optical/ and sar/ directories.")
    parser.add_argument("output_dir", help="Directory in which generated data will be stored.")
    parser.add_argument("--num-sample", type=int, default=5, help="Homographies sampled per image pair.")
    parser.add_argument("--cropnum", type=int, default=4, help="Crops generated per homography and direction.")
    parser.add_argument("--max-pairs", type=int, default=80000, help="Maximum number of output pairs.")
    parser.add_argument("--stype", type=int, default=0, help="Scale-range preset used by augmentation.")
    args = parser.parse_args()
    main_whu_opt_sar(
        args.input_dir,
        args.output_dir,
        num_sample=args.num_sample,
        cropnum=args.cropnum,
        max_pairs=args.max_pairs,
        stype=args.stype,
    )
    gc.collect()
    print("Generation complete.")
