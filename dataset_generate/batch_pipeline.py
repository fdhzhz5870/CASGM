import gc
import os
from gen_main_whu_opt_sar import main_whu_opt_sar
from split_dataset import split_dataset
from visualize_registration import visualize_dataset
BASE_ROOT = os.environ.get("DATASET_ROOT", os.path.join(os.getcwd(), "data"))
BATCHES = [
    {"NAME_SUFFIX": "-24", "stype": 1},
    {"NAME_SUFFIX": "-12", "stype": 0},
]
def _make_datasets(NAME_SUFFIX, stype):
    return [
        {
            "src_base":    os.path.join(BASE_ROOT, "WHU-SEN-City"),
            "name":        f"WHU-SEN-City{NAME_SUFFIX}",
            "gen_params":  {"num_sample": 6, "cropnum": 5, "max_pairs": 80000, "stype": stype},
            "split_ratio": [0.7, 0.1, 0.2],
        },
    ]
RUN_GEN   = True
RUN_SPLIT = True
RUN_VIS   = True
def gen_save_folder(name):
    return os.path.join(BASE_ROOT, name, "raw")
def split_save_folder(name):
    return os.path.join(BASE_ROOT, name, "split")
def vis_save_folder(name):
    return os.path.join(BASE_ROOT, name, "vis")
def run_pipeline(ds):
    name       = ds["name"]
    src_base   = ds["src_base"]
    gp         = ds["gen_params"]
    split_ratio = ds["split_ratio"]
    gen_folder   = gen_save_folder(name)
    split_folder = split_save_folder(name)
    vis_folder   = vis_save_folder(name)
    print(f"\n{'#'*70}")
    print(f"# 数据集: {name}")
    print(f"#   src   : {src_base}")
    print(f"#   gen   : {gen_folder}")
    print(f"#   split : {split_folder}")
    print(f"#   vis   : {vis_folder}")
    print(f"{'#'*70}\n")
    if RUN_GEN:
        print(f"[{name}] === 阶段 1/3: 生成图块 ===")
        if not os.path.exists(src_base):
            print(f"  ⚠ 警告: 源目录不存在，跳过 gen: {src_base}")
        else:
            main_whu_opt_sar(
                base_folder=src_base,
                save_folder=gen_folder,
                num_sample=gp["num_sample"],
                cropnum=gp["cropnum"],
                max_pairs=gp["max_pairs"],
                stype=gp.get("stype"),
            )
            gc.collect()
            print(f"[{name}] 阶段 1 完成\n")
    else:
        print(f"[{name}] 阶段 1 跳过 (RUN_GEN=False)\n")
    if RUN_SPLIT:
        print(f"[{name}] === 阶段 2/3: 划分数据集 ===")
        if not os.path.exists(gen_folder):
            print(f"  ⚠ 警告: gen 输出目录不存在，跳过 split: {gen_folder}")
        else:
            split_dataset(
                dataset_path=gen_folder,
                split_path=split_folder,
                split_ratio=split_ratio,
                sname_list=[],
            )
            print(f"[{name}] 阶段 2 完成\n")
    else:
        print(f"[{name}] 阶段 2 跳过 (RUN_SPLIT=False)\n")
    if RUN_VIS:
        print(f"[{name}] === 阶段 3/3: 可视化验证 ===")
        if not os.path.exists(split_folder):
            print(f"  ⚠ 警告: split 输出目录不存在，跳过 vis: {split_folder}")
        else:
            visualize_dataset(
                dataset_path=split_folder,
                output_path=vis_folder,
                dataset_names=[""],
                alpha=0.5,
                split_names=["train", "val", "test"],
            )
            print(f"[{name}] 阶段 3 完成\n")
    else:
        print(f"[{name}] 阶段 3 跳过 (RUN_VIS=False)\n")
    print(f"[{name}] 全部阶段完成\n")
def main():
    for batch in BATCHES:
        suffix = batch["NAME_SUFFIX"]
        st     = batch["stype"]
        print(f"\n{'='*70}")
        print(f"= 开始批次: NAME_SUFFIX={suffix}  stype={st}")
        print(f"{'='*70}")
        for ds in _make_datasets(suffix, st):
            run_pipeline(ds)
        gc.collect()
        print(f"\n=== 批次 {suffix} 处理完毕 ===")
    print("\n=== 所有批次处理完毕 ===")
if __name__ == "__main__":
    main()
