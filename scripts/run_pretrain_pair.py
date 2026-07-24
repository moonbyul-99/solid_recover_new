#!/usr/bin/env python3
"""一键编排脚本：读取 YAML config，自动执行单组学预训练 + 配对联合训练。

用法::

    python scripts/run_pretrain_pair.py configs/case_renal_cortex_pretrain_pair.yaml
    python scripts/run_pretrain_pair.py configs/case_renal_cortex.yaml --device cpu

若 config 含 ``pretrain_rna`` / ``pretrain_atac`` 段落，先分别执行单组学 VAE 预训练
（合并 train.h5mu + test.h5mu 全部细胞），再自动生成临时 pair_pretrain config 并调用
CLI 执行配对训练。

若无 ``pretrain_*`` 段落，直接透传给 CLI，完全兼容现有 pair_scratch / pair_pretrain。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import anndata as ad
import muon as mu
import yaml

# 确保项目根目录在 sys.path 中（无需安装 solid-recover）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def header(msg: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n[{_now()}] {msg}\n{bar}\n", flush=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_final_ckpt(models_dir: str) -> Optional[str]:
    """在 ``models_dir`` 下找 step 号最大的 ``ckpt_*.pth``。"""
    pattern = os.path.join(models_dir, "ckpt_*.pth")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None

    def _step(p: str) -> int:
        m = re.search(r"ckpt_(\d+)\.pth", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(ckpts, key=_step)[-1]


def find_pair_output_dir(base_project_dir: str) -> Optional[str]:
    """Find the actual pair-training output directory.

    The trainer may append a timestamp suffix to ``base_project_dir``.
    Returns the path of the directory that contains ``config.yaml``
    (preferring the most recent if multiple matches exist).
    """
    # 1) exact match
    if os.path.isfile(os.path.join(base_project_dir, "config.yaml")):
        return base_project_dir
    # 2) timestamped variants
    parent = os.path.dirname(base_project_dir)
    prefix = os.path.basename(base_project_dir)
    if not os.path.isdir(parent):
        return None
    candidates = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if name.startswith(prefix) and os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.yaml")):
            candidates.append(full)
    if not candidates:
        return None
    # return the most recently modified
    return max(candidates, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# single-omics pretraining
# ---------------------------------------------------------------------------

def run_pretrain_omic(
    cfg: Dict[str, Any],
    pretrain_section: Dict[str, Any],
    omic_label: str,  # "rna" or "atac"
    modality_key: str,
    hidden_params_key: str,  # "hidden_params_1" or "hidden_params_2"
    device: str,
) -> str:
    """执行单个组学的 VAE 预训练，返回最终 checkpoint 路径。

    Parameters
    ----------
    cfg : dict
        完整 YAML 配置。
    pretrain_section : dict
        ``pretrain_rna`` 或 ``pretrain_atac`` 段落。
    omic_label : str
        日志标签（"RNA" 或 "ATAC"）。
    modality_key : str
        h5mu 中的模态 key（"rna_count" 或 "peak_count"）。
    hidden_params_key : str
        model 区块中的隐层参数 key（"hidden_params_1" 或 "hidden_params_2"）。
    device : str
        训练设备。
    """
    from solid_recover.models.single import SinglePretrain

    header(f"阶段：{omic_label} 单组学预训练")

    # ---- 合并 train + test 数据 ----
    train_path = cfg["data"]["train_data_path"]
    test_path = cfg["data"]["test_data_path"]
    log(f"加载 train: {train_path}")
    mdata_train = mu.read_h5mu(train_path)
    log(f"加载 test:  {test_path}")
    mdata_test = mu.read_h5mu(test_path)

    adata_train = mdata_train[modality_key]
    adata_test = mdata_test[modality_key]
    log(f"Train {omic_label} cells: {adata_train.n_obs}, features: {adata_train.n_vars}")
    log(f"Test  {omic_label} cells: {adata_test.n_obs}, features: {adata_test.n_vars}")

    # concat all cells
    adata_all = ad.concat([adata_train, adata_test], join="outer")
    log(f"合并后总细胞数: {adata_all.n_obs}")

    # ---- 模型参数 ----
    feature_num = (
        cfg["model"]["feature_num_1"]
        if hidden_params_key == "hidden_params_1"
        else cfg["model"]["feature_num_2"]
    )
    hidden_params = list(cfg["model"][hidden_params_key])
    embed_dim = cfg["model"]["embed_dim"]
    use_rmsnorm = cfg["model"].get("use_rmsnorm", True)
    use_residual = cfg["model"].get("use_residual", True)
    dropout_p = cfg["model"].get("dropout_p", 0.0)

    # ---- 预训练超参数 ----
    project_dir = pretrain_section["project_dir"]
    train_steps = pretrain_section["train_steps"]
    eval_points = pretrain_section["eval_points"]
    save_points = pretrain_section["save_points"]
    batch_size = pretrain_section["batch_size"]
    lr = float(pretrain_section["lr"])
    warmup_steps = pretrain_section["warmup_steps"]
    steady_1_steps = pretrain_section["steady_1_steps"]
    cosine_anneal_steps = pretrain_section["cosine_anneal_steps"]
    min_lr = float(pretrain_section["min_lr"])
    vae_beta = float(pretrain_section["vae_beta"])

    log(f"架构: hidden_params={hidden_params}, embed_dim={embed_dim}")
    log(f"训练: train_steps={train_steps}, batch_size={batch_size}, lr={lr}")
    log(f"调度: warmup={warmup_steps}, steady={steady_1_steps}, cosine={cosine_anneal_steps}")
    log(f"输出: {project_dir}")

    # ---- 构建并训练 ----
    model = SinglePretrain(
        feature_num=feature_num,
        hidden_params=hidden_params,
        embed_dim=embed_dim,
        use_rmsnorm=use_rmsnorm,
        use_residual=use_residual,
        dropout_p=dropout_p,
    )
    train_ds, test_ds = model.create_dataset(adata_all, test_size=0.05, random_state=42)
    # create_dataset 内部调用 setup_data 时未传 batch_size，默认 128；
    # 这里重新 setup_data 以使用 config 中指定的 batch_size。
    model.setup_data(train_ds, test_ds, batch_size=batch_size)
    model.set_loss(beta=vae_beta)
    model.configure_optimizer(
        lr=lr,
        warmup_steps=warmup_steps,
        steady_1_steps=steady_1_steps,
        cosine_anneal_steps=cosine_anneal_steps,
        min_lr=min_lr,
    )
    model.set_project(project_dir)
    model.train(
        train_steps=train_steps,
        eval_points=eval_points,
        save_points=save_points,
        device=device,
    )

    # project_dir 可能被 Trainer 追加了时间戳
    actual_project_dir = model.project_dir
    log(f"实际输出目录: {actual_project_dir}")

    # ---- 发现最终 checkpoint ----
    models_dir = os.path.join(actual_project_dir, "models")
    ckpt_path = find_final_ckpt(models_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"未在 {models_dir} 中找到 checkpoint")
    ckpt_path = os.path.abspath(ckpt_path)
    log(f"最终 checkpoint: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"  {omic_label} 预训练完成 ✅")
    print(f"  checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")
    return ckpt_path


# ---------------------------------------------------------------------------
# generate pair_pretrain config
# ---------------------------------------------------------------------------

def generate_pair_pretrain_config(
    cfg: Dict[str, Any],
    ckpt_rna: str,
    ckpt_atac: str,
    output_path: str,
) -> None:
    """基于原 config 生成 pair_pretrain 临时 YAML。"""
    import copy

    pair_cfg = copy.deepcopy(cfg)

    # 切换 task 为 pair_pretrain
    pair_cfg["task"] = "pair_pretrain"

    # 移除 pretrain 相关段落（CLI 不需要这些字段）
    pair_cfg.pop("pretrain_rna", None)
    pair_cfg.pop("pretrain_atac", None)

    # 填入 checkpoint 路径
    pair_cfg["ckpt"] = {
        "omic_1": ckpt_rna,
        "omic_2": ckpt_atac,
    }

    with open(output_path, "w") as f:
        yaml.dump(pair_cfg, f, default_flow_style=False, sort_keys=False)

    log(f"临时 pair_pretrain config 已生成: {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="一键编排：单组学预训练 → 配对联合训练",
    )
    parser.add_argument(
        "config",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="训练设备 (default: cuda)",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        log(f"错误: 配置文件不存在: {config_path}")
        return 1

    cfg = load_yaml(config_path)
    log(f"加载 config: {config_path}")

    has_pretrain_rna = "pretrain_rna" in cfg
    has_pretrain_atac = "pretrain_atac" in cfg

    # ---- 路径：无 pretrain → 直接透传 ----
    if not has_pretrain_rna and not has_pretrain_atac:
        header("无 pretrain_* 段落，直接执行 CLI 训练")
        cmd = [
            sys.executable, "-m", "solid_recover.cli.main", "train",
            "--config", config_path,
        ]
        log(f"执行: {' '.join(cmd)}")
        return subprocess.call(cmd)

    # ---- 路径：有 pretrain → 预训练 + 配对训练 ----
    t_start = time.time()
    ckpt_rna: Optional[str] = None
    ckpt_atac: Optional[str] = None

    if has_pretrain_rna:
        ckpt_rna = run_pretrain_omic(
            cfg=cfg,
            pretrain_section=cfg["pretrain_rna"],
            omic_label="RNA",
            modality_key=cfg["data"]["key_1"],
            hidden_params_key="hidden_params_1",
            device=args.device,
        )

    if has_pretrain_atac:
        ckpt_atac = run_pretrain_omic(
            cfg=cfg,
            pretrain_section=cfg["pretrain_atac"],
            omic_label="ATAC",
            modality_key=cfg["data"]["key_2"],
            hidden_params_key="hidden_params_2",
            device=args.device,
        )

    # ---- 生成临时 pair_pretrain config ----
    assert ckpt_rna is not None and ckpt_atac is not None
    tmp_config = os.path.join(tempfile.gettempdir(), "pair_pretrain_renal_cortex.yaml")
    generate_pair_pretrain_config(cfg, ckpt_rna, ckpt_atac, tmp_config)

    header("阶段：配对联合训练 (pair_pretrain)")
    log(f"RNA  ckpt: {ckpt_rna}")
    log(f"ATAC ckpt: {ckpt_atac}")

    cmd = [
        sys.executable, "-m", "solid_recover.cli.main", "train",
        "--config", tmp_config,
    ]
    log(f"执行: {' '.join(cmd)}")
    exit_code = subprocess.call(cmd)

    # ---- 收尾 ----
    if exit_code == 0:
        log("配对训练完成 ✅")
        # 保存完整配置（含 pretrain_* 段落）到输出目录
        pair_output_dir = find_pair_output_dir(cfg["training"]["project_dir"])
        if pair_output_dir is not None:
            config_all_path = os.path.join(pair_output_dir, "config_all.yaml")
            with open(config_all_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            log(f"完整配置已保存: {config_all_path}")
        else:
            log("⚠ 未找到配对训练输出目录，跳过 config_all.yaml 保存")
    else:
        log(f"配对训练失败，退出码: {exit_code}")

    # 清理临时文件
    try:
        os.remove(tmp_config)
    except OSError:
        pass

    elapsed = time.time() - t_start
    log(f"总耗时: {elapsed / 60:.1f} 分钟")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
