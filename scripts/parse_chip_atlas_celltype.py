#!/usr/bin/env python3
"""
Parse cached ChIP-Atlas Target Genes TSVs to extract cell-type-resolved TF→TG scores.

Reads the per-experiment columns (format: SRX...|cell_type) from each TSV,
groups scores by cell type, and computes cell-type-specific average scores.

Output: chip_atlas_human_tf_target_by_celltype.parquet
Columns: TF, TG, cell_type, avg_score, n_experiments, genome, distance_kb
"""

import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "chip_atlas_cache"
OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "chip_atlas_human_tf_target_by_celltype.parquet"
)
INCREMENTAL_DIR = CACHE_DIR / "_incremental_celltype"

# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------


def parse_one_tsv_celltype(cache_path: Path) -> pd.DataFrame | None:
    """
    Parse a single cached TSV, extracting per-cell-type average scores.

    Returns DataFrame with columns: TF, TG, cell_type, avg_score, n_experiments
    """
    try:
        content = cache_path.read_text()
    except Exception:
        return None

    if not content.strip():
        return None

    lines = content.splitlines()
    if len(lines) < 2:
        return None

    header = lines[0].split("\t")
    if len(header) < 3:
        return None

    # Extract TF name from header[1]: "{TF}|Average"
    tf = header[1].split("|")[0] if "|" in header[1] else header[1]

    # Build cell_type → list of column indices
    ct_cols: dict[str, list[int]] = defaultdict(list)
    for i, col in enumerate(header[2:-1]):  # skip Target_genes, {TF}|Average, STRING
        idx = i + 2
        if "|" in col:
            ct = col.split("|", 1)[1].strip()
            if ct:
                ct_cols[ct].append(idx)

    if not ct_cols:
        return None

    # Pre-compute which columns belong to which cell type (as arrays for speed)
    ct_indices = {ct: np.array(idxs, dtype=int) for ct, idxs in ct_cols.items()}

    # Parse data rows
    rows = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        tg = cols[0].strip()
        if not tg:
            continue

        # For each cell type, compute average of non-zero scores
        for ct, idxs in ct_indices.items():
            scores = []
            for idx in idxs:
                if idx >= len(cols):
                    continue
                try:
                    v = float(cols[idx])
                except (ValueError, IndexError):
                    continue
                if v > 0:
                    scores.append(v)
            if scores:
                rows.append(
                    {
                        "TF": tf,
                        "TG": tg,
                        "cell_type": ct,
                        "avg_score": np.mean(scores),
                        "n_experiments": len(scores),
                    }
                )

    if not rows:
        return None

    return pd.DataFrame(rows)


def parse_task(args: tuple) -> tuple[str, bool]:
    """Wrapper for parallel execution. Returns (inc_key, success)."""
    cache_path, inc_key, genome, distance_kb = args
    try:
        df = parse_one_tsv_celltype(cache_path)
        if df is not None and len(df) > 0:
            df["genome"] = genome
            df["distance_kb"] = distance_kb
            inc_path = INCREMENTAL_DIR / f"{inc_key}.parquet"
            df.to_parquet(inc_path, index=False)
            return inc_key, True
        else:
            # Mark as empty so we don't retry
            inc_path = INCREMENTAL_DIR / f"{inc_key}.parquet"
            pd.DataFrame().to_parquet(inc_path, index=False)
            return inc_key, True
    except Exception as e:
        print(f"  [FAIL] {inc_key}: {e}", file=sys.stderr)
        return inc_key, False


def build_tasks() -> list[tuple]:
    """Scan cache directory and build parse tasks."""
    INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    already = set(f.stem for f in INCREMENTAL_DIR.glob("*.parquet"))

    tasks = []
    for genome_dir in sorted(CACHE_DIR.iterdir()):
        if not genome_dir.is_dir() or genome_dir.name.startswith("_"):
            continue
        genome = genome_dir.name
        for f in sorted(genome_dir.glob("*.tsv")):
            stem = f.stem  # e.g. "POU5F1.5"
            parts = stem.rsplit(".", 1)
            if len(parts) != 2:
                continue
            tf_safe, dist_str = parts
            try:
                distance_kb = int(dist_str)
            except ValueError:
                continue

            inc_key = f"{genome}_{tf_safe}_{dist_str}"
            if inc_key in already:
                continue

            tasks.append((f, inc_key, genome, distance_kb))

    return tasks


def main():
    tasks = build_tasks()
    if not tasks:
        print("[parse] All files already parsed.")
        # Load existing incremental files
        dfs = []
        for f in INCREMENTAL_DIR.glob("*.parquet"):
            df = pd.read_parquet(f)
            if len(df) > 0:
                dfs.append(df)
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            save_output(result)
        return

    print(f"[parse] {len(tasks)} files to parse with cell-type resolution")
    print(f"        Workers: {os.cpu_count() or 4}")

    success = 0
    fail = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(parse_task, t): t[1] for t in tasks}
        with tqdm(total=len(tasks), desc="Parsing cell-type") as pbar:
            for future in as_completed(futures):
                inc_key = futures[future]
                try:
                    _, ok = future.result()
                    if ok:
                        success += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
                pbar.set_postfix(ok=success, fail=fail)
                pbar.update(1)

    print(f"\n[parse] Done. OK={success}, FAIL={fail}")

    # Merge all incremental files into final output
    print("[merge] Loading incremental files...")
    dfs = []
    for f in tqdm(sorted(INCREMENTAL_DIR.glob("*.parquet")), desc="Merging"):
        df = pd.read_parquet(f)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        print("[merge] No data.")
        return

    result = pd.concat(dfs, ignore_index=True)
    save_output(result)


def save_output(df: pd.DataFrame):
    """Save final parquet and print summary."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n[output] Saved {len(df):,} rows to {OUTPUT_FILE}")
    print(f"[output] File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"[output] Columns: {list(df.columns)}")
    print(f"[output] Unique TFs: {df['TF'].nunique()}")
    print(f"[output] Unique TGs: {df['TG'].nunique()}")
    print(f"[output] Unique cell types: {df['cell_type'].nunique()}")

    # Per-distance stats
    for d in sorted(df["distance_kb"].unique()):
        sub = df[df["distance_kb"] == d]
        print(f"[output]   {d}kb: {len(sub):,} rows, {sub['cell_type'].nunique()} cell types")


if __name__ == "__main__":
    main()
