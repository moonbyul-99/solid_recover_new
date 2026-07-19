#!/usr/bin/env python3
"""
Download ChIP-Atlas Target Genes data for all human TFs.

Downloads TF→TG binding data from ChIP-Atlas for hg19 and hg38 genomes,
at three TSS distance thresholds (1kb, 5kb, 10kb).

Output: A structured parquet file with columns:
    TF, TG, avg_score, genome, distance_kb

Usage:
    python scripts/download_chip_atlas.py                    # full download + parse
    python scripts/download_chip_atlas.py --download-only    # only download raw TSVs
    python scripts/download_chip_atlas.py --parse-only       # only parse cached TSVs
    python scripts/download_chip_atlas.py --workers 20       # set concurrency
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANALYSIS_LIST = Path(__file__).resolve().parent.parent / "analysisList.tab"
CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "chip_atlas_cache"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "outputs" / "chip_atlas_human_tf_target.parquet"
PROGRESS_FILE = CACHE_DIR / "_download_progress.json"

BASE_URL = "https://chip-atlas.dbcls.jp/data/{genome}/target/{protein}.{distance}.tsv"
DISTANCES = [1, 5, 10]  # kb from TSS
HUMAN_GENOMES = ["hg19", "hg38"]

REQUEST_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_analysis_list(path: Path) -> pd.DataFrame:
    """Parse analysisList.tab into a DataFrame of human TF entries."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            tf_name = parts[0]
            cell_type_class = parts[1] if len(parts) > 1 else ""
            genome = parts[3]
            if genome in HUMAN_GENOMES:
                rows.append(
                    {
                        "tf": tf_name,
                        "cell_type_class": cell_type_class,
                        "genome": genome,
                    }
                )
    df = pd.DataFrame(rows)
    print(f"[parse]  Parsed {len(df)} human entries from analysisList.tab")
    print(f"         Unique TFs: {df['tf'].nunique()}")
    return df


def build_download_tasks(df: pd.DataFrame) -> list[dict]:
    """Build the list of download tasks: (genome, tf, distance)."""
    pairs = df[["tf", "genome"]].drop_duplicates()
    tasks = []
    for _, row in pairs.iterrows():
        for dist in DISTANCES:
            tasks.append(
                {
                    "tf": row["tf"],
                    "genome": row["genome"],
                    "distance": dist,
                }
            )
    print(f"[tasks] Total download tasks: {len(tasks)}")
    return tasks


def get_cache_path(tf: str, genome: str, distance: int) -> Path:
    """Return the local cache path for a downloaded TSV."""
    # Use hash of tf name to avoid filesystem issues with special chars
    safe_tf = tf.replace("/", "_").replace(" ", "_")
    return CACHE_DIR / genome / f"{safe_tf}.{distance}.tsv"


def load_progress() -> dict:
    """Load download progress from JSON."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(completed: set):
    """Save set of completed cache keys to JSON."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(completed), f)


def download_one(task: dict, session: requests.Session) -> tuple[str, bool, str]:
    """
    Download a single TSV file. Returns (cache_key, success, message).
    """
    tf = task["tf"]
    genome = task["genome"]
    distance = task["distance"]
    url = BASE_URL.format(genome=genome, protein=tf, distance=distance)
    cache_path = get_cache_path(tf, genome, distance)
    cache_key = str(cache_path.relative_to(CACHE_DIR))

    # Skip if already cached
    if cache_path.exists():
        return cache_key, True, "cached"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                cache_path.write_bytes(resp.content)
                return cache_key, True, f"ok (attempt {attempt})"
            elif resp.status_code == 404:
                # Some TFs may not have data at this distance
                cache_path.write_text("")  # marker for "no data"
                return cache_key, True, f"404 (attempt {attempt})"
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    return cache_key, False, f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return cache_key, False, str(e)[:100]

    return cache_key, False, "unknown"


def download_all(
    tasks: list[dict], workers: int = 15, resume: bool = True
) -> set:
    """
    Download all TSV files with concurrent workers.
    Returns the set of successfully downloaded cache keys.
    """
    completed = set(load_progress()) if resume else set()

    # Filter out already-completed tasks
    pending = []
    for t in tasks:
        ck = str(get_cache_path(t["tf"], t["genome"], t["distance"]).relative_to(CACHE_DIR))
        if ck not in completed:
            pending.append(t)

    if not pending:
        print("[download] All files already cached.")
        return completed

    print(f"[download] Pending: {len(pending)} / {len(tasks)} tasks")
    print(f"            Workers: {workers}")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; ChIP-Atlas-downloader/1.0)"}
    )

    success_count = 0
    fail_count = 0
    checkpoint_interval = 50  # save progress every N downloads

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, t, session): t for t in pending}

        with tqdm(total=len(pending), desc="Downloading") as pbar:
            for i, future in enumerate(as_completed(futures)):
                cache_key, ok, msg = future.result()
                if ok:
                    completed.add(cache_key)
                    success_count += 1
                else:
                    fail_count += 1
                    task = futures[future]
                    tqdm.write(
                        f"  [FAIL] {task['tf']} | {task['genome']} | "
                        f"{task['distance']}kb → {msg}"
                    )

                pbar.set_postfix(ok=success_count, fail=fail_count)
                pbar.update(1)

                # Periodic checkpoint
                if (i + 1) % checkpoint_interval == 0:
                    save_progress(completed)

    save_progress(completed)
    print(f"\n[download] Done. OK={success_count}, FAIL={fail_count}")
    return completed


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_one_tsv(
    tf: str, genome: str, distance: int
) -> pd.DataFrame | None:
    """
    Parse a cached TSV file into a DataFrame with columns:
    TF, TG, avg_score.
    """
    cache_path = get_cache_path(tf, genome, distance)
    if not cache_path.exists():
        return None

    content = cache_path.read_text()
    if not content.strip():
        return None  # 404 marker

    lines = content.splitlines()
    if len(lines) < 2:
        return None

    # Header line: Target_genes, {TF}|Average, SRX..., STRING
    header = lines[0].split("\t")
    if len(header) < 2:
        return None

    # Second column is avg score
    avg_col_idx = 1

    rows = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        tg = cols[0].strip()
        try:
            score = float(cols[avg_col_idx])
        except ValueError:
            score = 0.0
        if tg:
            rows.append({"TF": tf, "TG": tg, "avg_score": score})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["genome"] = genome
    df["distance_kb"] = distance
    return df


def parse_all(completed: set, workers: int = 8) -> pd.DataFrame:
    """
    Parse all cached TSV files and merge into a single DataFrame.
    Saves incremental parquet files per TF for crash recovery.
    """
    incremental_dir = CACHE_DIR / "_incremental"
    incremental_dir.mkdir(parents=True, exist_ok=True)

    # Determine which TFs need parsing
    tasks_to_parse = []
    seen_tf_groups = set()

    for ck in sorted(completed):
        parts = ck.replace(".tsv", "").split("/")
        if len(parts) != 2:
            continue
        genome, fname = parts
        # fname = "{tf}.{distance}"
        dot_idx = fname.rfind(".")
        if dot_idx == -1:
            continue
        safe_tf = fname[:dot_idx]
        distance_str = fname[dot_idx + 1 :]
        try:
            distance = int(distance_str)
        except ValueError:
            continue

        group_key = f"{genome}/{safe_tf}"
        if group_key not in seen_tf_groups:
            seen_tf_groups.add(group_key)

    # Map safe_tf back to real TF names using the cache files
    # Actually, let's just iterate completed set directly
    # Build mapping: safe_tf → real tf name from file content

    parsed_files = set(f.stem for f in incremental_dir.glob("*.parquet"))

    # Build task list: (tf_name, genome, distance)
    parse_tasks = []
    for ck in sorted(completed):
        parts = ck.replace(".tsv", "").split("/")
        if len(parts) != 2:
            continue
        genome, fname = parts
        dot_idx = fname.rfind(".")
        if dot_idx == -1:
            continue
        safe_tf = fname[:dot_idx]
        distance_str = fname[dot_idx + 1 :]
        try:
            distance = int(distance_str)
        except ValueError:
            continue

        # Read actual TF name from file
        cache_path = CACHE_DIR / genome / f"{safe_tf}.{distance}.tsv"
        real_tf = safe_tf  # fallback
        if cache_path.exists():
            first_line = cache_path.read_text().split("\n")[0]
            header_cols = first_line.split("\t")
            if len(header_cols) >= 2:
                # "{TF}|Average"
                avg_header = header_cols[1]
                if "|" in avg_header:
                    real_tf = avg_header.split("|")[0]

        inc_key = f"{genome}_{real_tf}"
        if inc_key in parsed_files:
            continue  # already parsed

        parse_tasks.append(
            {
                "tf": real_tf,
                "safe_tf": safe_tf,
                "genome": genome,
                "distance": distance,
                "inc_key": inc_key,
            }
        )

    if not parse_tasks:
        print("[parse] All files already parsed into incremental parquets.")
        # Load all incremental files
        dfs = []
        for f in incremental_dir.glob("*.parquet"):
            dfs.append(pd.read_parquet(f))
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    # Group by inc_key to parse all 3 distances together
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for t in parse_tasks:
        groups[t["inc_key"]].append(t)

    print(f"[parse] Parsing {len(groups)} TF-groups ({len(parse_tasks)} files)...")

    all_dfs = []

    def _parse_group(inc_key: str, tasks: list[dict]) -> pd.DataFrame | None:
        parts = []
        for t in tasks:
            df = parse_one_tsv(t["tf"], t["genome"], t["distance"])
            if df is not None:
                parts.append(df)
        if not parts:
            return None
        result = pd.concat(parts, ignore_index=True)
        # Save incremental
        inc_path = incremental_dir / f"{inc_key}.parquet"
        result.to_parquet(inc_path, index=False)
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_parse_group, inc_key, ts): inc_key
            for inc_key, ts in groups.items()
        }
        with tqdm(total=len(futures), desc="Parsing") as pbar:
            for future in as_completed(futures):
                inc_key = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        all_dfs.append(df)
                except Exception:
                    tqdm.write(f"  [PARSE FAIL] {inc_key}: {traceback.format_exc()}")
                pbar.update(1)

    if not all_dfs:
        print("[parse] No data parsed.")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Download ChIP-Atlas human TF→TG data")
    parser.add_argument(
        "--download-only", action="store_true", help="Only download, skip parsing"
    )
    parser.add_argument(
        "--parse-only", action="store_true", help="Only parse cached files, skip download"
    )
    parser.add_argument(
        "--workers", type=int, default=15, help="Concurrent download workers (default: 15)"
    )
    parser.add_argument(
        "--parse-workers", type=int, default=8, help="Concurrent parse workers (default: 8)"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from progress file"
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse analysis list
    df_tf = parse_analysis_list(ANALYSIS_LIST)

    # Step 2: Build tasks
    tasks = build_download_tasks(df_tf)

    if not args.parse_only:
        # Step 3: Download
        completed = download_all(
            tasks, workers=args.workers, resume=not args.no_resume
        )
    else:
        completed = set(load_progress())
        print(f"[main] Loaded {len(completed)} cached entries from progress file.")

    if args.download_only:
        print("[main] Download-only mode. Done.")
        return

    # Step 4: Parse
    result_df = parse_all(completed, workers=args.parse_workers)

    if result_df.empty:
        print("[main] No data to save.")
        return

    # Step 5: Save final output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n[main] Saved {len(result_df):,} rows to {OUTPUT_FILE}")
    print(f"[main] Columns: {list(result_df.columns)}")
    print(f"[main] Unique TFs: {result_df['TF'].nunique()}")
    print(f"[main] Unique TGs: {result_df['TG'].nunique()}")
    print(f"[main] Genomes: {result_df['genome'].unique().tolist()}")
    print(f"[main] Distances: {sorted(result_df['distance_kb'].unique())}")


if __name__ == "__main__":
    main()
