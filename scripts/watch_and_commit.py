#!/usr/bin/env python3
"""
exp-reports 后台监控 + 逐个实验 Git 提交
==========================================
监控 exp-reports/ 目录，每当新实验的 metrics_summary.json 出现，
立即提交到 GitHub exp-reports 分支。

用法: nohup python -u scripts/watch_and_commit.py > logs/watch_commit.out 2>&1 &
"""

import json
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/rsun@ZHANGroup.local/solid_recover_main")
EXP_REPORTS = ROOT / "exp-reports"

# 记录已提交的实验，避免重复提交
committed = set()

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def get_best_metrics(mp):
    """提取 top_1 和 foscttm"""
    with open(mp) as f:
        data = json.load(f)
    t1 = data['metrics']['top_1_hit']
    best_idx = max(range(len(t1['values'])), key=lambda i: t1['values'][i])
    return t1['values'][best_idx], data['metrics']['foscttm']['values'][best_idx]


def scan_and_commit():
    """扫描 exp-reports 中未提交的实验并逐个提交"""
    if not EXP_REPORTS.exists():
        return
    
    for phase_dir in sorted(EXP_REPORTS.iterdir()):
        if not phase_dir.is_dir():
            continue
        phase_name = phase_dir.name
        
        for exp_dir in sorted(phase_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_id = exp_dir.name
            mp = exp_dir / "metrics_summary.json"
            if not mp.exists():
                continue
            
            # 用路径唯一标识
            uid = f"{phase_name}/{exp_id}"
            if uid in committed:
                continue
            
            # 提取指标
            try:
                top1, fos = get_best_metrics(mp)
            except Exception as e:
                log(f"  ⚠️ {uid}: 读取失败 ({e})")
                continue
            
            # 提取阶段标签
            phase_label = phase_name.split('_')[-1] if '_' in phase_name else phase_name
            
            # 提交
            commit_msg = f"[exp] {phase_label} {exp_id}: top_1={top1:.6f} foscttm={fos:.6f}"
            log(f"  📤 {uid}: {commit_msg}")
            
            r = subprocess.run("git rev-parse --abbrev-ref HEAD", shell=True, cwd=ROOT, capture_output=True, text=True)
            cur = r.stdout.strip()
            
            # stash 如果有未提交更改
            stashed = False
            r = subprocess.run("git diff --quiet && git diff --cached --quiet", shell=True, cwd=ROOT)
            if r.returncode != 0:
                subprocess.run("git stash", shell=True, cwd=ROOT, capture_output=True)
                stashed = True
            
            try:
                subprocess.run("git checkout exp-reports", shell=True, cwd=ROOT, capture_output=True)
                subprocess.run(f"git add exp-reports/{uid}/", shell=True, cwd=ROOT, capture_output=True)
                
                r = subprocess.run(f'git commit -m "{commit_msg}"', shell=True, cwd=ROOT, capture_output=True, text=True)
                if r.returncode == 0:
                    r_push = subprocess.run("git push origin exp-reports", shell=True, cwd=ROOT, capture_output=True, text=True)
                    if r_push.returncode == 0:
                        log(f"  ✅ Pushed: {uid}")
                        committed.add(uid)
                    else:
                        log(f"  ⚠️ Push failed: {r_push.stderr[-100:]}")
                elif "nothing to commit" in (r.stdout + r.stderr):
                    log(f"  ⏭️  {uid}: nothing to commit")
                    committed.add(uid)
                else:
                    log(f"  ❌ Commit failed: {r.stderr[-100:]}")
            finally:
                subprocess.run(f"git checkout {cur}", shell=True, cwd=ROOT, capture_output=True)
                if stashed:
                    subprocess.run("git stash pop", shell=True, cwd=ROOT, capture_output=True)


def main():
    log("🚀 exp-reports 监控 + 逐个提交已启动")
    log(f"  监控目录: {EXP_REPORTS}")
    
    while True:
        scan_and_commit()
        time.sleep(30)  # 每 30 秒扫描一次


if __name__ == "__main__":
    main()
