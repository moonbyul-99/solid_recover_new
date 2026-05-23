#!/usr/bin/env python3
"""Report generation and email delivery for Solid Recover iteration log.

Generates Markdown reports and optionally sends them via SMTP to the
specified recipient.  Falls back to local-only saving if SMTP is unavailable.
"""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional


def generate_markdown_report(
    round_info: Dict,
    metrics: Dict,
    umap_paths: Dict,
    hypers: Optional[Dict] = None,
    training_info: Optional[Dict] = None,
) -> str:
    """Render a full iteration report as a Markdown string.

    Parameters
    ----------
    round_info : dict
        ``{"round": 0, "strategy": "Baseline", "notes": "", "config": "..."}``.
    metrics : dict
        Per-checkpoint ARI/NMI results keyed by ckpt_step.
    umap_paths : dict
        ``{ckpt_step: {"Class": "path.png", ...}}``.
    hypers : dict, optional
        Key hyperparameters table.
    training_info : dict, optional
        ``{"train_steps": 6000, "final_loss": 0.12, ...}``.

    Returns
    -------
    str — Markdown report content.
    """
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rnd = round_info.get("round", "?")
    name = round_info.get("strategy", "Unknown")
    config = round_info.get("config", "")

    lines.append(f"# Solid Recover 迭代报告 — Round {rnd}: {name}")
    lines.append("")
    lines.append(f"**生成时间**: {now}")
    lines.append(f"**配置文件**: `{config}`")
    lines.append(f"**策略**: {name}")
    lines.append("")

    # ---- 1. Training overview ----
    lines.append("## 一、训练概况")
    lines.append("")
    if training_info:
        lines.append("| 项目 | 值 |")
        lines.append("|------|----|")
        for k, v in training_info.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("> 基线模式 — 未训练，使用已有 embedding。")
    lines.append("")

    # ---- 2. Hyperparameters ----
    if hypers:
        lines.append("## 二、超参数")
        lines.append("")
        lines.append("| 参数 | 值 |")
        lines.append("|------|----|")
        for k, v in hypers.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # ---- 3. Quantitative metrics ----
    lines.append("## 三、定量评估结果 (ARI / NMI)")
    lines.append("")

    if not metrics:
        lines.append("> 无数据。")
    else:
        ckpt_steps = sorted(metrics.keys())
        label_keys = _collect_label_keys(metrics)

        # Header: ARI/NMI paired per label (matches row construction below)
        header_cols = ["Checkpoint"]
        for lk in label_keys:
            header_cols.append(f"{lk} ARI")
            header_cols.append(f"{lk} NMI")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["------"] * len(header_cols)) + "|")

        best_ckpt = None
        best_sum = -1
        for step in ckpt_steps:
            row = [str(step)]
            step_sum = 0
            for lk in label_keys:
                ari = metrics[step].get(lk, {}).get("best_ari", 0)
                nmi = metrics[step].get(lk, {}).get("best_nmi", 0)
                row.append(f"{ari:.4f}")
                row.append(f"{nmi:.4f}")
                step_sum += (ari or 0) + (nmi or 0)
            lines.append("| " + " | ".join(row) + " |")
            if step_sum > best_sum:
                best_sum = step_sum
                best_ckpt = step

        lines.append("")
        if best_ckpt is not None:
            lines.append(f"> 最佳 Checkpoint: **{best_ckpt}** 步")
        lines.append("")

    # ---- 4. UMAP visualizations ----
    lines.append("## 四、UMAP 可视化")
    lines.append("")

    if umap_paths:
        for ckpt_step in sorted(umap_paths.keys()):
            paths_by_color = umap_paths[ckpt_step]
            lines.append(f"### Checkpoint {ckpt_step}")
            lines.append("")
            # Show first row: Class | Subclass | cell_type
            row_keys = [k for k in ("Class", "Subclass", "cell_type") if k in paths_by_color]
            if row_keys:
                cols = " | ".join(row_keys)
                lines.append(f"| {cols} |")
                lines.append("|" + "|".join(["------"] * len(row_keys)) + "|")
                imgs = " | ".join(f"![{k}]({os.path.basename(paths_by_color[k])})" for k in row_keys)
                lines.append(f"| {imgs} |")
                lines.append("")
            # Second row: Group | Sample_ID | Region
            row_keys2 = [k for k in ("Group", "Sample_ID", "Region") if k in paths_by_color]
            if row_keys2:
                cols = " | ".join(row_keys2)
                lines.append(f"| {cols} |")
                lines.append("|" + "|".join(["------"] * len(row_keys2)) + "|")
                imgs = " | ".join(f"![{k}]({os.path.basename(paths_by_color[k])})" for k in row_keys2)
                lines.append(f"| {imgs} |")
                lines.append("")

    lines.append("---")
    lines.append(f"*报告由 `scripts/run_iteration.py` 自动生成于 {now}*")
    return "\n".join(lines)


def _collect_label_keys(metrics: Dict) -> List[str]:
    """Infer the label keys from the first checkpoint in metrics."""
    for step_data in metrics.values():
        keys = [k for k in step_data if k != "ckpt_step"]
        if keys:
            return keys
    return []


# ---------------------------------------------------------------------------
# SMTP
# ---------------------------------------------------------------------------

def test_smtp_connection(smtp_config: Dict) -> bool:
    """Try to connect and authenticate to the SMTP server (no email sent)."""
    try:
        host = smtp_config.get("host", "")
        port = int(smtp_config.get("port", 587))
        use_tls = smtp_config.get("use_tls", True)
        username = smtp_config.get("username", "")
        password = smtp_config.get("password", "")

        server = smtplib.SMTP(host, port, timeout=15)
        if use_tls:
            server.starttls()
        if username:
            server.login(username, password)
        server.quit()
        return True
    except Exception as e:
        print(f"[SMTP] Connection test failed: {e}")
        return False


def send_email_via_smtp(
    report_md: str,
    recipient: str,
    smtp_config: Dict,
    subject: Optional[str] = None,
) -> bool:
    """Send a plain-text Markdown report via SMTP."""
    try:
        host = smtp_config.get("host", "")
        port = int(smtp_config.get("port", 587))
        use_tls = smtp_config.get("use_tls", True)
        username = smtp_config.get("username", "")
        password = smtp_config.get("password", "")
        from_addr = smtp_config.get("from_addr", username or "")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject or "Solid Recover Iteration Report"
        msg["From"] = from_addr
        msg["To"] = recipient
        msg.attach(MIMEText(report_md, "plain", "utf-8"))

        server = smtplib.SMTP(host, port, timeout=30)
        if use_tls:
            server.starttls()
        if username:
            server.login(username, password)
        server.sendmail(from_addr, [recipient], msg.as_string())
        server.quit()
        print(f"[SMTP] Email sent to {recipient}")
        return True
    except Exception as e:
        print(f"[SMTP] Email send failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def send_report(
    round_info: Dict,
    metrics: Dict,
    umap_paths: Dict,
    output_dir: str,
    recipient: str = "sunrui171@mails.ucas.edu.cn",
    smtp_config: Optional[Dict] = None,
    hypers: Optional[Dict] = None,
    training_info: Optional[Dict] = None,
) -> str:
    """Generate report and deliver (email → fallback to local save).

    Returns the path to the saved Markdown file.
    """
    report_md = generate_markdown_report(
        round_info, metrics, umap_paths, hypers=hypers, training_info=training_info,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Always save locally
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[Report] Saved locally: {report_path}")

    # Also save structured metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[Report] Metrics saved: {metrics_path}")

    # Attempt email
    email_sent = False
    if smtp_config:
        ok = test_smtp_connection(smtp_config)
        if ok:
            email_sent = send_email_via_smtp(report_md, recipient, smtp_config)
        else:
            print("[Report] SMTP unavailable — report saved locally only.")
    else:
        print("[Report] No SMTP config provided — local save only.")

    if not email_sent:
        print(f"[Report] Email NOT sent; report is at {report_path}")

    return report_path


# ---------------------------------------------------------------------------
# CLI test mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test", "test-smtp"], help="Test mode")
    p.add_argument("--smtp-host", help="SMTP server hostname")
    p.add_argument("--smtp-port", type=int, default=587)
    p.add_argument("--smtp-user", help="SMTP username")
    p.add_argument("--smtp-pass", help="SMTP password")
    args = p.parse_args()

    if args.mode == "test":
        # Generate a dummy report
        ri = {"round": 0, "strategy": "Baseline (test)", "config": "config.yaml"}
        met = {
            "10000": {
                "Class": {"best_ari": 0.35, "best_nmi": 0.50, "best_resolution": 0.8},
                "Subclass": {"best_ari": 0.28, "best_nmi": 0.42, "best_resolution": 0.9},
            }
        }
        ump = {}
        out = "reports/test_report"
        report_path = send_report(ri, met, ump, out)
        print(f"\nTest report generated at: {report_path}")

    elif args.mode == "test-smtp":
        cfg = {
            "host": args.smtp_host,
            "port": args.smtp_port,
            "username": args.smtp_user,
            "password": args.smtp_pass,
            "from_addr": args.smtp_user,
        }
        ok = test_smtp_connection(cfg)
        print("SMTP test:", "OK" if ok else "FAILED")
