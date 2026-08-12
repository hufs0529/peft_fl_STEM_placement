"""라운드별 체크포인트 저장/복구 (계획서 v2 §6 리스크 테이블 대응책)."""

import glob
import os
from typing import Dict, Optional, Tuple

import torch


def checkpoint_path(base_dir: str, run_name: str, round_num: int) -> str:
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, f"round_{round_num:03d}.pt")


def save_checkpoint(base_dir: str, run_name: str, round_num: int, global_state_dict, extra_state=None):
    path = checkpoint_path(base_dir, run_name, round_num)
    torch.save({"round": round_num, "state_dict": global_state_dict, "extra": extra_state or {}}, path)
    return path


def find_latest_checkpoint(base_dir: str, run_name: str) -> Optional[str]:
    run_dir = os.path.join(base_dir, run_name)
    if not os.path.isdir(run_dir):
        return None
    checkpoints = sorted(glob.glob(os.path.join(run_dir, "round_*.pt")))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(path: str) -> Tuple[int, Dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu")
    return payload["round"], payload["state_dict"], payload.get("extra", {})


def resume_or_start_fresh(base_dir: str, run_name: str) -> Tuple[int, Optional[Dict[str, torch.Tensor]], dict]:
    latest = find_latest_checkpoint(base_dir, run_name)
    if latest is None:
        return 0, None, {}
    round_num, state_dict, extra = load_checkpoint(latest)
    print(f"[checkpoint] {latest} 에서 재개 — round {round_num}부터 이어서 진행")
    return round_num, state_dict, extra
