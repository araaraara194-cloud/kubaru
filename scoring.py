"""旅行者 × パッケージの「好き度」(1〜5点)を付ける。

AIを使わない場合は、興味タグの一致とスタイル(会う/会わない)からルールで採点する。
AIを使う場合は llm.py が同じ形の表を返す。
"""
from __future__ import annotations

import numpy as np

from .config import Package, World
from .personas import Traveler

FREE_ID = "FREE"


def rule_score(t: Traveler, p: Package, rng: np.random.Generator | None = None) -> float:
    interests = set(t.interests)
    tags = [tag for tag in p.tags if tag != "定番"]
    match = sum(1.0 for tag in tags if tag in interests) / max(1.0, np.sqrt(len(tags)))

    bonus = 0.0
    # 会う/会わないの相性
    if p.mode == "会う":
        bonus += {"会う派": 0.8, "会わない派": -1.4, "どちらでも": 0.0}[t.style]
    elif p.mode == "会わない":
        bonus += {"会う派": -0.2, "会わない派": 0.8, "どちらでも": 0.1}[t.style]
    # 定番は初めての人に人気、リピーターは飽きている
    if p.staple:
        bonus += 0.9 if t.first_time else -0.6
    if "家族" in p.tags and t.companion == "家族(子ども連れ)":
        bonus += 0.6

    noise = rng.normal(0, 0.35) if rng is not None else 0.0
    return float(np.clip(1.2 + 2.4 * match + bonus + noise, 1.0, 5.0))


def free_time_score(t: Traveler) -> float:
    """自由時間の点数。何もしないのも、会わない派にはそれなりに価値がある。"""
    return 2.4 if t.style == "会わない派" else 2.0


def rule_scores(world: World, travelers: list[Traveler], seed: int = 0) -> dict:
    """{traveler_id: {package_id: 点数, "FREE": 点数}}"""
    rng = np.random.default_rng(seed + 1000)
    out = {}
    for t in travelers:
        s = {pid: rule_score(t, p, rng) for pid, p in world.packages.items()}
        s[FREE_ID] = free_time_score(t)
        out[t.id] = s
    return out


def rescore_with_style(world: World, t: Traveler, style: str, seed: int = 0) -> dict:
    """1人の旅行者のスタイルだけを変えて採点し直す(画面の『モード切り替え』用)。"""
    from dataclasses import replace
    t2 = replace(t, style=style)
    interests = set(t2.interests) - {"静けさ", "交流"}
    if style == "会わない派":
        interests.add("静けさ")
    elif style == "会う派":
        interests.add("交流")
    t2.interests = sorted(interests)
    rng = np.random.default_rng(seed + 7)
    s = {pid: rule_score(t2, p, rng) for pid, p in world.packages.items()}
    s[FREE_ID] = free_time_score(t2)
    return s
