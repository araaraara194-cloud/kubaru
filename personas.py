"""旅行者(ペルソナ)を作る。

割合は data/population.json の調査データ・仮定に従ってPythonで決める。
性格の文章(description)は、AIを使う場合だけ llm.py で後から付け足す。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import World


@dataclass
class Traveler:
    id: str
    origin: str               # 外国人 / 道外(国内)
    country: str              # 外国人なら国・地域、国内なら「日本」
    first_time: bool
    age: str
    companion: str
    style: str                # 会う派 / 会わない派 / どちらでも
    interests: list
    fixed: dict = field(default_factory=dict)   # day_id -> {slot: (area_id, 予定名)}
    known: set = field(default_factory=set)     # アプリなしで知っているパッケージ
    description: str = ""

    @property
    def label(self) -> str:
        first = "初めて" if self.first_time else "リピーター"
        return f"{self.id} {self.country}・{self.age}・{self.companion}・{first}・{self.style}"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "出身": self.origin, "国・地域": self.country,
            "初めて": self.first_time, "年代": self.age, "同行者": self.companion,
            "スタイル": self.style, "興味": "、".join(self.interests),
            "人物像": self.description,
        }


def _choice(rng, values: dict):
    keys = list(values.keys())
    w = np.array([values[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=w / w.sum())]


def make_travelers(world: World, n: int = 300, seed: int = 0, quiet_share: float | None = None) -> list[Traveler]:
    """n人の旅行者を作る。quiet_share で「会わない派」の割合を上書きできる。"""
    rng = np.random.default_rng(seed)
    pop = world.population

    style_values = dict(pop["style_default"]["values"])
    if quiet_share is not None:
        rest = 1.0 - quiet_share
        meet_ratio = style_values["会う派"] / (style_values["会う派"] + style_values["どちらでも"])
        style_values = {"会う派": rest * meet_ratio, "会わない派": quiet_share, "どちらでも": rest * (1 - meet_ratio)}

    travelers = []
    for i in range(n):
        origin = _choice(rng, pop["origin"]["values"])
        country = _choice(rng, pop["foreign_country"]["values"]) if origin == "外国人" else "日本"
        first_time = bool(rng.random() < pop["first_time"]["values"][origin])
        age = _choice(rng, pop["age"]["values"])
        companion = _choice(rng, pop["companion"]["values"])
        style = _choice(rng, style_values)

        interests = [tag for tag, probs in pop["interests"]["values"].items() if rng.random() < probs[origin]]
        if style == "会わない派" and rng.random() < 0.8:
            interests.append("静けさ")
        if style == "会う派" and rng.random() < 0.8:
            interests.append("交流")
        if companion == "家族(子ども連れ)":
            interests.append("家族")
        if not interests:
            interests = ["美味しいもの"]

        # 来る前に決めてしまった定番の予定(AIは動かせない)
        p_fixed = pop["fixed_plan_probability"]["values"]["初めて" if first_time else "リピーター"]
        fixed = {}
        remaining = list(range(len(world.fixed_options)))   # 同じ予定は1回だけ
        for day in world.days:
            if remaining and rng.random() < p_fixed:
                k = remaining.pop(int(rng.integers(len(remaining))))
                opt = world.fixed_options[k]
                fixed[day["id"]] = {opt["slot"]: (opt["area"], opt["name"])}

        # アプリなしで知っている場所(定番は全員、それ以外は知名度しだい。リピーターは1.6倍知っている)
        known = set()
        for pid, pk in world.packages.items():
            if pk.matching:
                continue
            p_know = pk.awareness * (1.0 if first_time else 1.6)
            if pk.staple or rng.random() < p_know:
                known.add(pid)

        travelers.append(Traveler(
            id=f"T{i + 1:03d}", origin=origin, country=country, first_time=first_time,
            age=age, companion=companion, style=style, interests=sorted(set(interests)), fixed=fixed,
            known=known,
        ))
    return travelers
