"""data/ 以下のJSONを読み込んで、シミュレーションで使う形にまとめる。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Area:
    id: str
    name: str
    central: bool
    uphill: bool          # 坂の上・郊外など、ワゴンで送迎するエリア
    cap: dict             # slot -> 受け入れ上限(人)
    baseline: dict        # slot -> アプリを使わない人の数(平日)
    map: dict = field(default_factory=dict)   # 模式図上の位置 {"x","y","short"}


@dataclass
class Package:
    id: str
    name: str
    area: str
    slots: list
    mode: str             # 会う / 会わない / どちらでも
    staple: bool          # 定番かどうか
    tags: list
    awareness: float = 1.0  # アプリなしで旅行者が知っている確率
    matching: bool = False  # 地元の人とのマッチングが必要(このサービス経由でしか参加できない)
    icon: str = "・"
    kind: str = "案内"       # 予約(席を確保できる) / 案内(予約のない場所。すすめる人数だけ絞れる)
    seats: int = 0          # 予約型の1コマあたりの席数


@dataclass
class World:
    slots: list                       # [{"id","name","hours"}]
    days: list                        # [{"id","name","weekend"}]
    weekend_mult: float
    areas: dict                       # id -> Area
    packages: dict                    # id -> Package
    fixed_options: list               # 来る前に決めた定番の予定の候補
    population: dict
    free_time_name: str = "宿でのんびり・自由時間"
    slot_ids: list = field(default_factory=list)
    slot_names: dict = field(default_factory=dict)

    def baseline(self, area_id: str, day: dict, slot: str) -> float:
        base = self.areas[area_id].baseline[slot]
        return base * (self.weekend_mult if day["weekend"] else 1.0)

    def cap(self, area_id: str, slot: str) -> float:
        return self.areas[area_id].cap[slot]


def load_world(asahiyama_cap: int | None = None, n_days: int | None = None) -> World:
    """データを読み込む。asahiyama_cap を渡すと、旭山の住民上限を上書きする。"""
    a = _load("areas.json")
    p = _load("packages.json")
    pop = _load("population.json")

    areas = {}
    for ar in a["areas"]:
        cap = dict(ar["cap"])
        if ar["id"] == "asahiyama" and asahiyama_cap is not None:
            # 夜は昼の6割に(住宅地なので夜は控えめ)
            cap = {s: (round(asahiyama_cap * 0.6) if s == "night" else asahiyama_cap) for s in cap}
        areas[ar["id"]] = Area(ar["id"], ar["name"], ar["central"], ar["uphill"], cap, ar["baseline"],
                               ar.get("map", {"x": 0, "y": 0, "short": ar["name"]}))

    packages = {}
    for pk in p["packages"]:
        packages[pk["id"]] = Package(pk["id"], pk["name"], pk["area"], pk["slots"], pk["mode"], pk["staple"],
                                     pk["tags"], pk.get("awareness", 1.0), pk.get("matching", False),
                                     pk.get("icon", "・"), pk.get("kind", "案内"), pk.get("seats", 0))

    days = a["days"][: n_days] if n_days else a["days"]
    w = World(
        slots=a["slots"],
        days=days,
        weekend_mult=a["weekend_baseline_multiplier"],
        areas=areas,
        packages=packages,
        fixed_options=p["fixed_plans"]["options"],
        population=pop,
        free_time_name=p["free_time"]["name"],
    )
    w.slot_ids = [s["id"] for s in w.slots]
    w.slot_names = {s["id"]: f'{s["name"]}({s["hours"]})' for s in w.slots}
    return w
