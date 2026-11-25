"""街の配分窓口(本体)。

旅行者のアプリやAIエージェントから「この人の旅程をください」と問い合わせを受け、
住民と決めた上限(エリア×時間帯)を超えない範囲で、体験の枠を配る。

- 1件ずつ順番に配るので、後から来た人は「すでに埋まった枠」を避けて配られる
  (全員が同じ混雑マップを見て同じ所へ殺到する、を構造的に防ぐ)
- 配った枠はまず「仮押さえ」になり、確定・取消できる。仮押さえは一定時間で自動的に失効する
- 配分の理由と、上限などのルールは誰でも読めるように返す
"""
from __future__ import annotations

import json
import os
import threading
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import ai
from .config import World, load_world
from .personas import Traveler
from .scoring import FREE_ID, free_time_score, rule_score

STATE_PATH = Path(os.environ.get("KUBARU_STATE", Path(__file__).resolve().parent.parent / "state" / "bookings.json"))
HOLD_MINUTES = 15
LOCK = threading.RLock()   # 同時に来た問い合わせを1件ずつ処理するための鍵
NEG = -1e6

# 自由文の希望 → 興味タグ(日本語・英語の簡単な辞書。生成AIを使わなくても動くように)
KEYWORDS = {
    "美味しいもの": ["食", "グルメ", "美味", "海鮮", "寿司", "ラーメン", "ジンギスカン", "カニ", "food", "eat", "gourmet", "seafood", "ramen", "sushi"],
    "観光スポット": ["観光", "名所", "定番", "sightseeing", "landmark", "famous"],
    "買い物": ["買い物", "ショッピング", "お土産", "shopping", "souvenir"],
    "自然": ["自然", "森", "山", "公園", "紅葉", "雪", "nature", "forest", "mountain", "park", "hike", "hiking"],
    "写真": ["写真", "撮影", "映え", "photo", "instagram"],
    "歴史・学び": ["歴史", "学び", "博物館", "大学", "研究", "history", "museum", "learn", "university", "science"],
    "夜景": ["夜景", "night view", "city lights"],
    "温泉": ["温泉", "onsen", "hot spring", "spa"],
    "アクティビティ": ["登山", "アクティビティ", "スキー", "体を動かす", "activity", "ski", "trekking"],
    "夜": ["飲み", "はしご", "バー", "居酒屋", "nightlife", "bar", "drink"],
    "家族": ["家族", "子ども", "子供", "キッズ", "family", "kids", "children"],
    "交流": ["地元", "交流", "話", "ローカル", "人と", "local", "people", "meet", "talk"],
    "静けさ": ["静か", "のんびり", "ゆっくり", "ひとり", "一人", "人混み", "混雑が苦手", "quiet", "calm", "slow", "relax", "alone", "avoid crowds"],
}


def parse_interests(text: str) -> list[str]:
    t = (text or "").lower()
    tags = [tag for tag, words in KEYWORDS.items() if any(w.lower() in t for w in words)]
    return tags or ["美味しいもの", "観光スポット"]


def guess_style(tags: list[str], text: str = "") -> str:
    t = (text or "").lower()
    if "静けさ" in tags or re.search(r"会いたくない|人と会わない|avoid people", t):
        return "会わない派"
    if "交流" in tags:
        return "会う派"
    return "どちらでも"


@dataclass
class Booking:
    id: str
    hold_id: str
    agent: str
    traveler: str
    day: int
    slot: str
    package: str
    name: str
    area: str
    party: int
    status: str            # held / confirmed / cancelled / expired
    created: float
    reason: str = ""


@dataclass
class Window:
    world: World
    comfort: float = 0.7           # 快適ライン(上限の何割まで快適とみなすか)
    lam: float = 0.6               # 快適ラインを超える枠に配るときの減点
    margin: float = 0.2            # 案内型の安全の余裕(上限の何割を、予測のはずれ・案内に従わない人のために残すか)
    bookings: list = field(default_factory=list)
    persist: bool = True

    # ------------------------------------------------------------ 保存・読み込み
    @classmethod
    def open(cls, path: Path | None = None, persist: bool = True, asahiyama_cap: int | None = None,
             margin: float | None = None) -> "Window":
        w = cls(world=load_world(asahiyama_cap=asahiyama_cap), persist=persist)
        if margin is not None:
            w.margin = margin
        p = path or STATE_PATH
        if persist and p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            w.bookings = [Booking(**b) for b in raw.get("bookings", [])]
        w._expire()
        return w

    def save(self, path: Path | None = None) -> None:
        if not self.persist:
            return
        p = path or STATE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"bookings": [b.__dict__ for b in self.bookings]}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(p)

    def reset(self) -> None:
        self.bookings = []
        self.save()

    def _expire(self) -> None:
        now = time.time()
        for b in self.bookings:
            if b.status == "held" and now - b.created > HOLD_MINUTES * 60:
                b.status = "expired"

    # ------------------------------------------------------------ 枠の計算
    def _day(self, day: int) -> dict:
        for d in self.world.days:
            if d["id"] == day:
                return d
        raise ValueError(f"day は 1〜{len(self.world.days)} で指定してください")

    def booked(self, day: int, slot: str, area: str) -> int:
        return sum(b.party for b in self.bookings
                   if b.status in ("held", "confirmed") and b.day == day and b.slot == slot and b.area == area)

    def occupied(self, day: int, slot: str, area: str) -> float:
        """アプリを使わない人(baseline) + このサービスで配った人。"""
        return self.world.baseline(area, self._day(day), slot) + self.booked(day, slot, area)

    def seats_left(self, day: int, slot: str, pid: str) -> int:
        p = self.world.packages[pid]
        used = sum(b.party for b in self.bookings
                   if b.status in ("held", "confirmed") and b.day == day and b.slot == slot and b.package == pid)
        return max(0, p.seats - used)

    def sendable(self, day: int, slot: str, pid: str) -> int:
        """窓口がこの体験に、あと何人すすめられるか。

        予約型: 席の残り と エリアの残り(上限 - 予測人出 - 送った人数)の小さい方。
        案内型: エリアの残りから、さらに安全の余裕(上限 × margin)を引いた分。
        """
        p = self.world.packages[pid]
        cap = self.world.cap(p.area, slot)
        area_left = cap - self.occupied(day, slot, p.area)
        if p.kind == "予約":
            return int(max(0, min(self.seats_left(day, slot, pid), np.floor(area_left))))
        return int(max(0, np.floor(area_left - self.margin * cap)))

    def remaining(self, day: int, slot: str, area: str) -> int:
        return int(max(0, np.floor(self.world.cap(area, slot) - self.occupied(day, slot, area))))

    def ratio(self, day: int, slot: str, area: str) -> float | None:
        cap = self.world.cap(area, slot)
        return None if cap <= 0 else self.occupied(day, slot, area) / cap

    # ------------------------------------------------------------ 公開する情報
    def rules(self) -> dict:
        return {
            "説明": "くばるAIは、札幌の各エリア・時間帯の受け入れ上限(住民と決めた人数)を超えないように、"
                    "問い合わせ順に体験の枠を配ります。上限に達した枠は提案しません。"
                    "上限の7割(快適ライン)を超える枠は、他に良い選択肢があれば避けます。",
            "枠の定義": {
                "枠": "窓口がその場所・時間帯に送る人数の上限。実際にそこにいる人数のことではない。",
                "予約型": "席を確保できる体験(地元の人と会う体験・飲食店など)。仮押さえ→確定で席が確実に押さえられる。",
                "案内型": "予約のない場所(公園・市場・街歩きなど)。窓口はすすめる人数を絞れるだけ。"
                          "実際の人数は ふだんの人出(予測) + 案内した人のうち実際に行く人 で決まるので、"
                          f"上限の{self.margin:.0%}を安全の余裕として残して案内する。",
                "混雑率": "(ふだんの人出の予測 + 窓口が送った人数) ÷ 上限",
            },
            "快適ライン": self.comfort,
            "仮押さえの有効時間(分)": HOLD_MINUTES,
            "時間帯": [f'{s["name"]}({s["hours"]})' for s in self.world.slots],
            "エリアごとの上限": {ar.name: ar.cap for ar in self.world.areas.values()},
            "注意": "上限・混雑の数値はデモ用の仮の値です。",
        }

    def availability(self, day: int, slot: str | None = None) -> list[dict]:
        out = []
        slots = [slot] if slot else self.world.slot_ids
        for s in slots:
            for a, ar in self.world.areas.items():
                cap = self.world.cap(a, s)
                if cap <= 0:
                    continue
                r = self.ratio(day, s, a)
                out.append({
                    "day": day, "slot": s, "slot_name": self.world.slot_names[s], "area": ar.name,
                    "上限": int(cap), "残り": self.remaining(day, s, a),
                    "案内できる残り": int(max(0, np.floor(self.remaining(day, s, a) - self.margin * cap))),
                    "混雑率": round(r, 2), "状態": "満員" if self.remaining(day, s, a) == 0
                    else ("混みぎみ" if r > self.comfort else "空いている"),
                })
        return out

    def experiences(self) -> list[dict]:
        return [{"id": p.id, "name": p.name, "area": self.world.areas[p.area].name,
                 "枠の種類": p.kind + "型", "席数(1コマ)": p.seats if p.kind == "予約" else None,
                 "時間帯": [self.world.slot_names[s] for s in p.slots], "mode": p.mode, "tags": p.tags}
                for p in self.world.packages.values()]

    # ------------------------------------------------------------ 配る
    def request_plan(self, request: str = "", days: list[int] | None = None, interests: list[str] | None = None,
                     style: str | None = None, first_time: bool = True, party: int = 1, per_day: int = 2,
                     busy: dict | None = None, agent: str = "unknown", traveler: str = "") -> dict:
        """旅行者1人(1グループ)分の旅程を配り、仮押さえする。

        request: 自由文の希望(例「海鮮が好き。人混みは苦手」)
        days: 滞在する日(1〜7)。省略時は1〜3日目
        busy: すでに予定がある時間 {"1": ["afternoon"], ...}
        """
        self._expire()
        days = days or [1, 2, 3]
        method, summary = "指定", ""
        if interests:
            tags = interests
        else:
            read = ai.read_request(request, list(KEYWORDS))
            if read:
                tags, method, summary = read["interests"], f"AI({ai.MODEL})", read["summary"]
                style = style or read["style"]
            else:
                tags, method = parse_interests(request), "ルール(キーワード)"
        style = style or guess_style(tags, request)
        t = Traveler(id=traveler or f"guest-{uuid.uuid4().hex[:4]}", origin="外国人", country="-",
                     first_time=first_time, age="-", companion="家族(子ども連れ)" if "家族" in tags else "-",
                     style=style, interests=tags)
        scores = {pid: rule_score(t, p) for pid, p in self.world.packages.items()}
        scores[FREE_ID] = free_time_score(t)
        busy = {int(k): set(v) for k, v in (busy or {}).items()}

        hold_id = "H-" + uuid.uuid4().hex[:8]
        items, done = [], set()
        for day in days:
            d = self._day(day)
            slots = [s for s in self.world.slot_ids if s not in busy.get(day, set())]
            plan = self._best_day(day, slots, scores, done, party, per_day)
            for s in self.world.slot_ids:
                if s in busy.get(day, set()):
                    items.append({"day": day, "slot": s, "slot_name": self.world.slot_names[s], "type": "あなたの予定"})
                    continue
                pid = plan.get(s, FREE_ID)
                if pid == FREE_ID:
                    continue
                p = self.world.packages[pid]
                r_before = self.ratio(day, s, p.area)
                reason = self._reason(p, tags, r_before)
                b = Booking(id="B-" + uuid.uuid4().hex[:8], hold_id=hold_id, agent=agent, traveler=t.id,
                            day=day, slot=s, package=pid, name=p.name, area=p.area, party=party,
                            status="held", created=time.time(), reason=reason)
                self.bookings.append(b)
                done.add(pid)
                items.append({"day": day, "day_name": d["name"], "slot": s, "slot_name": self.world.slot_names[s],
                              "type": "体験", "name": p.name, "area": self.world.areas[p.area].name,
                              "枠の種類": p.kind + "型",
                              "booking_id": b.id, "理由": reason,
                              "この枠の混雑率(配分後)": round(self.ratio(day, s, p.area), 2)})
        self.save()
        return {
            "hold_id": hold_id,
            "状態": "仮押さえ",
            "有効時間(分)": HOLD_MINUTES,
            "読み取った興味": tags,
            "読み取り方法": method,
            **({"希望の要約": summary} if summary else {}),
            "スタイル": style,
            "旅程": sorted(items, key=lambda x: (x["day"], self.world.slot_ids.index(x["slot"]))),
            "次にすること": "旅行者が了承したら confirm_booking に hold_id を渡して確定してください。"
                          "予約型は席が確定します。案内型は予約不要の場所で、窓口が案内人数として記録します(入場の保証ではありません)。",
        }

    def _best_day(self, day: int, slots: list[str], scores: dict, done: set, party: int, per_day: int) -> dict:
        if not slots:
            return {}
        pids = list(self.world.packages)
        cols = pids + [f"{FREE_ID}:{s}" for s in slots]
        m = np.full((len(slots), len(cols)), NEG)
        for i, s in enumerate(slots):
            for j, pid in enumerate(pids):
                p = self.world.packages[pid]
                if s not in p.slots or pid in done or self.world.cap(p.area, s) <= 0:
                    continue
                if self.sendable(day, s, pid) < party:
                    continue                                  # 送れる人数を超える枠は配らない
                after = (self.occupied(day, s, p.area) + party) / self.world.cap(p.area, s)
                penalty = self.lam if after > self.comfort else 0.0
                m[i, j] = scores[pid] - penalty
            m[i, len(pids) + i] = scores[FREE_ID]
        rows, cidx = linear_sum_assignment(-m)
        plan = {slots[r]: (FREE_ID if c >= len(pids) else pids[c]) for r, c in zip(rows, cidx)}
        chosen = sorted([s for s, pid in plan.items() if pid != FREE_ID], key=lambda s: scores[plan[s]])
        while len(chosen) > per_day:
            plan[chosen.pop(0)] = FREE_ID
        return plan

    def _reason(self, p, tags: list[str], ratio_before: float | None) -> str:
        hit = [t for t in p.tags if t in tags]
        parts = []
        if hit:
            parts.append("興味(" + "・".join(hit) + ")に合う")
        if ratio_before is not None:
            if ratio_before <= self.comfort:
                parts.append(f"この時間はまだ空いている(混雑率{ratio_before * 100:.0f}%)")
            else:
                parts.append(f"やや混んでいるが他に合う枠がない(混雑率{ratio_before * 100:.0f}%)")
        if p.matching:
            parts.append("地元の人と会える、このサービス限定の体験")
        return "、".join(parts) or "空いている枠"

    # ------------------------------------------------------------ 確定・取消
    def confirm(self, hold_id: str) -> dict:
        self._expire()
        bs = [b for b in self.bookings if b.hold_id == hold_id]
        if not bs:
            return {"ok": False, "message": "その hold_id は見つかりません"}
        if any(b.status == "expired" for b in bs):
            return {"ok": False, "message": "仮押さえの有効時間が切れました。もう一度 request_plan してください"}
        for b in bs:
            if b.status == "held":
                b.status = "confirmed"
        self.save()
        kinds = [self.world.packages[b.package].kind for b in bs]
        return {"ok": True, "message": f"予約型{kinds.count('予約')}件の席を確定し、案内型{kinds.count('案内')}件を案内人数として記録しました",
                "items": [f"{b.name}({self.world.packages[b.package].kind}型)" for b in bs]}

    def cancel(self, hold_id: str) -> dict:
        bs = [b for b in self.bookings if b.hold_id == hold_id and b.status in ("held", "confirmed")]
        for b in bs:
            b.status = "cancelled"
        self.save()
        return {"ok": bool(bs), "message": f"{len(bs)}件を取り消し、枠を戻しました"}
