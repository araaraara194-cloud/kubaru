"""エージェント殺到テスト。

同じ旅行者N人の代理AIエージェントが、同時に札幌の旅程を組む。
  A) 混雑マップを読むエージェント: 全員が同じ時点の混雑情報を見て、「空いている」所を選ぶ
  B) くばる窓口に聞くエージェント: 窓口が1件ずつ配るので、埋まった枠は次の人に出ない
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .personas import make_travelers
from .scoring import FREE_ID, free_time_score, rule_score
from .window import NEG, Window


def _map_agent_plan(w: Window, t, day: int, done: set, per_day: int, busy_line: float) -> dict:
    """混雑マップ(全員同じ・更新されない)を見て、自分の好みだけで選ぶ。"""
    slots = w.world.slot_ids
    pids = list(w.world.packages)
    cols = pids + [f"{FREE_ID}:{s}" for s in slots]
    m = np.full((len(slots), len(cols)), NEG)
    for i, s in enumerate(slots):
        for j, pid in enumerate(pids):
            p = w.world.packages[pid]
            if s not in p.slots or pid in done or w.world.cap(p.area, s) <= 0:
                continue
            r = w.ratio(day, s, p.area)            # マップ上の混雑率(誰も予約していない時点)
            if r is not None and r >= busy_line:
                continue                          # 「混んでいる」と表示された所は避ける
            m[i, j] = rule_score(t, p)
        m[i, len(pids) + i] = free_time_score(t)
    rows, cidx = linear_sum_assignment(-m)
    plan = {slots[r]: (FREE_ID if c >= len(pids) else pids[c]) for r, c in zip(rows, cidx)}
    chosen = sorted([s for s, pid in plan.items() if pid != FREE_ID],
                    key=lambda s: rule_score(t, w.world.packages[plan[s]]))
    while len(chosen) > per_day:
        plan[chosen.pop(0)] = FREE_ID
    return plan


def run_rush(n: int = 200, days: list[int] | None = None, seed: int = 0, per_day: int = 2,
             busy_line: float = 0.8, asahiyama_cap: int | None = None,
             follow: float = 1.0, forecast_err: float = 0.0, margin: float = 0.2) -> dict:
    """follow: 窓口の案内(案内型)に従う人の割合。従わない人は混雑マップを見て自分で選ぶ。
    forecast_err: ふだんの人出の予測のはずれ(標準偏差、割合)。実際の人出 = 予測 × (1 + はずれ)。
    margin: 案内型の安全の余裕。
    予約型の体験は席を押さえているので、従わない人も予約は守る想定。"""
    days = days or [1, 2, 3]
    base = Window.open(persist=False, asahiyama_cap=asahiyama_cap, margin=margin)
    world = base.world
    travelers = make_travelers(world, n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    # 実際のふだんの人出(予測からのはずれ)。A・Bで同じものを使う
    noise = {(d, s, a): max(0.0, 1 + rng.normal(0, forecast_err)) if forecast_err > 0 else 1.0
             for d in days for s in world.slot_ids for a in world.areas}
    followers = {t.id: rng.random() < follow for t in travelers}

    # A) マップを読むエージェント
    load_a = defaultdict(int)
    score_a = []
    map_plans = {}
    for t in travelers:
        done = set()
        for d in days:
            plan = _map_agent_plan(base, t, d, done, per_day, busy_line)
            map_plans[(t.id, d)] = plan
            for s, pid in plan.items():
                if pid == FREE_ID:
                    continue
                p = world.packages[pid]
                load_a[(d, s, p.area)] += 1
                done.add(pid)
                score_a.append(rule_score(t, p))

    # B) くばる窓口に聞くエージェント(1件ずつ順番に配られる)
    win = Window.open(persist=False, asahiyama_cap=asahiyama_cap, margin=margin)
    for t in travelers:
        win.request_plan(days=days, interests=t.interests, style=t.style, first_time=t.first_time,
                         per_day=per_day, agent="agent", traveler=t.id)
    load_plan = defaultdict(int)     # 窓口が送った人数(予定)
    load_b = defaultdict(int)        # 実際に行った人数
    score_b = []
    by_id = {t.id: t for t in travelers}
    n_ignored = 0
    for b in win.bookings:
        load_plan[(b.day, b.slot, b.area)] += b.party
        p = world.packages[b.package]
        t = by_id[b.traveler]
        if p.kind == "予約" or followers[t.id]:
            load_b[(b.day, b.slot, b.area)] += b.party
            score_b.append(rule_score(t, p))
            continue
        # 案内に従わない人: その時間帯は、混雑マップを見て自分で選んだ所へ行く
        n_ignored += 1
        alt = map_plans[(t.id, b.day)].get(b.slot, FREE_ID)
        if alt != FREE_ID and world.packages[alt].kind != "予約":
            q = world.packages[alt]
            load_b[(b.day, b.slot, q.area)] += b.party
            score_b.append(rule_score(t, q))

    def table(load, actual=True):
        rows = []
        for d in days:
            day = base._day(d)
            for s in world.slot_ids:
                for a, ar in world.areas.items():
                    cap = world.cap(a, s)
                    if cap <= 0:
                        continue
                    bl = world.baseline(a, day, s) * (noise[(d, s, a)] if actual else 1.0)
                    total = bl + load.get((d, s, a), 0)
                    rows.append({"day": d, "slot": s, "slot_name": world.slot_names[s], "area": a,
                                 "area_name": ar.name, "total": total, "cap": cap, "ratio": total / cap})
        return rows

    ta, tb = table(load_a), table(load_b)

    def summary(rows, scores):
        over = [r for r in rows if r["total"] > r["cap"] + 1e-9]
        asahi = max(r["ratio"] for r in rows if r["area"] == "asahiyama")
        return {"上限を超えたコマ数": len(over),
                "上限を超えた人数(延べ)": int(round(sum(r["total"] - r["cap"] for r in over))),
                "最大混雑率": round(max(r["ratio"] for r in rows), 2),
                "旭山の最大混雑率": round(asahi, 2),
                "配った体験の数": len(scores),
                "好みとの一致(平均点)": round(float(np.mean(scores)), 2) if scores else 0}

    return {"n": n, "days": days, "follow": follow, "forecast_err": forecast_err, "margin": margin,
            "map": {"rows": ta, "summary": summary(ta, score_a)},
            "kubaru": {"rows": tb, "summary": summary(tb, score_b),
                       "案内に従わなかった件数": n_ignored}}
