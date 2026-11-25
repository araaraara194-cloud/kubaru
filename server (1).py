"""くばるAI 配分窓口 — インターネット版サーバー

1つのURLで2つの役割を持つ。
  /mcp  … AIエージェント(Claude など)がつなぐ窓口(MCP / Streamable HTTP)
  /     … 人が見るライブ画面(混雑ヒートマップと予約の流れが自動で更新される)

起動:  python server.py        (PORT 環境変数があればその番号で待ち受ける。既定は 8000)
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _unpack_if_needed():
    """kubaru/ と data/ フォルダが無いときに組み立て直す(GitHubにフォルダを上げられなかった場合の保険)。

    1) 中身のファイルが一番上にばらばらに置かれている → kubaru/ と data/ にコピーし直す
    2) zip が置かれている → zip から取り出す
    """
    if (ROOT / "kubaru" / "window.py").exists() and (ROOT / "data" / "areas.json").exists():
        return
    if (ROOT / "window.py").exists():
        import shutil
        (ROOT / "kubaru").mkdir(exist_ok=True)
        (ROOT / "data").mkdir(exist_ok=True)
        for name in ["__init__.py", "ai.py", "config.py", "personas.py", "scoring.py", "window.py", "rush.py",
                     "viz.py", "dashboard.html"]:
            if (ROOT / name).exists():
                shutil.copy(ROOT / name, ROOT / "kubaru" / name)
        for name in ["areas.json", "packages.json", "population.json"]:
            if (ROOT / name).exists():
                shutil.copy(ROOT / name, ROOT / "data" / name)
        if not (ROOT / "kubaru" / "__init__.py").exists():
            (ROOT / "kubaru" / "__init__.py").write_text("", encoding="utf-8")
        print("[kubaru] 一番上に置かれたファイルから kubaru/ と data/ を組み立てました", flush=True)
        return
    for z in sorted(ROOT.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                rel = name.split("/", 1)[1] if "/" in name else name
                if rel.startswith(("kubaru/", "data/")):
                    dest = ROOT / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))
        if (ROOT / "kubaru" / "window.py").exists():
            print(f"[kubaru] {z.name} から kubaru/ と data/ を取り出しました", flush=True)
            return


_unpack_if_needed()
sys.path.insert(0, str(ROOT))

import uvicorn
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from kubaru import ai
from kubaru.personas import make_travelers
from kubaru.window import LOCK, Window
from mcp_server import server

try:
    from mcp.server.transport_security import TransportSecuritySettings
except ImportError:  # 古い mcp
    TransportSecuritySettings = None

DASHBOARD = Path(__file__).resolve().parent / "kubaru" / "dashboard.html"
RUSH = {"running": False, "done": 0, "total": 0}


# ------------------------------------------------------------------ 画面とAPI
@server.custom_route("/", methods=["GET"])
async def dashboard(request: Request):
    return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))


@server.custom_route("/health", methods=["GET"])
async def health(request: Request):
    return JSONResponse({"ok": True})


@server.custom_route("/api/state", methods=["GET"])
async def state(request: Request):
    day = int(request.query_params.get("day", 1))
    with LOCK:
        w = Window.open()
        world = w.world
        cells = []
        for s in world.slot_ids:
            for a, ar in world.areas.items():
                cap = world.cap(a, s)
                if cap <= 0:
                    cells.append({"area": ar.name, "slot": s, "closed": True})
                    continue
                cells.append({"area": ar.name, "slot": s, "cap": int(cap), "sent": w.booked(day, s, a),
                              "ratio": round(w.ratio(day, s, a), 3), "remaining": w.remaining(day, s, a)})
        live = [b for b in w.bookings if b.status in ("held", "confirmed")]
        feed = [{"t": time.strftime("%H:%M:%S", time.localtime(b.created)), "agent": b.agent, "traveler": b.traveler,
                 "day": b.day, "slot": world.slot_names[b.slot], "name": b.name, "area": world.areas[b.area].name,
                 "kind": world.packages[b.package].kind + "型", "party": b.party,
                 "status": {"held": "仮押さえ", "confirmed": "確定", "cancelled": "取消", "expired": "失効"}[b.status]}
                for b in sorted(w.bookings, key=lambda b: -b.created)[:25]]
        over = sum(1 for c in cells if not c.get("closed") and c["ratio"] > 1.0 + 1e-9)
        return JSONResponse({
            "day": day, "days": [{"id": d["id"], "name": d["name"]} for d in world.days],
            "areas": [ar.name for ar in world.areas.values()],
            "slots": [{"id": s, "name": world.slot_names[s]} for s in world.slot_ids],
            "cells": cells, "feed": feed,
            "kpi": {"bookings": len(live), "people": sum(b.party for b in live),
                    "agents": len({b.agent for b in live}), "over": over},
            "ai": {"enabled": ai.enabled(), "model": ai.MODEL},
            "rush": RUSH, "comfort": w.comfort,
        })


def _rush(n: int):
    RUSH.update(running=True, done=0, total=n)
    try:
        world = Window.open(persist=False).world
        travelers = make_travelers(world, n, seed=random.randint(0, 99999))
        for i, t in enumerate(travelers):
            with LOCK:
                w = Window.open()
                res = w.request_plan(interests=t.interests, style=t.style, first_time=t.first_time, days=[1, 2, 3],
                                     agent=f"demo-agent-{i + 1:02d}", traveler=t.id)
                if random.random() < 0.8:
                    Window.open().confirm(res["hold_id"])
            RUSH["done"] = i + 1
            time.sleep(0.4)
    finally:
        RUSH["running"] = False


@server.custom_route("/api/rush", methods=["POST"])
async def rush(request: Request):
    if RUSH["running"]:
        return JSONResponse({"ok": False, "message": "いま流している最中です"})
    n = max(1, min(100, int(request.query_params.get("n", 20))))
    threading.Thread(target=_rush, args=(n,), daemon=True).start()
    return JSONResponse({"ok": True, "message": f"AIエージェント{n}人を順番に流します"})


@server.custom_route("/api/reset", methods=["POST"])
async def reset(request: Request):
    with LOCK:
        Window.open().reset()
    return JSONResponse({"ok": True})


@server.custom_route("/api/try", methods=["POST"])
async def try_plan(request: Request):
    body = await request.json()
    with LOCK:
        w = Window.open()
        return JSONResponse(w.request_plan(request=body.get("text", ""), days=body.get("days") or [1, 2],
                                           party=int(body.get("party", 1)), agent="web-visitor"))


# ------------------------------------------------------------------ 起動
def build_app():
    kw = {"host": "0.0.0.0", "stateless_http": True}
    if TransportSecuritySettings is not None:
        # 公開サーバーなので、どのドメイン名で来ても受け付ける(デモ用)
        kw["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return server.streamable_http_app(**kw)


app = build_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
