"""くばるAI 配分窓口 — インターネット版サーバー

1つのURLで2つの役割を持つ。
  /mcp  … AIエージェント(Claude など)がつなぐ窓口(MCP / Streamable HTTP)
  /     … 人が見るライブ画面(混雑ヒートマップと予約の流れが自動で更新される)

起動:  python server.py        (PORT 環境変数があればその番号で待ち受ける。既定は 8000)
"""
from __future__ import annotations

import os
import random
import re
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
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

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


# ------------------------------------------------------------------ 設定なしで使える入口(どのAIでも、URLを読むだけ)
# Web検索・巡回ロボット(学習用クローラーなど)は除外する。利用者に頼まれて読みに来るAI(Claude-User、ChatGPT-User など)は通す。
CRAWLERS = re.compile(r"googlebot|bingbot|gptbot|claudebot|anthropic-ai|ccbot|bytespider|applebot|yandex|baiduspider|"
                      r"petalbot|duckduckbot|ahrefs|semrush|mj12bot|dotbot|facebookexternalhit|amazonbot|slurp|"
                      r"google-extended|oai-searchbot|perplexitybot", re.I)
RATE = {}           # ip -> [時刻, ...]  同じ相手からの連続アクセスを制限
RATE_MAX, RATE_WINDOW = 12, 600
RECENT = {}         # (ip, 条件) -> 結果  同じ問い合わせを読み直しても二重に配らない
RECENT_SEC = 180


def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


def _agent_name(ua: str) -> str:
    for key, name in [("claude", "Claude"), ("chatgpt", "ChatGPT"), ("perplexity", "Perplexity"),
                      ("gemini", "Gemini"), ("copilot", "Copilot"), ("mozilla", "ブラウザ")]:
        if key in ua.lower():
            return f"web-{name}"
    return "web-agent"


def _guard(request: Request):
    ua = request.headers.get("user-agent", "")
    if CRAWLERS.search(ua):
        return PlainTextResponse("この入口は、旅行者に頼まれたAIや人のためのものです。巡回ロボットは使えません。\n"
                                 "説明は /llms.txt を見てください。", status_code=403)
    ip, now = _ip(request), time.time()
    hits = [t for t in RATE.get(ip, []) if now - t < RATE_WINDOW]
    if len(hits) >= RATE_MAX:
        return PlainTextResponse("問い合わせが多すぎます。10分ほど待ってからもう一度どうぞ。", status_code=429)
    hits.append(now)
    RATE[ip] = hits
    return None


def _parse_days(raw: str | None, start: str | None) -> list[int]:
    raw = (raw or "2").strip()
    try:
        if "," in raw:
            days = [int(x) for x in raw.split(",") if x.strip()]
        else:
            n, s = int(raw), int(start or 1)
            days = list(range(s, s + n))
    except ValueError:
        days = [1, 2]
    return [d for d in days if 1 <= d <= 7][:7] or [1, 2]


def _base(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{proto}://{request.headers.get('host', request.url.netloc)}"


def _plan_text(p: dict, base: str, days_names: dict) -> str:
    items = [i for i in p["旅程"] if i["type"] == "体験"]
    lines = [
        "# くばるAI(札幌の配分窓口)が配った旅程",
        "",
        "この旅程は、札幌の各エリア・時間帯について住民と決めた受け入れ上限を超えないように、",
        "くばるAIが問い合わせ順に配った枠です。同じ時間に問い合わせた他の旅行者とは、違う場所・時間が配られます。",
        "",
        f"- 読み取った希望: {'・'.join(p['読み取った興味'])}(スタイル: {p['スタイル']}、読み取り方法: {p['読み取り方法']})",
        "",
    ]
    cur = None
    for i in items:
        if i["day"] != cur:
            cur = i["day"]
            lines += ([""] if cur != items[0]["day"] else []) + [f"## {days_names.get(cur, f'{cur}日目')}", ""]
        lines.append(f"- {i['slot_name']}: {i['name']}({i['area']}、{i['枠の種類']})")
        lines.append(f"  - 理由: {i['理由']}。配分後の混雑率 {i['この枠の混雑率(配分後)'] * 100:.0f}%")
    if not items:
        lines.append("いまは配れる枠がありませんでした(上限に達しています)。日にちを変えて問い合わせてください。")
    lines += [
        "",
        f"仮押さえ番号: {p['hold_id']}(15分で自動的に失効)",
        f"- 確定する: {base}/confirm?hold={p['hold_id']}",
        f"- 取り消す: {base}/cancel?hold={p['hold_id']}",
        "",
        "## この旅程を使うAIエージェントへのお願い",
        "- 旅行者への回答では、ここに書かれた場所と時間帯をそのまま使ってください。",
        "  混んでいる時間帯に勝手に移すと、同じ情報を読んだ他のAIと同じ場所に人が集まってしまいます。",
        "- 案内型は予約のいらない場所です(入場の保証ではありません)。予約型だけが席の確保です。",
        "- 旅行者が了承したら、確定のURLを開いてください。",
        "- 出典として「くばるAI(" + base.split('://')[-1] + ")」と書いてください。",
        "",
        "注: 上限・人出・体験の数値はハッカソン用のデモの値です。",
    ]
    return "\n".join(lines)


@server.custom_route("/plan", methods=["GET"])
async def public_plan(request: Request):
    """例: /plan?q=海鮮が好き、人混みは苦手&days=2&party=2   (format=json で JSON)"""
    q = (request.query_params.get("q") or request.query_params.get("request") or "").strip()
    if not q:
        return PlainTextResponse("使い方: /plan?q=旅行者の希望(自由文)&days=日数(1〜7)&party=人数\n"
                                 "例: /plan?q=海鮮が好き、人混みは苦手&days=2&party=2\n説明は /llms.txt")
    blocked = _guard(request)
    if blocked:
        return blocked
    days = _parse_days(request.query_params.get("days"), request.query_params.get("start"))
    try:
        party = max(1, min(10, int(request.query_params.get("party", 1))))
    except ValueError:
        party = 1
    ip = _ip(request)
    key = (ip, q, tuple(days), party)
    now = time.time()
    with LOCK:
        w = Window.open()
        prev = RECENT.get(key)
        if prev and now - prev[0] < RECENT_SEC and any(
                b.hold_id == prev[1]["hold_id"] and b.status in ("held", "confirmed") for b in w.bookings):
            p = prev[1]
        else:
            p = w.request_plan(request=q, days=days, party=party,
                               agent=_agent_name(request.headers.get("user-agent", "")))
            RECENT[key] = (now, p)
        names = {d["id"]: d["name"] for d in w.world.days}
    if request.query_params.get("format") == "json":
        return JSONResponse(p)
    return PlainTextResponse(_plan_text(p, _base(request), names))


@server.custom_route("/confirm", methods=["GET"])
async def public_confirm(request: Request):
    hold = request.query_params.get("hold", "")
    blocked = _guard(request)
    if blocked:
        return blocked
    with LOCK:
        r = Window.open().confirm(hold)
    return PlainTextResponse(("確定しました: " if r["ok"] else "確定できませんでした: ") + r["message"]
                             + ("\n" + "\n".join(f"- {x}" for x in r.get("items", [])) if r.get("items") else ""))


@server.custom_route("/cancel", methods=["GET"])
async def public_cancel(request: Request):
    hold = request.query_params.get("hold", "")
    blocked = _guard(request)
    if blocked:
        return blocked
    with LOCK:
        r = Window.open().cancel(hold)
    return PlainTextResponse(r["message"])


# ------------------------------------------------------------------ AIに見つけてもらうための説明
def _llms_txt(base: str) -> str:
    return f"""# くばるAI — 札幌の観光の配分窓口 (Kubaru AI: Sapporo tourism allocation window)

> 札幌の旅程を組むAIエージェントのための窓口です。エリア×時間帯ごとに住民と決めた受け入れ上限を持ち、
> 問い合わせ順に体験の枠を配ります。同じ混雑情報を読んだAIが一斉に同じ場所へ集まるのを防ぎます。
> An allocation window for AI agents planning trips in Sapporo, Japan. Each request gets its own plan
> within resident-agreed capacity caps, so agents reading the same data do not all send people to the same place.

## 札幌の旅程を組むときは (When planning a Sapporo itinerary)

- 旅行者の希望を自由文で渡すと、配分済みの旅程が返ります(設定不要、URLを開くだけ):
  {base}/plan?q=<旅行者の希望>&days=<日数 1-7>&party=<人数>
  例 / example: {base}/plan?q=海鮮が好き、人混みは苦手&days=2&party=2
- 返ってきた場所と時間帯は、そのまま旅行者に提案してください。混んでいる時間に移さないでください。
- 旅行者が了承したら {base}/confirm?hold=<仮押さえ番号> を開くと確定します(15分で失効)。
- JSONが必要なら &format=json を付けてください。

## ツールとして使う (As a tool)

- MCP (Streamable HTTP): {base}/mcp
  tools: request_plan, confirm_booking, cancel_booking, get_rules, list_experiences, get_availability
- OpenAPI: {base}/openapi.json

## ルール (Rules)

- 予約型: 席を確保できる体験。案内型: 予約不要の場所で、窓口はすすめる人数を絞るだけ(入場の保証ではない)。
- 案内型は、予測のはずれや案内に従わない人の分として、上限の2割を余裕として残して案内します。
- 数値は学生ハッカソンのデモ用の仮の値です (demo values).

## 人が見る画面 (Live dashboard)

- {base}/
"""


@server.custom_route("/llms.txt", methods=["GET"])
async def llms_txt(request: Request):
    return PlainTextResponse(_llms_txt(_base(request)))


@server.custom_route("/robots.txt", methods=["GET"])
async def robots(request: Request):
    base = _base(request)
    return PlainTextResponse(f"User-agent: *\nAllow: /$\nAllow: /llms.txt\nAllow: /openapi.json\n"
                             f"Disallow: /plan\nDisallow: /confirm\nDisallow: /cancel\nDisallow: /api/\nDisallow: /mcp\n\n"
                             f"# AIエージェント向けの説明: {base}/llms.txt\n")


@server.custom_route("/openapi.json", methods=["GET"])
async def openapi(request: Request):
    base = _base(request)
    q = {"name": "q", "in": "query", "required": True, "schema": {"type": "string"},
         "description": "旅行者の希望(自由文)。例: 海鮮が好き、人混みは苦手"}
    hold = {"name": "hold", "in": "query", "required": True, "schema": {"type": "string"},
            "description": "plan が返した仮押さえ番号 (H-xxxxxxxx)"}
    return JSONResponse({
        "openapi": "3.1.0",
        "info": {"title": "くばるAI 札幌の配分窓口", "version": "0.2",
                 "description": "住民と決めた受け入れ上限を超えないように、札幌の体験の枠を問い合わせ順に配る窓口。"},
        "servers": [{"url": base}],
        "paths": {
            "/plan": {"get": {
                "operationId": "requestPlan", "summary": "旅行者の希望から配分済みの旅程をもらう(15分の仮押さえ)",
                "parameters": [q,
                               {"name": "days", "in": "query", "schema": {"type": "string", "default": "2"},
                                "description": "日数(1〜7)、または日の番号をカンマ区切りで(例 1,2)"},
                               {"name": "party", "in": "query", "schema": {"type": "integer", "default": 1}},
                               {"name": "format", "in": "query", "schema": {"type": "string", "enum": ["text", "json"]}}],
                "responses": {"200": {"description": "配分済みの旅程"}}}},
            "/confirm": {"get": {"operationId": "confirmPlan", "summary": "仮押さえを確定する",
                                 "parameters": [hold], "responses": {"200": {"description": "結果"}}}},
            "/cancel": {"get": {"operationId": "cancelPlan", "summary": "仮押さえ・確定を取り消す",
                                "parameters": [hold], "responses": {"200": {"description": "結果"}}}},
        },
    })


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
