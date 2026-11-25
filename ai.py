"""希望文の読み取り(生成AI)。

環境変数 ANTHROPIC_API_KEY があれば Claude で読み取り、なければ(または失敗したら)ルールで読む。
上限を守る配分そのものは、AIに任せずルールで行う(約束を確実に守るため)。
"""
from __future__ import annotations

import json
import os
import re

MODEL = os.environ.get("KUBARU_MODEL", "claude-haiku-5-5")
STYLES = ["会う派", "会わない派", "どちらでも"]


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def read_request(text: str, tags_allowed: list[str]) -> dict | None:
    """自由文 → {"interests": [...], "style": "...", "summary": "..."}。使えないときは None。"""
    if not enabled() or not (text or "").strip():
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=15.0, max_retries=1)
        prompt = (
            "旅行者が札幌旅行の希望を書きました。次の JSON だけを返してください。\n"
            '{"interests": [...], "style": "...", "summary": "..."}\n'
            f"- interests: 次の中から当てはまるものを1〜4個(この表記のまま): {', '.join(tags_allowed)}\n"
            f"- style: {' / '.join(STYLES)} のどれか。地元の人と交流したいなら会う派、人混みや人との接触を避けたいなら会わない派\n"
            "- summary: 旅行者の希望を日本語20字以内で\n\n"
            f"希望: {text}"
        )
        msg = client.messages.create(model=MODEL, max_tokens=300,
                                     messages=[{"role": "user", "content": prompt}])
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        interests = [t for t in data.get("interests", []) if t in tags_allowed][:4]
        style = data.get("style") if data.get("style") in STYLES else None
        if not interests:
            return None
        return {"interests": interests, "style": style, "summary": str(data.get("summary", ""))[:40]}
    except Exception:
        return None
