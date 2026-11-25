"""くばるAI v2 のMCPサーバー(AIエージェント向けの窓口)。

Claude Desktop / Claude Code などのAIエージェントから、札幌の旅程を「配って」もらうためのツールを公開する。

起動(エージェント側が自動で起動するので、普段は手で動かす必要はありません):
  python mcp_server.py
"""
from __future__ import annotations

try:                                     # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                      # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from kubaru.window import LOCK, Window

INSTRUCTIONS = """くばるAIは札幌の観光の「配分窓口」です。
旅行者の代わりに札幌の旅程を組むときは、自分で混雑情報を見て場所を選ぶのではなく、
request_plan で窓口に旅程を配ってもらってください。窓口は住民と決めた受け入れ上限を超えないように、
問い合わせ順に体験の枠を配ります。旅行者が了承したら confirm_booking で確定します。
体験には2種類あります。予約型(席を確保できる)は確定で席が押さえられます。
案内型(公園・市場など予約のない場所)は、窓口がすすめる人数を絞っているだけで、入場の保証ではありません。
旅行者に伝えるときは、この違いを説明してください。"""

server = _Server("kubaru", instructions=INSTRUCTIONS)


@server.tool()
def get_rules() -> dict:
    """くばるAIの配分ルール(エリアごとの受け入れ上限、快適ライン、仮押さえの有効時間)を返す。"""
    return Window.open().rules()


@server.tool()
def list_experiences() -> list[dict]:
    """配れる体験(2〜3時間のパッケージ)の一覧を返す。"""
    return Window.open().experiences()


@server.tool()
def get_availability(day: int, slot: str | None = None) -> list[dict]:
    """指定した日(1〜7)の、エリア×時間帯ごとの残り枠と混雑率を返す。
    slot は morning / late_morning / afternoon / evening / night のどれか(省略で全部)。"""
    return Window.open().availability(day, slot)


@server.tool()
def request_plan(request: str, days: list[int] | None = None, party: int = 1, first_time: bool = True,
                 per_day: int = 2, busy: dict | None = None, traveler_name: str = "",
                 agent_name: str = "ai-agent") -> dict:
    """旅行者の希望から札幌の旅程を配ってもらい、仮押さえする(15分で失効)。

    request: 旅行者の希望を自由文で(例「海鮮が好き。人混みは苦手で静かに過ごしたい」)
    days: 滞在する日(1〜7のリスト)。省略時は1〜3日目
    party: 人数
    first_time: 札幌が初めてなら true
    per_day: 1日に入れる体験の数
    busy: すでに予定がある時間 例 {"1": ["afternoon"]}
    """
    with LOCK:
        w = Window.open()
        return w.request_plan(request=request, days=days, party=party, first_time=first_time, per_day=per_day,
                              busy=busy, agent=agent_name, traveler=traveler_name)


@server.tool()
def confirm_booking(hold_id: str) -> dict:
    """request_plan で仮押さえした旅程を確定する。"""
    with LOCK:
        return Window.open().confirm(hold_id)


@server.tool()
def cancel_booking(hold_id: str) -> dict:
    """仮押さえまたは確定した旅程を取り消し、枠を戻す。"""
    with LOCK:
        return Window.open().cancel(hold_id)


@server.tool()
def reset_demo() -> dict:
    """デモ用: すべての予約を消して窓口を初期状態に戻す。"""
    with LOCK:
        Window.open().reset()
    return {"ok": True, "message": "窓口を初期状態に戻しました"}


if __name__ == "__main__":
    server.run()
