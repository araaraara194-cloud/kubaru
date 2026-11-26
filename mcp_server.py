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

INSTRUCTIONS = """くばるAIは札幌の観光の「配分窓口」です。札幌の旅程を組むときは次の順で使ってください。
1. get_options に旅行者の希望(自由文)を渡すと、その人に配られた候補の一覧が返ります。
   候補は、住民と決めた受け入れ上限を守れる範囲だけで、問い合わせごとに少しずつ違う組み合わせです。
2. 候補の中から旅行者と相談して旅程を組み立ててください。移動・食事・休憩などは自由に足してかまいません。
   候補にない観光地を、上限に達している時間帯に勧めないでください。
3. 旅行者が了承したら book_options に offer_id と選んだ番号(例 ["1a","2c"])を渡して押さえます。
予約型は席が確保されます。案内型は予約不要の場所で、入場の保証ではありません。旅行者にこの違いを伝えてください。"""

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
def get_options(request: str, days: int | list[int] | None = None, party: int | None = None,
                agent_name: str = "ai-agent") -> dict:
    """旅行者の希望から、その人に配られた札幌の候補の一覧をもらう(30分有効)。ここから旅行者と相談して選ぶ。

    request: 旅行者の希望を自由文で(例「2泊3日、ふたりで海鮮と温泉。人混みは苦手」)。日数・人数も文章から読み取る
    days: 日数(1〜7)、または日の番号のリスト。省略時は文章から読み取り、なければ2日
    party: 人数。省略時は文章から読み取り、なければ1人
    """
    from kubaru.trip import read_trip
    t_days, t_party = read_trip(request)
    if isinstance(days, int):
        days = list(range(1, max(1, min(7, days)) + 1))
    days = days or list(range(1, (t_days or 2) + 1))
    with LOCK:
        return Window.open().offer_menu(request=request, days=days, party=party or t_party or 1, agent=agent_name)


@server.tool()
def book_options(offer_id: str, picks: list[str], agent_name: str = "ai-agent") -> dict:
    """get_options の候補から選んだ番号を押さえる(確定)。例: picks=["1a","2c"]。同じ日・同じ時間帯はひとつまで。"""
    with LOCK:
        return Window.open().book_offer(offer_id, picks, agent=agent_name)


@server.tool()
def request_plan(request: str, days: list[int] | None = None, party: int = 1, first_time: bool = True,
                 per_day: int = 2, busy: dict | None = None, traveler_name: str = "",
                 agent_name: str = "ai-agent") -> dict:
    """(おまかせ) 旅行者の希望から、くばるAIが旅程を1つ組んで仮押さえする(15分で失効)。ふつうは get_options を使う。

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
