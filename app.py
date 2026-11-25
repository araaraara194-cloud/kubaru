"""くばるAI v2 — 札幌の観光の「配分窓口」(AIエージェント向け)

起動:  streamlit run app.py
MCPサーバー(エージェントの入口)は mcp_server.py。両方が同じ予約ファイル(state/bookings.json)を見ます。
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from kubaru.rush import run_rush
from kubaru.viz import heatmap, matrix
from kubaru.window import Window

st.set_page_config(page_title="くばるAI v2 | 札幌の配分窓口", layout="wide")

st.title("くばるAI — 札幌の観光の配分窓口")
st.caption("AIエージェントが旅程を組む時代に、「空いている所を見せる」のではなく「上限を超えないように配る」窓口。"
           "数値はすべてデモ用の仮の値です。")

win = Window.open()
world = win.world
AREAS = [ar.name for ar in world.areas.values()]
SLOTS = [world.slot_names[s] for s in world.slot_ids]


def window_rows(w: Window, days: list[int]) -> pd.DataFrame:
    rows = []
    for d in days:
        for s in w.world.slot_ids:
            for a, ar in w.world.areas.items():
                cap = w.world.cap(a, s)
                if cap <= 0:
                    continue
                total = w.occupied(d, s, a)
                rows.append({"day": d, "slot_name": w.world.slot_names[s], "area_name": ar.name,
                             "total": total, "cap": cap, "ratio": total / cap})
    return pd.DataFrame(rows)


tab_rush, tab_try, tab_state, tab_rules = st.tabs(
    ["① エージェント殺到テスト", "② 窓口に頼んでみる", "③ 窓口の状態", "④ ルールとつなぎ方"])

# ------------------------------------------------------------------ ① 殺到テスト
with tab_rush:
    st.subheader("同じ時刻に、200人分の代理AIが札幌の旅程を組んだら")
    st.markdown("**A. 混雑マップを読むAI**: 全員が同じ混雑情報を見て「空いている所」を選ぶ → 同じ所に集まる  \n"
                "**B. くばる窓口に聞くAI**: 窓口が1件ずつ配る → 埋まった枠は次の人に出ない")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("代理AIの数", 50, 400, 200, step=50)
    seed = c2.number_input("乱数のたね", 0, 9999, 0)
    day_pick = c3.selectbox("表示する日", ["3日間の最大", "1日目", "2日目", "3日目"])
    st.markdown("**現実のずれ**(案内型の場所は、窓口が人数を管理できない)")
    c4, c5, c6 = st.columns(3)
    follow = c4.slider("案内に従う人の割合", 0.3, 1.0, 0.7, step=0.1,
                       help="従わない人は、混雑マップを見て自分で選んだ所へ行く。予約型の体験は席を押さえているので守る想定。")
    err = c5.slider("ふだんの人出の予測のはずれ(±)", 0.0, 0.3, 0.15, step=0.05,
                    help="実際の人出 = 予測 × (1 + はずれ)。はずれは毎コマばらばらに起きる。")
    margin = c6.slider("案内型の安全の余裕(上限の何割を残すか)", 0.0, 0.4, 0.2, step=0.05)

    @st.cache_data(show_spinner="代理AIの旅程を組んでいます…")
    def rush(n, seed, follow, err, margin):
        return run_rush(n=n, days=[1, 2, 3], seed=seed, follow=follow, forecast_err=err, margin=margin)

    res = rush(n, seed, follow, err, margin)
    day = None if day_pick.startswith("3日間") else int(day_pick[0])
    sa, sb = res["map"]["summary"], res["kubaru"]["summary"]

    m = st.columns(4)
    m[0].metric("上限を超えたコマ", f'{sb["上限を超えたコマ数"]}', delta=f'{sb["上限を超えたコマ数"] - sa["上限を超えたコマ数"]}(Aとの差)',
                delta_color="inverse")
    m[1].metric("上限を超えた人数(延べ)", f'{sb["上限を超えた人数(延べ)"]}人',
                delta=f'{sb["上限を超えた人数(延べ)"] - sa["上限を超えた人数(延べ)"]}人', delta_color="inverse")
    m[2].metric("旭山の最大混雑率", f'{sb["旭山の最大混雑率"]:.0%}',
                delta=f'{(sb["旭山の最大混雑率"] - sa["旭山の最大混雑率"]) * 100:.0f}pt', delta_color="inverse")
    m[3].metric("好みとの一致(平均点)", f'{sb["好みとの一致(平均点)"]:.2f}',
                delta=f'{sb["好みとの一致(平均点)"] - sa["好みとの一致(平均点)"]:+.2f}')
    st.caption("数値はB(くばる窓口)で、案内に従わない人と予測のはずれを入れた後の「実際の人数」。矢印はA(混雑マップ)との差。"
               f'Bで案内に従わなかったのは{res["kubaru"]["案内に従わなかった件数"]}件。安全の余裕を0にすると、ずれで上限を超えるコマが出ることがあります。')

    left, right = st.columns(2)
    with left:
        st.markdown("#### A. 混雑マップを読むAI")
        r, t = matrix(pd.DataFrame(res["map"]["rows"]), AREAS, SLOTS, day)
        st.plotly_chart(heatmap(r, t), width="stretch", key="hm_a")
    with right:
        st.markdown("#### B. くばる窓口に聞くAI")
        r, t = matrix(pd.DataFrame(res["kubaru"]["rows"]), AREAS, SLOTS, day)
        st.plotly_chart(heatmap(r, t), width="stretch", key="hm_b")
    st.caption("色が濃いほど上限に近い。▲と赤い枠は上限超え。「休」はその時間帯は受け入れなし。")

    with st.expander("数字の一覧"):
        st.dataframe(pd.DataFrame({"A. 混雑マップ": sa, "B. くばる窓口": sb}).astype(str), width="stretch")

# ------------------------------------------------------------------ ② 窓口に頼んでみる
with tab_try:
    st.subheader("AIエージェントの代わりに、窓口へ頼んでみる")
    st.caption("実際のエージェント(Claudeなど)は MCP の request_plan で同じことをします。")
    with st.form("req"):
        text = st.text_area("旅行者の希望(自由文・英語もOK)", "海鮮が好き。人混みは苦手で、静かに過ごしたい。")
        c1, c2, c3 = st.columns(3)
        days = c1.multiselect("滞在する日", list(range(1, 8)), default=[1, 2])
        party = c2.number_input("人数", 1, 10, 2)
        per_day = c3.slider("1日に入れる体験", 1, 3, 2)
        first = st.checkbox("札幌は初めて", value=True)
        go = st.form_submit_button("旅程を配ってもらう", type="primary")
    if go:
        st.session_state["plan"] = Window.open().request_plan(
            request=text, days=sorted(days) or None, party=int(party), per_day=per_day,
            first_time=first, agent="streamlit-demo")
    plan = st.session_state.get("plan")
    if plan:
        st.markdown(f'**読み取った興味**: {"・".join(plan["読み取った興味"]) or "(特になし)"} / '
                    f'**スタイル**: {plan["スタイル"]} / **仮押さえ** `{plan["hold_id"]}`(15分で失効)')
        items = [i for i in plan["旅程"] if i["type"] == "体験"]
        if not items:
            st.warning("配れる枠が見つかりませんでした(上限に達しているかもしれません)。")
        for i in items:
            st.markdown(f'**{i["day"]}日目 {i["slot_name"]}** — {i["name"]}({i["area"]}) `{i["枠の種類"]}`  \n'
                        f'<span style="color:gray">{i["理由"]} / この枠の混雑率(配分後) {i["この枠の混雑率(配分後)"]:.0%}</span>',
                        unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("この旅程で確定", type="primary"):
            st.success(Window.open().confirm(plan["hold_id"])["message"])
        if b2.button("取り消す"):
            st.info(Window.open().cancel(plan["hold_id"])["message"])
            st.session_state.pop("plan")

# ------------------------------------------------------------------ ③ 窓口の状態
with tab_state:
    st.subheader("いま窓口が配っている枠")
    w = Window.open()
    c1, c2 = st.columns([1, 3])
    d = c1.selectbox("日", list(range(1, 8)), format_func=lambda x: f"{x}日目")
    if c1.button("更新"):
        st.rerun()
    if c1.button("デモをリセット(予約を全部消す)"):
        w.reset()
        st.session_state.pop("plan", None)
        st.rerun()
    with c2:
        r, t = matrix(window_rows(w, [d]), AREAS, SLOTS, d)
        st.plotly_chart(heatmap(r, t), width="stretch", key="hm_state")
        st.caption("ふだんの人出(仮定)+窓口が配った人数を、受け入れ上限で割った値。")
    live = [b for b in w.bookings if b.status in ("held", "confirmed")]
    st.markdown(f"#### 予約ログ({len(live)}件が有効)")
    if w.bookings:
        log = pd.DataFrame([{
            "時刻": time.strftime("%H:%M:%S", time.localtime(b.created)), "エージェント": b.agent,
            "旅行者": b.traveler, "日": b.day, "時間帯": world.slot_names[b.slot], "体験": b.name,
            "枠": world.packages[b.package].kind + "型", "人数": b.party, "状態": {"held": "仮押さえ", "confirmed": "確定(案内型は案内済み)", "cancelled": "取消",
                                    "expired": "失効"}.get(b.status, b.status), "hold_id": b.hold_id,
        } for b in sorted(w.bookings, key=lambda b: -b.created)])
        st.dataframe(log.astype(str), width="stretch", hide_index=True)
    else:
        st.caption("まだ予約はありません。②のタブか、MCPでつないだAIエージェントから頼んでみてください。")

# ------------------------------------------------------------------ ④ ルール
with tab_rules:
    st.subheader("窓口のルール(エージェントにも get_rules で同じものを返します)")
    st.json(win.rules())
    st.subheader("AIエージェントからつなぐ")
    st.markdown("Claude Code なら:")
    st.code("claude mcp add kubaru -- python /path/to/kubaru_v2/mcp_server.py", language="bash")
    st.markdown("Claude Desktop なら設定ファイル(claude_desktop_config.json)に:")
    st.code('{"mcpServers": {"kubaru": {"command": "python", "args": ["/path/to/kubaru_v2/mcp_server.py"]}}}',
            language="json")
    st.markdown("使えるツール: `get_rules` `list_experiences` `get_availability` `request_plan` "
                "`confirm_booking` `cancel_booking` `reset_demo`")
    st.caption("将来は、AIが予約・購入まで行う共通規格(GoogleのUCPなど)にも同じ窓口を出す想定です。")
