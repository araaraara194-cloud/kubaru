"""旅行者の文章から日数と人数を読む(「2泊3日」「ふたり」「家族4人」など)。"""
from __future__ import annotations

import re

KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
             "ひとり": 1, "ふたり": 2, "1人": 1}


def _num(tok: str) -> int | None:
    tok = tok.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if tok.isdigit():
        return int(tok)
    return KANJI_NUM.get(tok)


def read_trip(text: str) -> tuple[int | None, int | None]:
    """自由文から日数と人数を読む。例: 2泊3日→3日、3日間→3日、日帰り→1日、2人・ふたり・夫婦→2人、家族4人→4人"""
    t = text or ""
    days = party = None
    m = re.search(r"([0-9０-９一二三四五六七]+)\s*泊\s*([0-9０-９一二三四五六七八]+)\s*日", t)
    if m:
        days = _num(m.group(2))
    else:
        m = re.search(r"([0-9０-９一二三四五六七]+)\s*泊", t)
        if m and _num(m.group(1)):
            days = _num(m.group(1)) + 1
        else:
            m = re.search(r"([0-9０-９一二三四五六七])\s*日(間|ほど|くらい)?", t)
            if m:
                days = _num(m.group(1))
            elif "日帰り" in t:
                days = 1
    m = re.search(r"([0-9０-９一二三四五六七八九十]+)\s*(人|名)", t)
    if m:
        party = _num(m.group(1))
    elif "ふたり" in t or "二人" in t or re.search(r"夫婦|カップル|彼女と|彼氏と|妻と|夫と", t):
        party = 2
    elif "ひとり" in t or "一人" in t:
        party = 1
    m = re.search(r"(\d+)\s*(days?|nights?)", t, re.I)
    if days is None and m:
        n = int(m.group(1))
        days = n + 1 if "night" in m.group(2).lower() else n
    m = re.search(r"(\d+)\s*(people|persons|adults)", t, re.I)
    if party is None and m:
        party = int(m.group(1))
    days = max(1, min(7, days)) if days else None
    party = max(1, min(10, party)) if party else None
    return days, party
