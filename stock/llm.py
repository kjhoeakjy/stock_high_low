"""Claude로 후보 종목의 밤사이 뉴스·공시 제목을 읽어 호재·악재 판단과 한 줄 근거를 붙인다.

ANTHROPIC_API_KEY 가 없으면 건너뛴다. 모델 확률은 바꾸지 않고, 화면에 '뉴스 판독' 항목만 추가한다.
"""
from __future__ import annotations

import json
import os
import re

MODEL = os.environ.get("STOCK_LLM_MODEL", "claude-sonnet-5-5")
BETAS = ["server-side-fallback-2026-07-01"]

SYSTEM = """당신은 한국 주식시장 뉴스·공시 판독기다. 각 종목의 최근 뉴스·공시 제목만 보고 다음 거래일 주가에 미칠 방향을 판단한다.
규칙: 제목에 없는 사실을 지어내지 않는다. 투자 권유 표현 금지. 문어체, 한 문장, 40자 이내.
출력은 JSON 배열만. 각 원소: {"code": 종목코드, "verdict": "호재"|"악재"|"중립"|"정보없음", "strength": 0~3 정수, "note": 한 문장 근거, "event": 핵심 사건 분류(예: 공급계약, 무상증자, 유상증자, 최대주주변경, 임상, 실적, 테마, 거래정지, 횡령, 없음)}"""


def _client():
    try:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        return anthropic.Anthropic()
    except Exception:  # noqa: BLE001
        return None


def annotate(items: list[dict], side: str) -> None:
    cli = _client()
    if cli is None or not items:
        return
    blocks = []
    for it in items:
        ev = it.get("evidence") or {}
        lines = [f'- 공시 {d["t"]} {d["title"]}' for d in ev.get("disclosures", [])[:6]]
        lines += [f'- 뉴스 {n["t"]} {n["title"]}' for n in ev.get("news", [])[:6]]
        blocks.append(f'[{it["code"]}] {it["name"]} (전일 {it["ret1"]*100:+.1f}%, 모델 {"상한가" if side=="up" else "하한가"} 확률 {it["p"]*100:.0f}%)\n'
                      + ("\n".join(lines) if lines else "- 최근 뉴스·공시 없음"))
    prompt = f"다음 {len(items)}개 종목을 판독하라. 판독 대상은 {'상한가' if side=='up' else '하한가'} 후보다.\n\n" + "\n\n".join(blocks)
    try:
        with cli.beta.messages.stream(model=MODEL, max_tokens=4000, system=SYSTEM, betas=BETAS, fallbacks="default",
                                      messages=[{"role": "user", "content": prompt}]) as s:
            text = s.get_final_message().content[0].text
    except Exception as e:  # noqa: BLE001
        try:
            r = cli.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM,
                                    messages=[{"role": "user", "content": prompt}])
            text = r.content[0].text
        except Exception as e2:  # noqa: BLE001
            print(f"[llm] fail: {e} / {e2}")
            return
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        print("[llm] no json")
        return
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        print("[llm] bad json")
        return
    by = {str(a.get("code")): a for a in arr if isinstance(a, dict)}
    for it in items:
        a = by.get(it["code"])
        if a:
            it["llm"] = {"verdict": a.get("verdict", "정보없음"), "strength": int(a.get("strength") or 0),
                         "note": str(a.get("note") or "")[:80], "event": str(a.get("event") or "")[:20]}
    print(f"[llm] {side}: {sum(1 for it in items if 'llm' in it)}/{len(items)} annotated")
