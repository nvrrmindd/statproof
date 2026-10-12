"""Живой ответ на вопрос из командной строки (пересчёт на лету).

    python3 qol_ai/ask.py "Связан ли доход с обрядами?"
    python3 qol_ai/ask.py "..." --llm      # пересказ через Claude (нужен ANTHROPIC_API_KEY и pip install anthropic)

LLM получает только JSON с фактами. Если в её тексте есть число, которого нет в фактах,
ответ отклоняется и выводится детерминированный вердикт движка.
"""
import json
import sys
import engine as E

SYSTEM = ("Ты — аналитик официальной статистики. Перескажи вердикт простым языком для чиновника в 3–4 предложениях. "
          "Используй ТОЛЬКО числа из переданного JSON, не добавляй новых чисел, причин и источников. "
          "Сохрани уровень доказательности (HOUSEHOLD/REGIONAL/NONE) и оговорки.")


def narrate_llm(facts: dict) -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic()
    msg = client.beta.messages.create(
        model="claude-opus-5-5", max_tokens=1024, system=SYSTEM,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
    )
    if msg.stop_reason == "refusal":
        return None
    return "".join(b.text for b in msg.content if b.type == "text")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    q = " ".join(args) or "Связаны ли лишения с удовлетворённостью жизнью?"
    d = E.load()
    r = E.answer(d, q)
    facts = {k: r.get(k) for k in ["question", "labels", "level", "verdict", "caveat", "claims"]}
    print(f"[{r['level']}] {r['verdict']}")
    if r.get("caveat"):
        print("Оговорка:", r["caveat"])
    for c in r["claims"]:
        print(f"  · {c['text']}: {c['value']}  ДИ {c.get('ci')}" + (f"  p={c['p']}" if "p" in c else ""))
    if "--llm" in sys.argv:
        text = narrate_llm(facts)
        if text is None:
            print("\nLLM недоступна — показан детерминированный вердикт.")
        else:
            bad = E.grounding_check(text, facts)
            print("\nПересказ LLM:" if not bad else f"\nПересказ LLM ОТКЛОНЁН: числа без источника {bad}")
            if not bad:
                print(text)


if __name__ == "__main__":
    main()
