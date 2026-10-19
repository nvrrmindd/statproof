"""Понимание текста облачной языковой моделью (бесплатные лимиты Gemini и Groq).

Модель только переводит вопрос или утверждение в структуру: какие показатели каталога упомянуты
(с дословной цитатой), направление связи, регионы или семьи, есть ли отрицание.
Чисел она не считает и не видит данных: в модель уходит только текст вопроса.
Ответ проверяется: показатель вне каталога или цитата, которой нет в тексте, отбрасываются.

Ключи читаются из qol_ai/.env (файл не хранится в git):
    GEMINI_API_KEY=...        # aistudio.google.com, бесплатно
    GROQ_API_KEY=...          # console.groq.com, бесплатно
    LLM_MODEL_EXTRACT=gemini:<модель>,groq:<модель>   # цепочка: при исчерпании лимита берётся следующая
"""
from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import engine as E

HERE = Path(__file__).resolve().parent
CACHE = HERE / "eval" / "llm_cache_api.json"
BASE = {"gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "groq": "https://api.groq.com/openai/v1"}
KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
try:  # Python с python.org на macOS не видит системные сертификаты
    import certifi
    SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL = ssl.create_default_context()


def _env() -> dict:
    env = dict(os.environ)
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.split("#")[0].strip())
    return env


def chain() -> list[tuple[str, str, str]]:
    env = _env()
    spec = env.get("LLM_MODEL_EXTRACT", "")
    out = []
    for item in [x.strip() for x in spec.split(",") if ":" in x]:
        prov, model = item.split(":", 1)
        if prov == "llama":   # собственный сервер llama.cpp (например, через Tailscale): LLAMA_BASE_URL=http://100.x.x.x:8080/v1
            if env.get("LLAMA_BASE_URL"):
                BASE["llama"] = env["LLAMA_BASE_URL"].rstrip("/")
                out.append((prov, model, env.get("LLAMA_API_KEY", "none")))
            continue
        key = env.get(KEYS.get(prov, ""), "")
        if prov in BASE and key:
            out.append((prov, model, key))
    return out


HINTS = {
    "income_pc": "доход, зарплата, заработок, деньги на человека; каз. табыс, жалақы",
    "spend_pc": "расходы, траты; каз. шығыс",
    "borrowed": "кредиты, займы, ипотека, долги, долговая нагрузка; каз. несие, қарыз",
    "inc_social": "пособия, АСП, социальные выплаты, помощь государства; каз. жәрдемақы",
    "dep_n": "лишения, бедность, нужда; каз. кедейлік, тапшылық",
    "dep_shock": "не покрыть непредвиденный расход, нет финансовой подушки",
    "dep_ritual": "не может провести той, свадьбу, похороны без долгов",
    "dep_holiday": "не может позволить отпуск или отдых",
    "dep_heat": "не может поддерживать тепло, холод дома, отопление",
    "life_sat": "удовлетворённость жизнью, счастье, довольны жизнью; каз. өмірге қанағаттану",
    "fin_sat": "удовлетворённость финансами, материальное положение, финансовое самочувствие",
    "self_class": "самооценка достатка, ощущение достатка, считают себя средним классом",
    "optimism": "ожидания на будущее, оптимизм, пессимизм, уверенность в завтрашнем дне",
    "fies_raw": "голод, недоедание, нехватка еды; каз. аштық",
    "hh_size": "размер семьи, многодетные, число членов семьи, едоков",
    "n_children": "число детей; каз. балалар",
    "n_elderly": "пожилые, пенсионеры, старики",
    "n_employed": "число работающих, занятость, безработица (как отсутствие работающих)",
    "any_higher_edu": "высшее образование, диплом вуза",
    "head_female": "глава семьи — женщина",
    "total_area": "площадь жилья, квадратные метры, метраж, теснота",
    "amenities_missing": "нет удобств: водопровода, канализации",
    "urban": "город или село, горожане, сельчане, аул; каз. қала, ауыл",
}

SYSTEM = (
    "Ты разбираешь вопросы и утверждения о статистике домохозяйств Казахстана на русском или казахском. "
    "Верни только JSON вида {\"vars\":[{\"key\":\"…\",\"quote\":\"…\"}],\"direction\":\"up|down|none\",\"regional\":true|false,\"negated\":true|false}.\n"
    "vars — не больше двух показателей из каталога, которые упомянуты в тексте, в порядке упоминания. "
    "quote — слово или фраза ДОСЛОВНО из текста, по которой ты узнал показатель. "
    "Если показателя из каталога в тексте нет или он не подходит по смыслу (например, возраст, погода, цены), не указывай его. "
    "direction: up — утверждается прямая связь (чем больше одного, тем больше другого), down — обратная, none — направление не указано или это вопрос. "
    "Учитывай смысл: «сытые» — это меньше голода, «сельчане» — нижний конец шкалы «город/село», «теснее» — меньше площадь. "
    "regional: true, если речь о регионах или областях, а не о семьях и людях. negated: true, если утверждается, что связи нет.\n\n"
    "Пример: «Жители аулов реже имеют канализацию» → {\"vars\":[{\"key\":\"urban\",\"quote\":\"аулов\"},{\"key\":\"amenities_missing\",\"quote\":\"канализацию\"}],\"direction\":\"up\",\"regional\":false,\"negated\":false}\n"
    "(реже имеют канализацию = больше отсутствующих удобств; аул = нижний конец шкалы город/село, поэтому связь с «город» обратная, но направление указывай по словам текста о первом показателе: здесь это аулы → больше нет удобств → up)\n"
    "Пример: «Сколько стоит проезд?» → {\"vars\":[],\"direction\":\"none\",\"regional\":false,\"negated\":false}\n\n"
    "Каталог (ключ: смысл):\n" + "\n".join(f"{k}: {E.CATALOG[k][0]} ({HINTS[k]})" for k in E.CATALOG)
)


EXPLAIN_SYSTEM = {
    "ru": "Ты аналитик официальной статистики. Объясни вердикт простыми словами для руководителя в 2–3 предложениях на русском языке. "
          "Используй ТОЛЬКО числа из переданного JSON, не добавляй новых чисел, процентов, причин и источников. "
          "Сохрани смысл вердикта: о семьях, только о регионах или не подтверждается. Без вступлений и заголовков.",
    "kk": "Сен ресми статистика талдаушысысың. Шешімді басшыға арнап қазақ тілінде 2–3 сөйлеммен қарапайым түсіндір. "
          "Тек берілген JSON-дағы сандарды қолдан, жаңа сандар, пайыздар, себептер мен дереккөздер қоспа. "
          "Шешімнің мағынасын сақта: отбасылар туралы, тек өңірлер туралы немесе расталмайды. Кіріспесіз.",
}


def explain(facts: dict, lang: str = "ru") -> dict:
    """Пересказ вердикта моделью. Каждое число в тексте сверяется с фактами; при расхождении текст блокируется."""
    sysmsg = EXPLAIN_SYSTEM.get(lang, EXPLAIN_SYSTEM["ru"])
    user = json.dumps(facts, ensure_ascii=False)
    for prov, model, key in chain():
        body = {"model": model, "temperature": 0, "max_tokens": 400,
                "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]}
        if prov == "gemini":
            body["reasoning_effort"] = "low"
            body["max_tokens"] = 400 + 2048
        try:
            req = urllib.request.Request(BASE[prov] + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=30, context=SSL) as r:
                text = json.load(r)["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
        bad = E.grounding_check(text, facts)
        return {"text": text, "blocked": bool(bad), "bad": bad, "model": f"{prov}:{model}", "lang": lang}
    return {"text": None, "blocked": True, "bad": [], "model": None, "lang": lang}


def _call(prov: str, model: str, key: str, text: str) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": 400, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    if prov == "gemini":
        body["reasoning_effort"] = "low"
        body["max_tokens"] = 400 + 2048
    req = urllib.request.Request(BASE[prov] + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30, context=SSL) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0) if m else content)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip(" «»\"'.,?!:;—-")


def understand(text: str, use_cache: bool = True) -> dict | None:
    """Структура смысла текста или None, если ни одна модель цепочки не ответила."""
    cache = json.loads(CACHE.read_text()) if use_cache and CACHE.exists() else {}
    if text not in cache:
        raw, used = None, None
        for prov, model, key in chain():
            for attempt in range(2):
                try:
                    raw, used = _call(prov, model, key, text), f"{prov}:{model}"
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 503) and attempt == 0:
                        time.sleep(2)
                        continue
                    break
                except Exception:
                    break
            if raw is not None:
                break
        if raw is None:
            return None
        cache[text] = {"raw": raw, "model": used}
        if use_cache:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    raw, model = cache[text]["raw"], cache[text]["model"]
    low = _norm(text)
    vars_ = []
    for item in raw.get("vars", []) if isinstance(raw.get("vars"), list) else []:
        k, q = item.get("key"), _norm(str(item.get("quote", "")))
        if k in E.CATALOG and len(q) >= 3 and q in low and k not in vars_:   # проверка: каталог + дословная цитата
            vars_.append(k)
    d = {"up": 1, "down": -1}.get(raw.get("direction"), 0)
    return {"vars": vars_[:2], "direction": d, "regional": bool(raw.get("regional")),
            "negated": bool(raw.get("negated")), "source": "llm", "model": model}


if __name__ == "__main__":
    import sys
    print([f"{p}:{m}" for p, m, _ in chain()])
    for t in sys.argv[1:] or ["Зарабатывают ли горожане больше сельчан?"]:
        print(t, "→", understand(t, use_cache=False))
