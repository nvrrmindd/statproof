"""Понимание вопроса или утверждения локальной языковой моделью (Ollama).

Модель только переводит текст в структуру: какие показатели из каталога упомянуты,
какое направление связи утверждается, о регионах ли речь, есть ли отрицание.
Чисел она не считает и не пишет: все оценки по-прежнему даёт статистический движок.
Ответ модели проверяется: показатель вне каталога отбрасывается.

Модель работает на этом компьютере, данные и вопросы никуда не отправляются.
"""
from __future__ import annotations
import hashlib
import json
import urllib.request
from pathlib import Path

import engine as E

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b"
CACHE = Path(__file__).resolve().parent / "eval" / "llm_cache.json"

SCHEMA = {
    "type": "object",
    "properties": {
        "vars": {"type": "array", "maxItems": 2, "items": {"type": "object", "properties": {
            "key": {"type": "string", "enum": list(E.CATALOG)},
            "quote": {"type": "string"}}, "required": ["key", "quote"]}},
        "direction": {"type": "string", "enum": ["up", "down", "none"]},
        "regional": {"type": "boolean"},
        "negated": {"type": "boolean"},
    },
    "required": ["vars", "direction", "regional", "negated"],
}

HINTS = {
    "income_pc": "доход, зарплата, деньги, богатство, бедность по доходу",
    "spend_pc": "расходы, траты, потребление",
    "borrowed": "кредиты, займы, ипотека, долги, занимать деньги",
    "inc_social": "пособия, АСП, социальные выплаты",
    "dep_n": "лишения, депривация, бедность, нужда, нехватка",
    "dep_shock": "не покрыть непредвиденный расход, нет финансовой подушки",
    "dep_ritual": "не может провести той, свадьбу, похороны, обряды без долгов",
    "dep_holiday": "не может позволить отпуск или отдых вне дома",
    "dep_heat": "не может поддерживать тепло, отопление",
    "life_sat": "удовлетворённость жизнью, счастье, довольны жизнью",
    "fin_sat": "удовлетворённость финансами, материальное положение",
    "self_class": "самооценка достатка, считают себя средним классом",
    "optimism": "ожидания на будущее, оптимизм, пессимизм",
    "fies_raw": "голод, недоедание, нехватка еды, продовольственная небезопасность",
    "hh_size": "размер семьи, многодетные, число членов, большие семьи",
    "n_children": "число детей",
    "n_elderly": "пожилые, пенсионеры, старше 63",
    "n_employed": "число работающих, занятость",
    "any_higher_edu": "высшее образование, диплом вуза",
    "head_female": "глава семьи женщина",
    "total_area": "площадь жилья, квадратные метры, размер квартиры или дома",
    "amenities_missing": "нет удобств, водопровода, канализации",
    "urban": "город или село, горожане, сельские и деревенские жители",
}

SYSTEM = (
    "Ты разбираешь вопросы и утверждения о статистике домохозяйств Казахстана. "
    "Верни JSON. vars: ключи показателей из каталога, которые упомянуты в тексте (не больше двух, в порядке упоминания); "
    "пустой список, если текст не о данных из каталога. "
    "direction: up, если утверждается прямая связь (чем больше одно, тем больше другое), down, если обратная, none, если направление не указано. "
    "regional: true, если речь о регионах или областях, а не о семьях и людях; «внутри регионов» означает false. "
    "negated: true, если утверждается, что связи нет. "
    "Для каждого показателя в quote дословно процитируй слово или фразу из текста, по которой ты его узнал. "
    "Если такого слова в тексте нет, показатель не указывай. Ничего не придумывай.\n\n"
    "Примеры:\n"
    "«Жители аулов реже имеют канализацию?» → vars: urban (аулов), amenities_missing (канализацию); direction: down\n"
    "«Сколько стоит проезд в метро?» → vars: []\n"
    "«В областях с низкими заработками больше нуждающихся» → vars: income_pc (заработками), dep_n (нуждающихся); direction: down; regional: true\n"
    "«Квартиры горожан и сельчан по метражу» → vars: urban (горожан), total_area (метражу)\n"
    "«Иждивенцы-дети и потребительские расходы не связаны» → vars: n_children (дети), spend_pc (потребительские расходы); negated: true\n\nКаталог:\n"
    + "\n".join(f"{k}: {E.CATALOG[k][0]} ({HINTS[k]})" for k in E.CATALOG)
)
PROMPT_ID = hashlib.sha1((SYSTEM + MODEL + json.dumps(SCHEMA)).encode()).hexdigest()[:10]


def available(timeout=2) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=timeout) as r:
            return any(m["name"].startswith(MODEL) for m in json.load(r)["models"])
    except Exception:
        return False


def _ask(text: str) -> dict:
    body = {"model": MODEL, "stream": False, "format": SCHEMA, "options": {"temperature": 0, "seed": 7},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(json.load(r)["message"]["content"])


def understand(text: str, use_cache=True) -> dict | None:
    """Структура смысла текста или None, если модель недоступна или ответила не по схеме."""
    cache = json.loads(CACHE.read_text()) if use_cache and CACHE.exists() else {}
    key = PROMPT_ID + "|" + text
    if key not in cache:
        try:
            raw = _ask(text)
        except Exception:
            return None
        cache[key] = raw
        if use_cache:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    raw = cache[key]
    # проверка ответа модели: показатель из каталога и дословная цитата из текста (не короче 3 букв)
    low = text.lower().replace("ё", "е")
    vars_ = []
    for item in raw.get("vars", []):
        k, q = item.get("key"), str(item.get("quote", "")).lower().replace("ё", "е").strip(" «»\"'.,?!")
        if k in E.CATALOG and len(q) >= 3 and q in low and k not in vars_:
            vars_.append(k)
    vars_ = vars_[:2]
    d = {"up": 1, "down": -1}.get(raw.get("direction"), 0)
    return {"vars": vars_, "direction": d, "regional": bool(raw.get("regional")), "negated": bool(raw.get("negated")), "source": "llm"}
