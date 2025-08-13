import json
from typing import Any, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


# -------------------------------
# Внешний контракт (английский)
# -------------------------------
class OlegtinkoffSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]  # external interface in English
    confidence: float
    reasoning: str  # Russian text (persona speaks Russian)


# -------------------------------
# Агент Олега Тинькова
# -------------------------------
def oleg_tinkoff_agent(state: AgentState, agent_id: str = "oleg_tinkoff_agent"):
    """
    «Гротескный Тиньков»: агрессивный, дерзкий, ориентирован на рост/бренд/маркетинг.
    Внешний протокол — как у базового агента (английские сигналы).
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data: dict[str, Any] = {}
    tinkoff_analysis: dict[str, Any] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
                "gross_profit",
                "revenue",
                "free_cash_flow",
            ],
            end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # --- Блоки «тиньковского» анализа (коротко, хлестко и полезно) ---
        progress.update_status(agent_id, ticker, "Assessing growth & brand proxies")
        growth = analyze_growth(financial_line_items)
        brand = analyze_brand_power(financial_line_items, metrics)

        progress.update_status(agent_id, ticker, "Assessing profitability vs burn")
        unit_econ = analyze_profitability_vs_burn(financial_line_items)

        progress.update_status(agent_id, ticker, "Assessing leadership boldness")
        bold = analyze_bold_management(financial_line_items)

        progress.update_status(agent_id, ticker, "Quick EV sanity check")
        ev_check = quick_ev_check(financial_line_items, market_cap)

        # Итоговый счёт (условно 40 макс.)
        total_score = growth["score"] + brand["score"] + unit_econ["score"] + bold["score"] + ev_check["score"]
        max_possible_score = growth["max_score"] + brand["max_score"] + unit_econ["max_score"] + bold["max_score"] + ev_check["max_score"]

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "growth": growth,
            "brand": brand,
            "unit_economics": unit_econ,
            "bold_management": bold,
            "ev_check": ev_check,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating tinkoff-style decision")
        tinkoff_output = generate_tinkoff_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        # Сохраняем в унифицированном формате
        tinkoff_analysis[ticker] = {
            "signal": tinkoff_output.signal,
            "confidence": tinkoff_output.confidence,
            "reasoning": tinkoff_output.reasoning,  # Russian text
        }

        progress.update_status(agent_id, ticker, "Done", analysis=tinkoff_output.reasoning)

    # Сообщение для графа
    message = HumanMessage(content=json.dumps(tinkoff_analysis, ensure_ascii=False), name=agent_id)

    # Опционально показываем reasoning
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(tinkoff_analysis, agent_id)

    # Записываем результат в общий контейнер сигналов
    state["data"]["analyst_signals"][agent_id] = tinkoff_analysis
    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}


# -------------------------------
# Аналитические блоки (utility)
# -------------------------------
def _safe_series(items: list, attr: str) -> List[float]:
    vals: List[float] = []
    for it in items:
        v = getattr(it, attr, None)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def analyze_growth(financial_line_items: list) -> dict[str, Any]:
    """
    Рост и «огонь» — главная любовь Тинькова.
    Оцениваем:
      - CAGR выручки (последние периоды)
      - Признаки ускорения (последние шаги > прошлых)
      - Валовая маржа как прокси «кричащего» спроса/бренда
    """
    max_score, score, details = 12, 0, []
    rev = _safe_series(financial_line_items, "revenue")
    gp = _safe_series(financial_line_items, "gross_profit")

    # CAGR выручки
    if len(rev) >= 3 and rev[-1] > 0:
        years = len(rev) - 1
        try:
            cagr = (rev[0] / rev[-1]) ** (1 / years) - 1
        except Exception:
            cagr = 0.0
        details.append(f"CAGR выручки ≈ {cagr:.1%}")
        if cagr > 0.40:
            score += 6
        elif cagr > 0.25:
            score += 5
        elif cagr > 0.15:
            score += 4
        elif cagr > 0.05:
            score += 2
        elif cagr <= 0:
            score -= 1  # фу, падение!

        # Простая проверка ускорения
        if len(rev) >= 5 and rev[2] > 0 and rev[3] > 0:
            try:
                g_now = (rev[0] - rev[1]) / max(rev[1], 1e-9)
                g_prev = (rev[2] - rev[3]) / max(rev[3], 1e-9)
                if g_now - g_prev > 0.05:
                    score += 2
                    details.append("Ускоряемся — это по-нашему!")
            except Exception:
                pass

    # Валовая маржа как прокси любви клиентов/бренда
    if rev and gp and rev[0] > 0:
        gm = gp[0] / rev[0]
        details.append(f"Валовая маржа (посл.) ≈ {gm:.1%}")
        if gm > 0.60:
            score += 4
        elif gm > 0.40:
            score += 3
        elif gm > 0.25:
            score += 2

    return {"score": max(min(score, max_score), -3), "max_score": max_score, "details": "; ".join(details)}


def analyze_brand_power(financial_line_items: list, metrics: list) -> dict[str, Any]:
    """
    Бренд/маркетинг: у Тинькова это святое.
    Прокси:
      - стабильные/высокие валовые маржи
      - улучшение маржи vs прошлые периоды
    """
    max_score, score, details = 8, 0, []
    rev = _safe_series(financial_line_items, "revenue")
    gp = _safe_series(financial_line_items, "gross_profit")
    margins = []
    for i in range(min(len(rev), len(gp))):
        if rev[i] > 0:
            margins.append(gp[i] / rev[i])

    if len(margins) >= 4:
        recent = sum(margins[:2]) / 2
        older = sum(margins[2:4]) / 2
        delta = recent - older
        details.append(f"Дельта маржи ≈ {delta:.1%}")
        if delta > 0.03:
            score += 3
        elif delta > 0.0:
            score += 2

    if margins:
        avg = sum(margins) / len(margins)
        details.append(f"Средняя маржа ≈ {avg:.1%}")
        if avg > 0.50:
            score += 3
        elif avg > 0.35:
            score += 2
        elif avg > 0.25:
            score += 1

    return {"score": min(score, max_score), "max_score": max_score, "details": "; ".join(details)}


def analyze_profitability_vs_burn(financial_line_items: list) -> dict[str, Any]:
    """
    Тиньков терпит убытки, если есть разгон и бренд.
    Но «вечный котёл без дна» — никуда.
    Прокси:
      - Net Income (посл.)
      - FCF маржа (посл.)
    """
    max_score, score, details = 8, 0, []
    ni = _safe_series(financial_line_items, "net_income")
    rev = _safe_series(financial_line_items, "revenue")
    fcf = _safe_series(financial_line_items, "free_cash_flow")

    # Net Income
    if ni:
        if ni[0] > 0:
            score += 3
            details.append("Чистая прибыль: плюс — вкусно")
        else:
            score -= 1
            details.append("Чистая прибыль: минус — терпимо, если есть огонь")

    # FCF margin
    if fcf and rev and rev[0] > 0:
        fcf_m = fcf[0] / rev[0]
        details.append(f"FCF-маржа ≈ {fcf_m:.1%}")
        if fcf_m > 0.08:
            score += 3
        elif fcf_m > 0.03:
            score += 2
        elif fcf_m < -0.10:
            score -= 2

    return {"score": max(min(score, max_score), -3), "max_score": max_score, "details": "; ".join(details)}


def analyze_bold_management(financial_line_items: list) -> dict[str, Any]:
    """
    Дерзость руководства (очень упрощённо):
      - выкуп акций (share repurchase) — ок
      - размывание через частые эмиссии — аккуратнее
      - дивиденды не критичны (Тиньков любит рост)
    """
    max_score, score, details = 6, 0, []
    latest = financial_line_items[0] if financial_line_items else None

    if latest is not None:
        # Выкуп акций (отток денег с отрицательным знаком)
        val = getattr(latest, "issuance_or_purchase_of_equity_shares", None)
        if isinstance(val, (int, float)):
            if val < 0:
                score += 2
                details.append("Байбэки есть — жирный плюс")
            elif val > 0:
                score -= 1
                details.append("Эмиссия — можно, но аккуратно")

        # Дивиденды: нейтрально/слегка отрицательно (лучше вкладывать в рост)
        div = getattr(latest, "dividends_and_other_cash_distributions", None)
        if isinstance(div, (int, float)) and div < 0:
            score -= 1
            details.append("Дивиденды? Лучше вкладывать в разгон")

    return {"score": max(min(score, max_score), -2), "max_score": max_score, "details": "; ".join(details)}


def quick_ev_check(financial_line_items: list, market_cap: float | None) -> dict[str, Any]:
    """
    Простейшая «нюх-проверка» оценки:
      - если market_cap огромная, а FCF крошечный/отрицательный — будьте осторожны
      - если ревеню растёт и FCF близок к нулю/+, — норм
    """
    max_score, score, details = 6, 0, []
    rev = _safe_series(financial_line_items, "revenue")
    fcf = _safe_series(financial_line_items, "free_cash_flow")

    if market_cap is None:
        return {"score": 0, "max_score": max_score, "details": "Нет market cap — пропускаем"}

    details.append(f"Market cap ≈ {market_cap:,.0f}")
    if fcf and rev and rev[0] > 0:
        fcf_m = fcf[0] / rev[0]
        if fcf_m > 0.05:
            score += 3
            details.append("FCF положительный — оценка может быть оправдана")
        elif fcf_m < -0.10 and market_cap > max(rev[0] * 10, 1):
            score -= 3
            details.append("FCF −, а капа огромна — перебор")
    return {"score": max(min(score, max_score), -3), "max_score": max_score, "details": "; ".join(details)}


# -------------------------------
# Генерация ответа (LLM)
# -------------------------------
def generate_tinkoff_output(
    ticker: str,
    analysis_data: dict[str, Any],
    state: AgentState,
    agent_id: str = "oleg_tinkoff_agent",
) -> OlegtinkoffSignal:
    """
    LLM-вывод в стиле Олега Тинькова (русский язык), но с внешним контрактом:
    signal ∈ {bullish, bearish, neutral}
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты — Олег Тиньков. Отвечаешь ТОЛЬКО на русском. 
Стиль — дерзкий, хлёсткий, без воды: короткие фразы, метафоры, жаргон из реального бизнеса.
Ты ориентирован на рост, маркетинг, драйв, харизму собственника и победу в категории.
Готов терпеть убытки, если это разгон и захват внимания. Но «вечный котёл без дна» — в топку.
""",
            ),
            (
                "human",
                """Проанализируй {ticker} по вычисленным данным ниже (последние периоды — первыми):

{analysis_data}

Сформируй ОТВЕТ ТОЛЬКО в JSON (без пояснений снаружи) со строгими ключами на английском:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": число 0..100,
  "reasoning": "Твой разбор на русском, хлестко и по делу"
}}

Правила:
- signal строго из набора: bullish / bearish / neutral (английские слова).
- reasoning — строго на русском языке, стиль «тиньковский»: дерзкий, образный, с акцентом на рост/бренд/маркетинг.
- будь конкретен: укажи, что тянет вверх (рост, маржа, бренд) и что тянет вниз (затяжной burn, слабая динамика).
- никаких английских слов в reasoning.
""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2, ensure_ascii=False), "ticker": ticker})

    def default_signal():
        return OlegtinkoffSignal(signal="neutral", confidence=50.0, reasoning="Данных мало. Спокойно стоим, смотрим на динамику. Если не ускорятся — выходим.")

    return call_llm(
        prompt=prompt,
        pydantic_model=OlegtinkoffSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default_signal,
    )
