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


# ---------------------------------
# External interface model
# ---------------------------------
class IngvarKampradSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str  # Will be written in English, Kamprad-style


# ---------------------------------
# Main agent
# ---------------------------------
def ingvar_kamprad_agent(state: AgentState, agent_id: str = "ingvar_kamprad_agent"):
    """
    Analyzes stocks in the style of Ingvar Kamprad (IKEA founder):
    - Frugality, simplicity, cost efficiency
    - Preference for scalable, well-run, low-overhead businesses
    - Conservative valuation assumptions
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data: dict[str, Any] = {}
    kamprad_analysis: dict[str, Any] = {}

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

        # --- Kamprad-style analysis blocks ---
        cost_eff = analyze_cost_efficiency(financial_line_items, metrics)
        scale_ops = analyze_scale_operations(financial_line_items)
        debt_discipline = analyze_debt_discipline(metrics)
        steady_growth = analyze_steady_growth(financial_line_items)
        valuation = conservative_intrinsic_value(financial_line_items, discount_rate=0.11, terminal_growth=0.02)

        total_score = cost_eff["score"] + scale_ops["score"] + debt_discipline["score"] + steady_growth["score"]
        max_possible_score = cost_eff["max_score"] + scale_ops["max_score"] + debt_discipline["max_score"] + steady_growth["max_score"]

        margin_of_safety = None
        if valuation.get("intrinsic_value") and market_cap:
            try:
                margin_of_safety = (valuation["intrinsic_value"] - market_cap) / market_cap
            except ZeroDivisionError:
                margin_of_safety = None

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "cost_efficiency": cost_eff,
            "scale_operations": scale_ops,
            "debt_discipline": debt_discipline,
            "steady_growth": steady_growth,
            "valuation": valuation,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status(agent_id, ticker, "Generating Kamprad-style decision")
        kamprad_output = generate_kamprad_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        kamprad_analysis[ticker] = {
            "signal": kamprad_output.signal,
            "confidence": kamprad_output.confidence,
            "reasoning": kamprad_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=kamprad_output.reasoning)

    message = HumanMessage(content=json.dumps(kamprad_analysis, ensure_ascii=False), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(kamprad_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = kamprad_analysis
    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}


# ---------------------------------
# Analysis blocks
# ---------------------------------
def _safe_series(items: list, attr: str) -> List[float]:
    vals: List[float] = []
    for it in items:
        v = getattr(it, attr, None)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def analyze_cost_efficiency(financial_line_items: list, metrics: list) -> dict[str, Any]:
    """Cost efficiency via margin analysis and implied overhead."""
    max_score, score, details = 10, 0, []
    rev = _safe_series(financial_line_items, "revenue")
    gp = _safe_series(financial_line_items, "gross_profit")

    if rev and gp and rev[0] > 0:
        gross_margin = gp[0] / rev[0]
        details.append(f"Gross margin ≈ {gross_margin:.1%}")
        if 0.25 <= gross_margin <= 0.45:
            score += 4
            details.append("Efficient pricing without luxury markups")
        elif gross_margin > 0.45:
            score += 2
            details.append("High margin; ensure not from overpricing")
        elif gross_margin < 0.25:
            score -= 1
            details.append("Thin margin; may lack cost buffer")

    if metrics and getattr(metrics[0], "operating_margin", None) is not None:
        op_margin = metrics[0].operating_margin
        details.append(f"Operating margin ≈ {op_margin:.1%}")
        if op_margin > 0.15:
            score += 3
        elif op_margin > 0.08:
            score += 2
        else:
            score += 0

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_scale_operations(financial_line_items: list) -> dict[str, Any]:
    """Economies of scale via asset turnover and FCF margin."""
    max_score, score, details = 8, 0, []
    rev = _safe_series(financial_line_items, "revenue")
    assets = _safe_series(financial_line_items, "total_assets")
    fcf = _safe_series(financial_line_items, "free_cash_flow")

    if rev and assets and assets[0] > 0:
        asset_turnover = rev[0] / assets[0]
        details.append(f"Asset turnover ≈ {asset_turnover:.2f}")
        if asset_turnover > 1.5:
            score += 4
        elif asset_turnover > 1.0:
            score += 3

    if fcf and rev and rev[0] > 0:
        fcf_margin = fcf[0] / rev[0]
        details.append(f"FCF margin ≈ {fcf_margin:.1%}")
        if fcf_margin > 0.08:
            score += 3
        elif fcf_margin > 0.03:
            score += 2

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_debt_discipline(metrics: list) -> dict[str, Any]:
    """Debt discipline via debt/equity and current ratio."""
    max_score, score, details = 6, 0, []
    if metrics:
        d2e = getattr(metrics[0], "debt_to_equity", None)
        if d2e is not None:
            details.append(f"Debt/Equity ≈ {d2e:.2f}")
            if d2e < 0.5:
                score += 3
            elif d2e < 1.0:
                score += 2
            else:
                score -= 1
        cr = getattr(metrics[0], "current_ratio", None)
        if cr is not None:
            details.append(f"Current ratio ≈ {cr:.2f}")
            if cr >= 1.5:
                score += 2
            elif cr >= 1.0:
                score += 1
    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_steady_growth(financial_line_items: list) -> dict[str, Any]:
    """Steady, sustainable growth check."""
    max_score, score, details = 6, 0, []
    rev = _safe_series(financial_line_items, "revenue")
    if len(rev) >= 3 and rev[-1] > 0:
        years = len(rev) - 1
        try:
            cagr = (rev[0] / rev[-1]) ** (1 / years) - 1
        except Exception:
            cagr = 0.0
        details.append(f"Revenue CAGR ≈ {cagr:.1%}")
        if 0.03 <= cagr <= 0.10:
            score += 4
            details.append("Sustainable moderate growth")
        elif cagr > 0.10:
            score += 3
            details.append("Higher growth; ensure efficiency maintained")
        elif cagr < 0:
            score -= 1
            details.append("Declining revenue")
    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def conservative_intrinsic_value(financial_line_items: list, discount_rate=0.10, terminal_growth=0.02) -> dict[str, Any]:
    """Conservative DCF intrinsic value estimate."""
    if not financial_line_items:
        return {"intrinsic_value": None, "details": "No data"}

    fcf = _safe_series(financial_line_items, "free_cash_flow")
    if not fcf:
        return {"intrinsic_value": None, "details": "No FCF data"}

    current_fcf = fcf[0]
    years = 5
    growth_rate = 0.04  # modest
    projections = []
    for year in range(1, years + 1):
        current_fcf *= (1 + growth_rate)
        projections.append(current_fcf)

    discounted_sum = sum(f / ((1 + discount_rate) ** i) for i, f in enumerate(projections, start=1))
    terminal_value = projections[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_present = terminal_value / ((1 + discount_rate) ** years)

    intrinsic_value = discounted_sum + terminal_present
    return {
        "intrinsic_value": intrinsic_value,
        "details": f"DCF with {growth_rate:.0%} growth, {discount_rate:.0%} discount, {terminal_growth:.0%} terminal",
    }


# ---------------------------------
# LLM output generation
# ---------------------------------
def generate_kamprad_output(
    ticker: str,
    analysis_data: dict[str, Any],
    state: AgentState,
    agent_id: str = "ingvar_kamprad_agent",
) -> IngvarKampradSignal:
    """
    Generate decision in the voice of Ingvar Kamprad:
    humble, practical, focused on frugality, efficiency, long-term usefulness.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Ingvar Kamprad, founder of IKEA.
You speak in a humble, practical tone, focusing on frugality, efficiency, usefulness to ordinary people, and long-term stability.
You avoid hype and are skeptical of corporate excess.
""",
            ),
            (
                "human",
                """Analyze {ticker} with the computed data below:

{analysis_data}

Return ONLY valid JSON:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": number between 0 and 100,
  "reasoning": "Your assessment in plain English, in Ingvar Kamprad's style: humble, cost-conscious, long-term oriented"
}}"""
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def default_signal():
        return IngvarKampradSignal(signal="neutral", confidence=50.0, reasoning="Insufficient data for confident decision.")

    return call_llm(
        prompt=prompt,
        pydantic_model=IngvarKampradSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default_signal,
    )