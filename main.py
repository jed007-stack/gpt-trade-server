from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse
import openai
import os
import logging
import json
import re
from datetime import datetime, time
import pytz

# === Setup ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")
openai_client = openai.OpenAI(api_key=api_key)

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === Data Models ===
class MACD(BaseModel):
    main: Optional[float] = None
    signal: Optional[float] = None

class Ichimoku(BaseModel):
    tenkan: Optional[float] = None
    kijun: Optional[float] = None
    senkou_a: Optional[float] = None
    senkou_b: Optional[float] = None

class Indicators(BaseModel):
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    stoch_j: Optional[float] = None
    macd: Optional[MACD] = None
    ema: Optional[float] = None
    ema_period: Optional[int] = None
    lwma: Optional[float] = None
    lwma_period: Optional[int] = None
    smma: Optional[float] = None
    smma_period: Optional[int] = None
    adx: Optional[float] = None
    mfi: Optional[float] = None
    williams_r: Optional[float] = None
    ichimoku: Optional[Ichimoku] = None
    rsi_array: Optional[List[float]] = None
    price_array: Optional[List[float]] = None
    support_resistance: Optional[Dict[str, List[float]]] = None
    fibonacci: Optional[Dict[str, Any]] = None
    candlestick_patterns: Optional[List[str]] = None
    atr: Optional[float] = None

class Candle(BaseModel):
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None

class Position(BaseModel):
    direction: Optional[str] = None
    open_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None
    pnl: Optional[float] = None

class Account(BaseModel):
    balance: Optional[float] = None
    equity: Optional[float] = None
    margin: Optional[float] = None

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    update_type: Optional[str] = None
    cross_signal: Optional[str] = None
    cross_meaning: Optional[str] = None
    indicators: Optional[Indicators] = None
    h1_indicators: Optional[Indicators] = None
    h4_indicators: Optional[Indicators] = None
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = None
    candles2: Optional[List[Candle]] = None
    candles3: Optional[List[Candle]] = None
    news_override: Optional[bool] = False
    live_candle1: Optional[Candle] = None
    live_candle2: Optional[Candle] = None
    last_trade_was_loss: Optional[bool] = False
    unrecovered_loss: Optional[float] = 0.0

class TradeWrapper(BaseModel):
    data: TradeData

def flatten_action(decision):
    if isinstance(decision, dict) and "action" in decision:
        return decision
    if isinstance(decision, str):
        cleaned = re.sub(r"```(\w+)?", "", decision).strip()
        try:
            d = json.loads(cleaned)
            if "action" in d:
                return d
        except Exception:
            pass
    if isinstance(decision, dict) and "raw" in decision:
        cleaned = re.sub(r"```(\w+)?", "", decision["raw"]).strip()
        try:
            d = json.loads(cleaned)
            if "action" in d:
                return d
        except Exception:
            pass
    return {"action": "hold", "reason": "Could not decode action.", "confidence": 0}

def uk_time_now():
    utc_now = datetime.utcnow()
    london = pytz.timezone('Europe/London')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(london)

def is_between_uk_time(start_h, end_h):
    now = uk_time_now().time()
    return time(start_h, 0) <= now < time(end_h, 0)

def is_friday_5pm_or_later():
    now = uk_time_now()
    return now.weekday() == 4 and now.time() >= time(17, 0)

def extract_categories(reason):
    m = re.search(r"Confluences?:\s*([^.]+)", reason, re.IGNORECASE)
    if not m:
        return set()
    cats = m.group(1)
    found = set()
    for cat in ["Trend", "Momentum", "Volatility", "Volume", "Structure", "ADX"]:
        if re.search(r"\b{}\b".format(cat), cats, re.IGNORECASE):
            found.add(cat.lower())
    return found

@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind_5m = trade.indicators or Indicators()
    ind_15m = trade.h1_indicators or Indicators()
    ind_1h = trade.h4_indicators or Indicators()
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles_5m = (trade.candles1 or [])[-5:]
    candles_15m = (trade.candles2 or [])[-5:]
    candles_1h = (trade.candles3 or [])[-5:]
    cross_signal = trade.cross_signal or "none"
    cross_meaning = trade.cross_meaning or "none"

    # -------- Recovery Mode Logic --------
    in_recovery_mode = False
    unrecovered_loss = 0.0
    try:
        unrecovered_loss = float(getattr(trade, "unrecovered_loss", 0.0) or 0.0)
    except Exception:
        unrecovered_loss = 0.0
    if getattr(trade, "last_trade_was_loss", False):
        in_recovery_mode = True
    if unrecovered_loss > 0.0:
        in_recovery_mode = True

    logging.info(f"🔻 RAW PAYLOAD:\n{wrapper.json()}\n---")
    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={
            "action": "hold",
            "reason": "News conflict — override active",
            "confidence": 0,
            "categories": [],
            "recovery_mode": in_recovery_mode
        })

    recovery_note = ""
    if in_recovery_mode:
        recovery_note = (
            "\n---\n"
            "RECOVERY MODE: The last trade was a loss and has not yet been recovered. "
            "You must only recommend a trade if at least **4 out of 6 unique confluence categories** align and overall confidence is 8 or higher. "
            "You must *explicitly state which categories* are satisfied, and only one indicator per category is allowed (see below). "
            "If fewer than 4 unique categories are satisfied, hold. Justify recovery trades with extra strictness, and reference recovery mode in your reason."
            "\n---\n"
        )

    prompt = f"""{recovery_note}
You are a decisive, disciplined prop firm trading assistant. DO NOT be lazy or generic; always justify every action using live indicator values and current price context. NEVER reuse generic logic.
If you do not explicitly list at least {(4 if in_recovery_mode else 3)} unique categories (Trend, Momentum, Volatility, Volume, Structure, ADX) in your reason as in the example, your action will be set to 'hold' and the trade will not be taken.
Never say just 'multiple confluences' or generic logic. List each category and which indicator fills it, every time.

CONFLUENCE LOGIC:
- There are **6 unique categories**: 
  1. TREND: (choose one, e.g. EMA/LWMA cross OR Ichimoku OR SMMA—but only one counts as Trend)
  2. MOMENTUM: (MACD OR RSI OR Stochastic—only one counts as Momentum)
  3. VOLATILITY: (Bollinger Bands OR ATR—only one counts as Volatility)
  4. VOLUME: (MFI OR volume spike—only one counts as Volume)
  5. STRUCTURE: (Key support/resistance OR Fibonacci alignment OR reversal candle—only one counts as Structure)
  6. ADX: (ADX > 20 and direction—only one counts as ADX)
- When justifying an entry, you **must explicitly state which categories are satisfied**, e.g.: 
  `Confluences: Trend (EMA cross), Momentum (MACD), Volatility (BB), Structure (Fib), ADX (trend > 20).`
- **Only one indicator per category can count towards the confluence total.**
- If more than one indicator in the same category aligns, only count the strongest or most significant.
- When replying, provide a line listing the *categories* being counted. Do not “double-count” indicators in the same category.
- Do NOT claim more than 6 confluences; do not count two momentum or two trend indicators as separate.

ENTRY RULES:
- Never recommend a trade if SMMA (slow MA) is trending opposite to the entry direction. If so, reply with "hold" and explain: "Trade direction conflicts with SMMA trend."
- Only take a trade if at least {(4 if in_recovery_mode else 3)} different categories align (not just indicators).
- In recovery mode, at least 4 out of 6 categories must align, per above.
- If ALL 6 align (including strong SMMA confirmation), lot = 3. If 4-5, lot = 2. If 3, lot = 1.
- If price is close to, hugging, or crossing SMMA, reduce confidence and consider holding or warning of possible trend reversal.
- Always mention SMMA status for every signal and justify your confidence.

RISK/SESSION GUARDS:
- DO NOT move SL if SL is already at breakeven (SL == entry price).
- DO NOT suggest or open new trades after 17:00 UK time Friday, or between 21:00-23:00 UK time.
- Always try to close profitable trades before 22:00 UK or before the weekend.
- EA handles all partial profit and break-even moves.

SL/TP RULES:
- SL: Just beyond last swing high/low or min 1xATR.
- TP: At least 2xSL, or at next major SR/Fibonacci level.

When replying, ALWAYS reference at least {(4 if in_recovery_mode else 3)} unique indicator categories (not just momentum!), and ALWAYS mention SMMA status. DO NOT skip or be generic.

EXAMPLES (JSON only, strictly follow this style):
{{
  "action": "buy",
  "reason": "Confluences: Trend (EMA/LWMA cross up), Momentum (MACD positive), Volatility (BB squeeze breakout). SMMA is sloping up, confirming trend.",
  "confidence": 9,
  "lot": 2,
  "new_sl": 2290,
  "new_tp": 2310
}}
{{
  "action": "hold",
  "reason": "Confluences: Trend (EMA cross down). Sell signal detected but SMMA is sloping up, uptrend still intact, not entering against dominant trend. MACD and RSI also not confirming.",
  "confidence": 2
}}
{{
  "action": "close",
  "reason": "Confluences: Trend (SMMA turning down), Momentum (MACD), Structure (support break). Trend reversal: SMMA has turned down and price crossed SMMA. 15m MACD down, price broke below Ichimoku cloud, major SR break.",
  "confidence": 9
}}

Current Cross Signal: {cross_signal}
Current Cross Meaning: {cross_meaning}
Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
SL: {getattr(pos, 'sl', None)} | Open: {getattr(pos, 'open_price', None)}
Recent 5m Candles: {[candle.dict() for candle in candles_5m]}
Indicators (5m): {ind_5m.dict()}
Indicators (15m): {ind_15m.dict()}
Indicators (1H): {ind_1h.dict()}
"""

    logging.info(f"\n========= SENDING PROMPT TO GPT =========\n{prompt}\n========================================\n")

    try:
        chat = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an elite, disciplined, risk-aware SCALPER trade assistant. Reply ONLY in valid JSON. Fill all fields. For every buy or sell, always suggest new_sl and new_tp."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.12
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        action = flatten_action(decision)
        allowed = {"hold", "close", "trail_sl", "trail_tp", "buy", "sell"}

        claimed = extract_categories(action.get("reason", ""))
        cat_count = len(claimed)
        conf = action.get("confidence", 0)

        # HARD anti-lazy: must have confluences and enough
        if ("confluences:" not in action.get("reason", "").lower()) or (cat_count < (4 if in_recovery_mode else 3)):
            action["action"] = "hold"
            action["reason"] += f" | GPT did not properly explain confluences or unique categories ({cat_count} found)."

        if in_recovery_mode:
            if action.get("action") in {"buy", "sell"} and conf < 8:
                action["action"] = "hold"
                action["reason"] += " | Recovery mode: Not enough confluence/confidence for recovery entry."
        else:
            if action.get("action") in {"buy", "sell"} and conf < 7:
                action["action"] = "hold"
                action["reason"] += " (confidence too low for entry)"

        if action.get("action") in {"buy", "sell"} and "lot" not in action:
            action["lot"] = 2 if conf >= 9 else 1

        if "reason" not in action or not action["reason"]:
            action["reason"] = "No reasoning returned by GPT."

        # SL Breakeven Guard
        if pos and pos.open_price and pos.sl is not None:
            if abs(pos.sl - pos.open_price) < 1e-5:
                if "new_sl" in action and action["new_sl"] != pos.sl:
                    action["reason"] += " | SL at breakeven, not allowed to move."
                    action["new_sl"] = pos.sl

        # Friday/weekend/session guards
        if is_friday_5pm_or_later():
            if pos and pos.pnl and pos.pnl > 0 and action.get("action") not in {"close", "hold"}:
                action["action"] = "close"
                action["reason"] += " | Closing profitable trade before weekend."
            elif action.get("action") in {"buy", "sell"}:
                action["action"] = "hold"
                action["reason"] += " | No new trades after 17:00 UK time Friday (weekend risk)."

        if pos and pos.pnl and acc:
            if is_between_uk_time(21, 23) and action.get("action") in {"buy", "sell"}:
                action["action"] = "hold"
                action["reason"] += " | No new trades between 21:00 and 23:00 UK time."
            if is_between_uk_time(21, 22) and pos.pnl > 0 and action.get("action") not in {"close", "hold"}:
                action["action"] = "close"
                action["reason"] += " | Closing profitable trade before 22:00 UK."

        action["categories"] = sorted(list(claimed))
        action["recovery_mode"] = in_recovery_mode

        logging.info(f"📝 GPT Action: {action.get('action')} | Lot: {action.get('lot', 1)} | Confidence: {action.get('confidence', 0)} | Reason: {action.get('reason','(none)')} | Categories: {action['categories']} | Recovery: {action['recovery_mode']}")
        if action.get("action") in allowed:
            return JSONResponse(content=action)
        else:
            return JSONResponse(content=action)

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={
            "action": "hold",
            "reason": str(e),
            "confidence": 0,
            "categories": [],
            "recovery_mode": in_recovery_mode
        })

@app.get("/")
async def root():
    return {"message": "SmartGPT EA SCALPER (3.5-turbo, full/verbose) - Multi-confluence, strict SMMA, strict category-checked confluence, strict SL/TP suggestion, prop/session safety, E8 loss recovery, anti-lazy JSON enforcement."}
