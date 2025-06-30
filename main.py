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

# === Data Models (Match EA Payload) ===

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
    indicators: Optional[Indicators] = None     # 5m
    h1_indicators: Optional[Indicators] = None  # 15m
    h4_indicators: Optional[Indicators] = None  # 1h
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = None     # 5m
    candles2: Optional[List[Candle]] = None     # 15m
    candles3: Optional[List[Candle]] = None     # 1h
    news_override: Optional[bool] = False
    live_candle1: Optional[Candle] = None
    live_candle2: Optional[Candle] = None

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

    logging.info(f"🔻 RAW PAYLOAD:\n{wrapper.json()}\n---")

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active", "confidence": 0})

    logging.info(f"✅ {trade.symbol} | 5m Dir: {getattr(pos, 'direction', None)} | {getattr(pos, 'open_price', None)} → {getattr(pos, 'pnl', None)}")
    logging.info(
        f"📊 5m BB: ({ind_5m.bb_upper}, {ind_5m.bb_middle}, {ind_5m.bb_lower}) | "
        f"Stoch: K={ind_5m.stoch_k}, D={ind_5m.stoch_d}, J={ind_5m.stoch_j} | "
        f"MACD: {getattr(ind_5m.macd, 'main', None)}/{getattr(ind_5m.macd, 'signal', None)} | "
        f"EMA: {ind_5m.ema} LWMA: {ind_5m.lwma} SMMA: {ind_5m.smma} | "
        f"ADX: {ind_5m.adx} | MFI: {ind_5m.mfi} | WillR: {ind_5m.williams_r}"
    )

    prompt = f"""
You are a decisive prop firm trading assistant.

- The EA handles all partial profits and break-even moves.
- **IF THE STOP LOSS IS AT BREAKEVEN (SL == entry price), YOU MUST NOT SUGGEST OR MOVE THE SL.**
- DO NOT open any new trades after 17:00 UK time on Friday. Only close or manage existing trades.
- DO NOT open any new trades between 21:00 and 23:00 UK time.
- Always try to close profitable trades before 22:00 UK time or before the weekend.
- You CAN suggest a new take profit (TP) or a full close if necessary.
- Require at least 3 confluences for a new entry.
- ONLY reply in VALID JSON using the example format.
- When replying "hold", ALWAYS explain the decision using the live indicator values or actual price action context. NEVER copy a template; your reason should be unique and reflect the real market.

IMPORTANT:
- DO NOT move or suggest a new SL if the SL is already at breakeven.
- DO NOT suggest new entries after 17:00 UK Friday or between 21:00 and 23:00 UK time.
- Only take entries when 5m EMA/LWMA cross matches the trend of 15m or 1H and you have at least three confluences.

ENTRY RULES:
- The latest cross_signal from the EA is: {cross_signal}
- The latest cross_meaning from the EA is: {cross_meaning}
- Only take trades if the 5m EMA/LWMA cross matches the trend of at least one higher timeframe (15m or 1H).
- If you detect 3 or more of the following ("confluences") with no direct conflicts, issue a trade ("buy" or "sell"):
  - MACD
  - SMMA
  - RSI or Stochastic
  - ADX > 20
  - Ichimoku agrees
  - Bollinger Bands breakout or squeeze
  - Candlestick reversal at a key level (S/R/fibonacci)
- If ALL indicators align (5m, 15m, 1H), lot size should be 2. Otherwise, use 1.

EXIT/SCALP:
- Do NOT suggest or move SL if SL is at breakeven (SL == entry price).
- Suggest a new TP or a full close if a strong reversal/confluence break occurs.
- Always try to close profitable trades before the weekend.

SL/TP:
- Always suggest new SL and TP based on high timeframe, but only if SL is not at breakeven.
- SL: Just beyond nearest 5m or 15m swing high/low (or min 1xATR)
- TP: At least 2xSL or next major S/R.

EXAMPLES (JSON):
{{
  "action": "buy",
  "reason": "5m EMA over LWMA, 15m uptrend, MACD, ADX, and BB breakout. Three confluences: MACD rising, ADX>20, BB squeeze breakout.",
  "confidence": 9,
  "lot": 2,
  "new_sl": 2301.5,
  "new_tp": 2310.0
}}
{{
  "action": "close",
  "reason": "Reversal on 15m, confluence breakdown: MACD cross, ADX falling.",
  "confidence": 9
}}
{{
  "action": "hold",
  "reason": "Hold, price action choppy and MACD/RSI are not aligned. Current indicators: MACD {ind_5m.macd}, RSI {ind_5m.rsi_array}, BB {ind_5m.bb_upper}/{ind_5m.bb_lower}.",
  "confidence": 2
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
                {"role": "system", "content": "You are an elite, disciplined, risk-aware SCALPER trade assistant. Reply ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.15,
            response_format={"type": "json_object"}
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        action = flatten_action(decision)
        allowed = {"hold", "close", "trail_sl", "trail_tp", "buy", "sell"}

        conf = action.get("confidence", 0)
        if action.get("action") in {"buy", "sell"} and conf < 7:
            action["action"] = "hold"
            action["reason"] = (action.get("reason") or "") + " (confidence too low for entry)"
        if action.get("action") in {"buy", "sell"} and "lot" not in action:
            action["lot"] = 2 if "double" in (action.get("reason") or "").lower() or conf >= 9 else 1

        if "reason" not in action or not action["reason"]:
            action["reason"] = "No reasoning returned by GPT."

        # === SL Breakeven Guard ===
        if pos and pos.open_price and pos.sl is not None:
            if abs(pos.sl - pos.open_price) < 1e-5:
                if "new_sl" in action and action["new_sl"] != pos.sl:
                    action["reason"] += " | SL at breakeven, not allowed to move."
                    action["new_sl"] = pos.sl

        # === FRIDAY 5PM+ GUARD: No new trades after 17:00 Friday, close profits ===
        if is_friday_5pm_or_later():
            if pos and pos.pnl and pos.pnl > 0 and action.get("action") not in {"close", "hold"}:
                action["action"] = "close"
                action["reason"] += " | Closing profitable trade before weekend."
            elif action.get("action") in {"buy", "sell"}:
                action["action"] = "hold"
                action["reason"] += " | No new trades after 17:00 UK time Friday (weekend risk)."

        # === TIME GUARD: Block new entries after 21:00, force closing profits before 22:00 ===
        if pos and pos.pnl and acc:
            if is_between_uk_time(21, 23) and action.get("action") in {"buy", "sell"}:
                action["action"] = "hold"
                action["reason"] += " | No new trades between 21:00 and 23:00 UK time."
            if is_between_uk_time(21, 22) and pos.pnl > 0 and action.get("action") not in {"close", "hold"}:
                action["action"] = "close"
                action["reason"] += " | Closing profitable trade before 22:00 UK to avoid spread widening."

        logging.info(f"📝 GPT Action: {action.get('action')} | Lot: {action.get('lot', 1)} | Confidence: {action.get('confidence', 0)} | Reason: {action.get('reason','(none)')}")
        if action.get("action") in allowed:
            return JSONResponse(content=action)
        else:
            return JSONResponse(content={"action": "hold", "reason": f"Could not decode: {decision}", "confidence": 0})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "reason": str(e), "confidence": 0})

@app.get("/")
async def root():
    return {"message": "SmartGPT EA SCALPER - All Sessions, EMA/LWMA/SMMA confluence, 3-confluence filter, 5m/15m/1H logic, prop firm weekend safety, partial profits & BE handled by EA"}
