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
    indicators: Optional[Indicators] = None     # m5
    h1_indicators: Optional[Indicators] = None  # m15
    h4_indicators: Optional[Indicators] = None  # m30
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = None     # m5
    candles2: Optional[List[Candle]] = None     # m15
    candles3: Optional[List[Candle]] = None     # m30
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

def in_london_ny_session():
    # London: 07:00–17:00 UK, NY: 13:00–21:00 UK
    now = uk_time_now().time()
    is_london = time(8, 0) <= now < time(17, 0)
    is_ny = time(13, 0) <= now < time(22, 0)
    return is_london or is_ny

@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind_m5 = trade.indicators or Indicators()
    ind_m15 = trade.h1_indicators or Indicators()
    ind_m30 = trade.h4_indicators or Indicators()
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles_m5 = (trade.candles1 or [])[-5:]
    candles_m15 = (trade.candles2 or [])[-5:]
    candles_m30 = (trade.candles3 or [])[-5:]
    cross_signal = trade.cross_signal or "none"
    cross_meaning = trade.cross_meaning or "none"

    logging.info(f"🔻 RAW PAYLOAD:\n{wrapper.json()}\n---")

    if not in_london_ny_session():
        logging.info("⏳ Out of London/NY session, no new trades.")
        return JSONResponse(content={"action": "hold", "reason": "Outside London/New York session", "confidence": 0})

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active", "confidence": 0})

    logging.info(f"✅ {trade.symbol} | m5 Dir: {getattr(pos, 'direction', None)} | {getattr(pos, 'open_price', None)} → {getattr(pos, 'pnl', None)}")
    logging.info(
        f"📊 m5 BB: ({ind_m5.bb_upper}, {ind_m5.bb_middle}, {ind_m5.bb_lower}) | "
        f"Stoch: K={ind_m5.stoch_k}, D={ind_m5.stoch_d}, J={ind_m5.stoch_j} | "
        f"MACD: {getattr(ind_m5.macd, 'main', None)}/{getattr(ind_m5.macd, 'signal', None)} | "
        f"EMA: {ind_m5.ema} LWMA: {ind_m5.lwma} SMMA: {ind_m5.smma} | "
        f"ADX: {ind_m5.adx} | MFI: {ind_m5.mfi} | WillR: {ind_m5.williams_r}"
    )

    # ========== GPT PROMPT ==========
    prompt = f"""
You are an elite, decisive prop firm scalper trade assistant. 
***You must take every setup that meets the entry rules.*** 
Only reply "hold" if there is a direct conflict or a clear lack of confluence.

**Session/time rules:**
- Trade at all times except between 21:00 and 23:00 UK time.
- Between 21:00 and 23:00 UK, do NOT open new trades but continue to manage (move SL, take profit, close) existing positions as needed.
- Always prioritize closing any trades in profit before 22:00 UK time to avoid spread widening.

**Entry:**
- The latest cross_signal from the EA is: {cross_signal}
- The latest cross_meaning from the EA is: {cross_meaning}
- Only take trades if the m5 EMA/LWMA cross matches the trend of at least one higher timeframe (m15 or m30).
- If you detect 3 or more of the following ("confluences") with no direct conflicts, issue a trade ("buy" or "sell"): 
  - MACD
  - SMMA
  - RSI or Stochastic
  - ADX > 20
  - Ichimoku agrees
  - Bollinger Bands breakout or squeeze
  - Candlestick reversal at a key level (S/R/fibonacci)
- If ALL indicators align (m5, m15, m30), lot size should be 2. Otherwise, use 1.

**Exit/scalp:**
- When unrealized profit (pos.pnl) >= 0.10% of account.balance (e.g. £100 on £100,000), always reply with BOTH a partial profit close (e.g. "partial_close": 0.5) AND moving the stop loss to entry ("new_sl": position.open_price) in the SAME response. Never partial close near break-even.
- Take second partial only if pos.pnl >= 0.20% of account.balance.
- At 1% profit (pos.pnl >= 1% of balance), use a 0.30% trailing stop.
- Exit the rest if 2+ indicators reverse or structure breaks, but only if trade is in profit.
(Unrealized profit = pos.pnl, Balance = account.balance)

**SL/TP:**  
- Always suggest new SL and TP based on high timeframe. 
- SL: Just beyond nearest m5 or m15 swing high/low (or min 1xATR)
- TP: At least 2xSL or next major S/R.

**Quality filter:**
- If at least 3 confluences and the main signal matches a higher timeframe, always take a trade.
- Default to "hold" only if clear conflict or no setup.
- Always include "confidence" (1-10).
- Do NOT allow trades that risk being held after 21:00 UK or over the weekend.
- Always give all confluences in 'reason'.

Example JSON reply:
{{
  "action": "hold",
  "reason": "Partial profit at $100, SL moved to entry as required.",
  "confidence": 9,
  "partial_close": 0.5,
  "new_sl": 2310.0
}}
{{
  "action": "buy",
  "reason": "m5 EMA over LWMA, m15 uptrend, MACD and ADX > 20, BB breakout. Aggressive trailing stop set, partial close triggered.",
  "confidence": 9,
  "lot": 2,
  "new_sl": 2301.5,
  "new_tp": 2310.0,
  "partial_close": 0.5
}}
{{
  "action": "hold",
  "reason": "m15 trend flat, or conflict between MACD and price action.",
  "confidence": 3
}}

Current Cross Signal: {cross_signal}
Current Cross Meaning: {cross_meaning}
Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
Recent m5 Candles: {[candle.dict() for candle in candles_m5]}
Indicators (m5): {ind_m5.dict()}
Indicators (m15): {ind_m15.dict()}
Indicators (m30): {ind_m30.dict()}
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
            temperature=0.11,
            response_format={"type": "json_object"}
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        action = flatten_action(decision)
        allowed = {"hold", "close", "trail_sl", "trail_tp", "buy", "sell"}

        # Confidence filter: only trade if 7+
        conf = action.get("confidence", 0)
        if action.get("action") in {"buy", "sell"} and conf < 7:
            action["action"] = "hold"
            action["reason"] = (action.get("reason") or "") + " (confidence too low for entry)"
        # Lot sizing for high confluence
        if action.get("action") in {"buy", "sell"} and "lot" not in action:
            action["lot"] = 2 if "double" in (action.get("reason") or "").lower() or conf >= 9 else 1

        if "reason" not in action or not action["reason"]:
            action["reason"] = "No reasoning returned by GPT."

        # === GUARD: No partial close under 0.1% profit, and add SL to entry after partial 1 ===
        if action.get("partial_close") and pos and acc:
            min_partial = acc.balance * 0.001  # 0.10% of balance
            if pos.pnl < min_partial:
                logging.warning(f"🚫 Blocking early partial close — PnL {pos.pnl:.2f} < {min_partial:.2f}")
                action.pop("partial_close", None)
                action["reason"] += f" (partial blocked — not enough profit)"
            else:
                # After first partial, move SL to entry if not already set
                if not action.get("new_sl") and pos.open_price:
                    action["new_sl"] = pos.open_price
                    action["reason"] += f" | SL moved to entry after partial"

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
    return {"message": "SmartGPT EA SCALPER - London/NY session, EMA/LWMA/SMMA confluence, aggressive trailing stop, partial profits"}
