from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse
import openai
import os
import logging
import json
import re

# === Setup ===
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === Data Models ===

class MACD(BaseModel):
    main: float
    signal: float

class Ichimoku(BaseModel):
    tenkan: float
    kijun: float
    senkou_a: float
    senkou_b: float

class Indicators(BaseModel):
    bb_upper: float
    bb_middle: float
    bb_lower: float
    stoch_k: float
    stoch_d: float
    stoch_j: Optional[float]
    macd: MACD
    sma100: Optional[float]
    ema40: Optional[float]
    adx: Optional[float]
    mfi: Optional[float]
    williams_r: Optional[float]
    ichimoku: Optional[Ichimoku]
    rsi_array: Optional[List[float]]
    price_array: Optional[List[float]]
    support_resistance: Optional[Dict[str, List[float]]] = None
    fibonacci: Optional[Dict[str, Any]] = None
    candlestick_patterns: Optional[List[str]] = None
    atr: Optional[float] = None   # <-- Optionally add ATR if you include it in payload later

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class Position(BaseModel):
    direction: Optional[str]
    open_price: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    lot: Optional[float]
    pnl: Optional[float]

class Account(BaseModel):
    balance: float
    equity: float
    margin: Optional[float]

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    direction: Optional[str] = None
    open_price: Optional[float] = None
    current_price: Optional[float] = None
    news_override: Optional[bool] = False
    indicators: Indicators
    h1_indicators: Optional[Indicators] = None
    h4_indicators: Optional[Indicators] = None
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = []
    candles2: Optional[List[Candle]] = []
    candles3: Optional[List[Candle]] = []
    live_candle1: Optional[Candle] = None
    live_candle2: Optional[Candle] = None

class TradeWrapper(BaseModel):
    data: TradeData

# === Helper: Flatten GPT response ===
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
    return {"action": "hold", "reason": "Could not decode action."}

# === GPT Manager ===
@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind = trade.indicators
    h1 = trade.h1_indicators or Indicators(**{})
    h4 = trade.h4_indicators or Indicators(**{})
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles = trade.candles1[-5:] if trade.candles1 else []

    logging.info(f"✅ {trade.symbol} | Dir: {trade.direction} | {trade.open_price} → {trade.current_price}")
    logging.info(
        f"📊 BB: ({ind.bb_upper}, {ind.bb_middle}, {ind.bb_lower}) | "
        f"Stoch: K={ind.stoch_k}, D={ind.stoch_d}, J={getattr(ind, 'stoch_j', None)} | "
        f"MACD: {ind.macd.main}/{ind.macd.signal} | "
        f"SMA100: {ind.sma100}, EMA40: {ind.ema40} | "
        f"ADX: {getattr(ind, 'adx', None)} | "
        f"MFI: {getattr(ind, 'mfi', None)} | Williams %R: {getattr(ind, 'williams_r', None)} | "
        f"Ichimoku: {ind.ichimoku.dict() if ind.ichimoku else None}"
    )

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active"})

    # === Enhanced Prompt ===
    prompt = f"""
You are an elite, disciplined algorithmic trade manager for prop firm challenges.
Your job is to maximize profit and account growth while **holding trades for the full potential of the trend**, unless there is a true structural break.

**Updated Strategy Rules for Optimal Trade Duration:**
- **Hold trades for as long as higher timeframe (H1/H4) trend and structure remain in agreement with entry direction.**
- Do **not** move stop loss to breakeven after minor moves, especially if higher TF trend is intact.
- Only suggest moving stop loss up (to breakeven or profit) if BOTH local and higher timeframes show warning signs, OR if the trade is up more than +50 pips or 1.5x ATR.
- Do **not** close trades for small profits if the higher timeframe is strong—**ride the trend!**
- Suggest `"hold_duration"` (in candles or minutes) if you want to hold even longer.
- Trail stop loss only after strong moves (e.g., +1 ATR or clear trend extension).
- Partial profit is permitted, but prefer to hold the entire position unless technical structure breaks or trade confidence drops.
-Be a sniper: take frequent trades, but only when the setup meets strict, high-quality criteria. Prioritise quality, but do not hesitate when genuine high-probability opportunities present themselves—even if that means trading more often. When a top-tier entry appears, double your usual lot size.

**Classic Confluence Criteria:**
- Trade only with confluence: candle + structure + multiple indicators.
- In strong trends, treat overbought/oversold as momentum, not as a reversal by itself.
- ADX > 20–25 and Ichimoku support = real trend; don't exit early.
- Confirm with EMA/SMA crossover, MACD, and candlestick at key levels.
- RSI extremes alone are **not** exit signals—look for structure and HTF trend shift.

**Risk & Execution Rules:**
- Never break drawdown rules (daily or max).
- `"lot": 2` only with overwhelming confluence and strong trend on all timeframes.
- No adding to open positions. No flipping unless true reversal confirmed on multiple timeframes.
- `"hold"` only if there is no clear edge or confluence.
- `"buy"` or `"sell"` when everything aligns—**hold as long as possible**.

**SL/TP Management:**
- Only return `"new_sl"` or `"trail_sl"` if structure or higher timeframe confirms.
- `"new_sl"` to breakeven after +50 pips **and** higher TF agrees or trade is at least +1 ATR in profit.
- `"hold_duration"` is encouraged in strong trends.

**ALWAYS reply in strict JSON:**
Examples:
{{"action":"hold","hold_duration":12,"reason":"Higher timeframe trend still bullish, holding for more extension."}}
{{"action":"trail_sl","new_sl":1.2350,"reason":"Up +80p and H1/H4 structure still supports uptrend—trail below last swing low."}}
{{"action":"close","reason":"H1 structure break, MACD crosses against, exiting."}}

Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
Recent Candles: {[candle.dict() for candle in candles]}

Market Data:
Symbol: {trade.symbol}
Timeframe: {trade.timeframe}
Open Price: {trade.open_price}
Current Price: {trade.current_price}
M5 Indicators: {ind.dict()}
H1 Indicators: {h1.dict() if h1 else None}
H4 Indicators: {h4.dict() if h4 else None}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an elite, disciplined, risk-aware trade assistant. Reply ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.18
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        allowed = {"hold", "close", "trail_sl", "trail_tp", "martingale", "buy", "sell"}
        action = flatten_action(decision)
        if action.get("action") in {"buy", "sell", "martingale"} and "lot" not in action:
            action["lot"] = 1
        if action.get("action") in allowed or action.get("action") == "hold":
            logging.info(f"📝 GPT Action: {action.get('action')} | Lot: {action.get('lot', 1)} | Reason: {action.get('reason','(none)')}")
            return JSONResponse(content=action)
        else:
            return JSONResponse(content={"action": "hold", "reason": f"Could not decode: {decision}"})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "reason": str(e)})

@app.get("/")
async def root():
    return {"message": "SmartGPT EA Trade Server running!"}
