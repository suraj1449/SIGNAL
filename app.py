"""
NIFTY OI Dashboard  v7  —  Long Buildup / Short Covering Signals + LTP fix
═══════════════════════════════════════════════════════════════════════════
NEW IN v7:
  ① Long Buildup / Short Covering signals in the IV table:
      Each row now shows a "Buildup" column for CE and PE independently.

      Signal logic (compares current bar vs N-bars-ago, where N = TF):
        Price % Change   OI % Change   Signal
        ────────────────────────────────────────────────────────
        ≥ +5%            ≥ +5%         Long Build-up   ✅  Strong Buy
        ≥ +5%            ≤ -5%         Short Covering  🔥🔥 Very Strong Buy
        ≤ -5%            ≥ +5%         Short Build-up  🔻  Bearish
        ≤ -5%            ≤ -5%         Long Unwinding  ⚠️   Caution
        Other                          Neutral          —

      "Price" = option LTP (CE or PE separately)
      "OI"    = option OI (CE or PE separately)
      Both are compared over the selected timeframe window (1/3/5/10 min).

  ② Live LTP fix in IV table:
      - IV table LTP cells get a unique `id` per strike (`iv-ce-ltp-{strike}`)
      - The existing 1-second LTP loop now also updates these cells
      - LTP is no longer stale between OI polls

  ③ Price+OI history stored alongside IV:
      - `price_oi_history[strike]` = deque of {t, ce_ltp, pe_ltp, ce_oi, pe_oi}
      - `resample_price_oi()` resamples to TF and computes % changes for signal

RETAINED FROM v6:
  - Black-Scholes IV + EMA(9) signal
  - 4-day OI history chart (3 historical + today live)
  - All existing OI table, PCR bar, stat cards, Price/VWAP charts

DEPENDENCIES:  pip install flask kiteconnect
═══════════════════════════════════════════════════════════════════════════
"""

import math, os, time, threading
from datetime import datetime, date, timedelta
from collections import deque
from flask import Flask, jsonify, render_template_string, request
from kiteconnect import KiteConnect

API_KEY      = os.environ.get("KITE_API_KEY", "")
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")

STRIKE_GAP   = 50
OTM_DEPTH    = 5        # 11 strikes total
OI_INTERVAL  = 180      # seconds (3 min)
LTP_INTERVAL = 1
RISK_FREE    = 0.10    # 6.5% annualised risk-free rate (Indian T-bill proxy)

app  = Flask(__name__)
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

oi_data      = {}
ltp_data     = {}
last_oi_time = None
error_msg    = None
ltp_symbols  = []

PORT = int(os.environ.get("PORT", "5000"))
_startup_lock = threading.Lock()
_startup_done = False


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

_prev_oi        = {}
_instrument_map = {}
_token_map      = {}
_sym_to_token   = {}
_nearest_expiry = None

# Live OI snapshots: {strike_int: [{t, ce, pe}, …]}
oi_history: dict[int, list] = {}

# Historical OI cache
_hist_cache:       dict[int, dict] = {}
_hist_total_cache: dict            = {}
_hist_otm_cache:   dict            = {}
HIST_CACHE_TTL = 1800
HIST_TOTAL_CACHE_KEY = "total"
HIST_OTM_CACHE_KEY   = "otm"

# ── IV storage ────────────────────────────────────────────
# iv_history[strike] = {"ce": deque([{t, iv}, …]), "pe": deque([{t, iv}, …])}
# We keep at most 300 raw snapshots per strike
IV_HISTORY_MAXLEN = 300
iv_history: dict[int, dict] = {}   # populated during fetch_oi

# ── Price + OI history for buildup signals ────────────────
# price_oi_history[strike] = deque([{t, ce_ltp, pe_ltp, ce_oi, pe_oi}, …])
# Same cadence as fetch_oi (every OI_INTERVAL seconds)
price_oi_history: dict[int, deque] = {}   # populated during fetch_oi


# ═══════════════════════════════════════════════════════════
#  BLACK-SCHOLES IV  (Newton-Raphson, pure Python)
# ═══════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _bs_price(S, K, T, r, sigma, flag: str) -> float:
    """Black-Scholes European option price. flag = 'c' | 'p'."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if flag == 'c' else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if flag == 'c':
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_vega(S, K, T, r, sigma) -> float:
    """Black-Scholes vega (∂price/∂sigma)."""
    if T <= 0 or sigma <= 0:
        return 1e-8
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * _norm_cdf(d1) * math.sqrt(T)   # simplified; _norm_cdf here is pdf proxy


def _bs_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs_vega_correct(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0:
        return 1e-8
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * _bs_pdf(d1) * math.sqrt(T)


def implied_vol(S: float, K: float, T: float, r: float,
                market_price: float, flag: str,
                max_iter: int = 100, tol: float = 1e-6) -> float | None:
    """
    Compute implied volatility using Newton-Raphson.
    Returns None if price is below intrinsic (no real IV) or fails to converge.
    S     = spot price
    K     = strike
    T     = time to expiry in years
    r     = risk-free rate
    flag  = 'c' for call, 'p' for put
    """
    if T <= 0:
        return None
    intrinsic = max(0.0, (S - K) if flag == 'c' else (K - S))
    if market_price < intrinsic - 0.01:
        return None
    if market_price < 0.01:
        return None   # too cheap to compute meaningful IV

    sigma = 0.25   # initial guess
    for _ in range(max_iter):
        price = _bs_price(S, K, T, r, sigma, flag)
        vega  = _bs_vega_correct(S, K, T, r, sigma)
        diff  = price - market_price
        if abs(diff) < tol:
            break
        if abs(vega) < 1e-10:
            return None
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4
        if sigma > 5.0:
            sigma = 5.0
    return round(sigma * 100, 2) if 0 < sigma < 5.0 else None   # return as %


def tte_years(expiry: date) -> float:
    """Calendar days to expiry / 365."""
    today = date.today()
    delta = (expiry - today).days
    return max(delta / 365.0, 1 / 365.0)


# ── EMA helper ────────────────────────────────────────────

def compute_ema(values: list, period: int = 9) -> list:
    """
    Returns EMA list of same length as values.
    First <period> values are None (not enough data).
    """
    if len(values) < period:
        return [None] * len(values)
    k   = 2.0 / (period + 1)
    out = [None] * len(values)
    # Seed with SMA of first `period` points
    sma = sum(v for v in values[:period] if v is not None) / period
    out[period - 1] = round(sma, 2)
    for i in range(period, len(values)):
        if values[i] is None or out[i - 1] is None:
            out[i] = out[i - 1]
        else:
            out[i] = round(values[i] * k + out[i - 1] * (1 - k), 2)
    return out


def resample_iv_history(strike: int, side: str, tf_minutes: int) -> list:
    """
    Resample raw IV snapshots (1-min cadence) into `tf_minutes` bars.
    Returns list of {t, iv, ema9} dicts.
    """
    raw = iv_history.get(strike, {}).get(side, deque())
    if not raw:
        return []

    if tf_minutes <= 1:
        pts = list(raw)
    else:
        # Group by floored time slot
        from collections import OrderedDict
        buckets: dict = OrderedDict()
        for pt in raw:
            try:
                dt = datetime.strptime(pt["t"], "%d-%b %H:%M")
            except Exception:
                continue
            # Floor to tf_minutes
            floored_min = (dt.minute // tf_minutes) * tf_minutes
            slot_dt     = dt.replace(minute=floored_min, second=0)
            slot_lbl    = slot_dt.strftime("%d-%b %H:%M")
            buckets[slot_lbl] = pt["iv"]   # last-of-period IV

        pts = [{"t": t, "iv": iv} for t, iv in buckets.items()]

    ivs  = [p["iv"] for p in pts]
    emas = compute_ema(ivs, 9)
    for i, pt in enumerate(pts):
        pt["ema9"] = emas[i]
    return pts


def resample_price_oi(strike: int, tf_minutes: int) -> list:
    """
    Resample raw price+OI snapshots into `tf_minutes` bars.
    Returns list of {t, ce_ltp, pe_ltp, ce_oi, pe_oi} — last-of-period.
    """
    raw = price_oi_history.get(strike)
    if not raw:
        return []
    if tf_minutes <= 1:
        return list(raw)
    from collections import OrderedDict
    ce_buckets: dict = OrderedDict()
    pe_buckets: dict = OrderedDict()
    oi_ce_buckets: dict = OrderedDict()
    oi_pe_buckets: dict = OrderedDict()
    for pt in raw:
        try:
            dt = datetime.strptime(pt["t"], "%d-%b %H:%M")
        except Exception:
            continue
        floored_min = (dt.minute // tf_minutes) * tf_minutes
        slot_lbl    = dt.replace(minute=floored_min, second=0).strftime("%d-%b %H:%M")
        ce_buckets[slot_lbl]    = pt["ce_ltp"]
        pe_buckets[slot_lbl]    = pt["pe_ltp"]
        oi_ce_buckets[slot_lbl] = pt["ce_oi"]
        oi_pe_buckets[slot_lbl] = pt["pe_oi"]
    slots = list(ce_buckets.keys())
    return [{"t": s, "ce_ltp": ce_buckets[s], "pe_ltp": pe_buckets[s],
             "ce_oi": oi_ce_buckets[s], "pe_oi": oi_pe_buckets[s]} for s in slots]


def buildup_signal(price_pct: float | None, oi_pct: float | None, side: str) -> dict:
    """
    Returns dict with signal name, emoji, strength, and action text.
    Thresholds: price ≥ ±5%,  OI ≥ ±5%

    For CE (calls):
      price ↑5% + OI ↑5%  → Long Build-up   (bulls adding longs)
      price ↑5% + OI ↓5%  → Short Covering  (shorts buying back)
      price ↓5% + OI ↑5%  → Short Build-up  (bears adding shorts)
      price ↓5% + OI ↓5%  → Long Unwinding  (bulls exiting)

    For PE (puts) the same price logic applies to the put premium itself.
    """
    THRESH = 5.0
    if price_pct is None or oi_pct is None:
        return {"name": "Neutral", "emoji": "—", "strength": "", "action": "Insufficient data", "cls": "bu-neutral"}
    p_up   = price_pct >=  THRESH
    p_dn   = price_pct <= -THRESH
    oi_up  = oi_pct    >=  THRESH
    oi_dn  = oi_pct    <= -THRESH

    if p_up and oi_up:
        return {"name": "Long Build-up",  "emoji": "✅",    "strength": "Strong",       "action": "Buy & hold",      "cls": "bu-long"}
    if p_up and oi_dn:
        return {"name": "Short Covering", "emoji": "🔥🔥", "strength": "Very Strong",  "action": "Buy fast",        "cls": "bu-cover"}
    if p_dn and oi_up:
        return {"name": "Short Build-up", "emoji": "🔻",   "strength": "Bearish",      "action": "Avoid / hedge",   "cls": "bu-short"}
    if p_dn and oi_dn:
        return {"name": "Long Unwinding", "emoji": "⚠️",   "strength": "Caution",      "action": "Exit longs",      "cls": "bu-unwind"}
    # Partial signals (one leg near threshold)
    if p_up:
        return {"name": "Price Rising",   "emoji": "📈",   "strength": "Watch",        "action": "OI confirm pending", "cls": "bu-watch"}
    if oi_up:
        return {"name": "OI Rising",      "emoji": "📊",   "strength": "Watch",        "action": "Price confirm pending", "cls": "bu-watch"}
    return {"name": "Neutral",            "emoji": "—",    "strength": "",             "action": "No clear signal", "cls": "bu-neutral"}


def get_buildup_for_strike(strike: int, tf_minutes: int) -> dict:
    """
    Compute buildup signals for CE and PE for a given strike and TF.
    Compares the most recent bar vs the bar `tf_minutes` ago.
    Returns {"ce": buildup_dict, "pe": buildup_dict, "ce_price_pct", "pe_price_pct", "ce_oi_pct", "pe_oi_pct"}
    """
    bars = resample_price_oi(strike, tf_minutes)
    if len(bars) < 2:
        neutral = buildup_signal(None, None, "ce")
        return {"ce": neutral, "pe": neutral,
                "ce_price_pct": None, "pe_price_pct": None,
                "ce_oi_pct": None, "pe_oi_pct": None}

    cur  = bars[-1]
    prev = bars[-2]   # one TF bar ago

    def pct(cur_val, prev_val):
        if prev_val is None or prev_val == 0:
            return None
        return round((cur_val - prev_val) / prev_val * 100, 2)

    ce_price_pct = pct(cur["ce_ltp"], prev["ce_ltp"])
    pe_price_pct = pct(cur["pe_ltp"], prev["pe_ltp"])
    ce_oi_pct    = pct(cur["ce_oi"],  prev["ce_oi"])
    pe_oi_pct    = pct(cur["pe_oi"],  prev["pe_oi"])

    return {
        "ce":           buildup_signal(ce_price_pct, ce_oi_pct, "ce"),
        "pe":           buildup_signal(pe_price_pct, pe_oi_pct, "pe"),
        "ce_price_pct": ce_price_pct,
        "pe_price_pct": pe_price_pct,
        "ce_oi_pct":    ce_oi_pct,
        "pe_oi_pct":    pe_oi_pct,
    }


# ═══════════════════════════════════════════════════════════
#  INSTRUMENTS  (unchanged from v5)
# ═══════════════════════════════════════════════════════════

def load_nifty_instruments():
    global _instrument_map, _token_map, _sym_to_token
    instruments = kite.instruments("NFO")
    today = date.today()
    imap, tmap, stmap, expset = {}, {}, {}, set()
    for inst in instruments:
        if inst["name"] != "NIFTY":
            continue
        if inst["instrument_type"] not in ("CE", "PE"):
            continue
        exp = inst["expiry"]
        if hasattr(exp, "date"):
            exp = exp.date()
        if exp < today:
            continue
        key = (int(inst["strike"]), inst["instrument_type"], exp)
        sym = f"NFO:{inst['tradingsymbol']}"
        imap[key]  = sym
        tmap[key]  = int(inst["instrument_token"])
        stmap[sym] = int(inst["instrument_token"])
        expset.add(exp)
    _instrument_map = imap
    _token_map      = tmap
    _sym_to_token   = stmap
    print(f"  [INST] Loaded {len(imap)} NIFTY NFO contracts, {len(expset)} expiries")
    return sorted(expset)


def get_nearest_expiry():
    global _nearest_expiry
    expiries = load_nifty_instruments()
    if not expiries:
        raise RuntimeError("No NIFTY NFO instruments found")
    _nearest_expiry = expiries[0]
    return _nearest_expiry


def zerodha_expiry_code(expiry):
    MONTH_CODE = {1:"1",2:"2",3:"3",4:"4",5:"5",
                  6:"6",7:"7",8:"8",9:"9",
                  10:"O",11:"N",12:"D"}
    import calendar
    yr, mo, dy = expiry.year, expiry.month, expiry.day
    last_day       = calendar.monthrange(yr, mo)[1]
    expiry_weekday = expiry.weekday()
    is_monthly = True
    for d in range(dy + 1, last_day + 1):
        if date(yr, mo, d).weekday() == expiry_weekday:
            is_monthly = False; break
    yy = str(yr)[-2:]
    if is_monthly:
        return f"{yy}{expiry.strftime('%b').upper()}"
    else:
        return f"{yy}{MONTH_CODE[mo]}{dy:02d}"


def opt_sym(strike, kind, expiry):
    key = (int(strike), kind, expiry)
    if key in _instrument_map:
        return _instrument_map[key]
    exp_code = zerodha_expiry_code(expiry)
    return f"NFO:NIFTY{exp_code}{int(strike)}{kind}"


def get_spot():
    data = kite.ltp(["NSE:NIFTY 50"])
    return list(data.values())[0]["last_price"]


def round_atm(price):
    return round(price / STRIKE_GAP) * STRIKE_GAP


# ═══════════════════════════════════════════════════════════
#  OI DELTA
# ═══════════════════════════════════════════════════════════

def oi_change(sym, cur):
    prev = _prev_oi.get(sym)
    if prev is None:
        return None, None
    diff = cur - prev
    if prev == 0:
        return diff, None
    pct = round((diff / prev) * 100, 2)
    return diff, pct


# ═══════════════════════════════════════════════════════════
#  HISTORICAL OI  (unchanged from v5)
# ═══════════════════════════════════════════════════════════

def _fetch_historical_for_strike(strike: int) -> dict:
    expiry = _nearest_expiry
    if expiry is None:
        raise RuntimeError("Expiry not loaded yet")
    ce_tok = _token_map.get((strike, "CE", expiry))
    pe_tok = _token_map.get((strike, "PE", expiry))
    if not ce_tok or not pe_tok:
        raise RuntimeError(f"Tokens not found for strike {strike}")
    from_dt = date.today() - timedelta(days=5)
    to_dt   = date.today()
    ce_raw = kite.historical_data(ce_tok, from_dt, to_dt, "3minute", oi=True)
    pe_raw = kite.historical_data(pe_tok, from_dt, to_dt, "3minute", oi=True)
    ts_map: dict[str, dict] = {}
    for c in ce_raw:
        lbl = c["date"].strftime("%d-%b %H:%M")
        ts_map.setdefault(lbl, {})["ce"] = c.get("oi", 0)
    for c in pe_raw:
        lbl = c["date"].strftime("%d-%b %H:%M")
        ts_map.setdefault(lbl, {})["pe"] = c.get("oi", 0)
    sorted_ts = sorted(ts_map.keys())
    return {"ts": sorted_ts,
            "ce": [ts_map[t].get("ce", 0) for t in sorted_ts],
            "pe": [ts_map[t].get("pe", 0) for t in sorted_ts]}


def _fetch_historical_total() -> dict:
    expiry = _nearest_expiry
    atm    = oi_data.get("atm")
    if expiry is None or not atm:
        raise RuntimeError("Expiry or ATM not loaded yet")
    from_dt = date.today() - timedelta(days=5)
    to_dt   = date.today()
    ts_ce: dict[str, int] = {}
    ts_pe: dict[str, int] = {}
    for offset in range(-OTM_DEPTH, OTM_DEPTH + 1):
        strike = atm + offset * STRIKE_GAP
        ce_tok = _token_map.get((int(strike), "CE", expiry))
        pe_tok = _token_map.get((int(strike), "PE", expiry))
        if not ce_tok or not pe_tok:
            continue
        try:
            ce_raw = kite.historical_data(ce_tok, from_dt, to_dt, "3minute", oi=True)
            pe_raw = kite.historical_data(pe_tok, from_dt, to_dt, "3minute", oi=True)
        except Exception as e:
            print(f"  [HIST-TOTAL] Failed for strike {strike}: {e}"); continue
        for c in ce_raw:
            lbl = c["date"].strftime("%d-%b %H:%M")
            ts_ce[lbl] = ts_ce.get(lbl, 0) + (c.get("oi") or 0)
        for c in pe_raw:
            lbl = c["date"].strftime("%d-%b %H:%M")
            ts_pe[lbl] = ts_pe.get(lbl, 0) + (c.get("oi") or 0)
    all_ts = sorted(set(ts_ce) | set(ts_pe))
    return {"ts": all_ts,
            "ce": [ts_ce.get(t, 0) for t in all_ts],
            "pe": [ts_pe.get(t, 0) for t in all_ts]}


def _fetch_historical_otm() -> dict:
    expiry = _nearest_expiry
    atm    = oi_data.get("atm")
    if expiry is None or not atm:
        raise RuntimeError("Expiry or ATM not loaded yet")
    from_dt = date.today() - timedelta(days=5)
    to_dt   = date.today()
    ts_ce: dict[str, int] = {}
    ts_pe: dict[str, int] = {}
    for offset in range(-OTM_DEPTH, OTM_DEPTH + 1):
        if offset == 0:
            continue
        strike = atm + offset * STRIKE_GAP
        ce_tok = _token_map.get((int(strike), "CE", expiry))
        pe_tok = _token_map.get((int(strike), "PE", expiry))
        if not ce_tok or not pe_tok:
            continue
        try:
            ce_raw = kite.historical_data(ce_tok, from_dt, to_dt, "3minute", oi=True)
            pe_raw = kite.historical_data(pe_tok, from_dt, to_dt, "3minute", oi=True)
        except Exception as e:
            print(f"  [HIST-OTM] Failed for strike {strike}: {e}"); continue
        if offset > 0:
            for c in ce_raw:
                lbl = c["date"].strftime("%d-%b %H:%M")
                ts_ce[lbl] = ts_ce.get(lbl, 0) + (c.get("oi") or 0)
        if offset < 0:
            for c in pe_raw:
                lbl = c["date"].strftime("%d-%b %H:%M")
                ts_pe[lbl] = ts_pe.get(lbl, 0) + (c.get("oi") or 0)
    all_ts = sorted(set(ts_ce) | set(ts_pe))
    return {"ts": all_ts,
            "ce": [ts_ce.get(t, 0) for t in all_ts],
            "pe": [ts_pe.get(t, 0) for t in all_ts]}


def _merge_hist_total_with_live(hist: dict) -> dict:
    ts_map = {t: {"ce": hist["ce"][i], "pe": hist["pe"][i]} for i, t in enumerate(hist["ts"])}
    live_ts: dict[str, dict] = {}
    for sk_pts in oi_history.values():
        for pt in sk_pts:
            t = pt["t"]
            if t not in live_ts:
                live_ts[t] = {"ce": 0, "pe": 0}
            live_ts[t]["ce"] += pt["ce"]
            live_ts[t]["pe"] += pt["pe"]
    for t, v in live_ts.items():
        ts_map[t] = v
    sorted_ts = sorted(ts_map.keys())
    return {"ts": sorted_ts,
            "ce": [ts_map[t]["ce"] for t in sorted_ts],
            "pe": [ts_map[t]["pe"] for t in sorted_ts]}


def _merge_hist_with_live(strike: int, hist: dict) -> dict:
    ts_map = {t: {"ce": hist["ce"][i], "pe": hist["pe"][i]} for i, t in enumerate(hist["ts"])}
    for pt in oi_history.get(strike, []):
        ts_map[pt["t"]] = {"ce": pt["ce"], "pe": pt["pe"]}
    sorted_ts = sorted(ts_map.keys())
    return {"ts": sorted_ts,
            "ce": [ts_map[t]["ce"] for t in sorted_ts],
            "pe": [ts_map[t]["pe"] for t in sorted_ts]}


def _merge_hist_otm_with_live(hist: dict) -> dict:
    ts_map = {t: {"ce": hist["ce"][i], "pe": hist["pe"][i]} for i, t in enumerate(hist["ts"])}
    live_ts: dict[str, dict] = {}
    for row in oi_data.get("rows", []):
        if row.get("offset", 0) == 0:
            continue
        sk = int(row["strike"])
        for pt in oi_history.get(sk, []):
            t = pt["t"]
            if t not in live_ts:
                live_ts[t] = {"ce": 0, "pe": 0}
            if row["offset"] > 0:
                live_ts[t]["ce"] += pt["ce"]
            if row["offset"] < 0:
                live_ts[t]["pe"] += pt["pe"]
    for t, v in live_ts.items():
        ts_map[t] = v
    sorted_ts = sorted(ts_map.keys())
    return {"ts": sorted_ts,
            "ce": [ts_map[t]["ce"] for t in sorted_ts],
            "pe": [ts_map[t]["pe"] for t in sorted_ts]}


# ═══════════════════════════════════════════════════════════
#  OI FETCH  (extended to also compute & store IV)
# ═══════════════════════════════════════════════════════════

def fetch_oi():
    global oi_data, last_oi_time, error_msg, ltp_symbols, _prev_oi
    try:
        expiry = get_nearest_expiry()
        spot   = get_spot()
        atm    = round_atm(spot)
        T      = tte_years(expiry)
        rows, syms = [], []

        for offset in range(-OTM_DEPTH, OTM_DEPTH + 1):
            strike = atm + offset * STRIKE_GAP
            ce_k   = opt_sym(strike, "CE", expiry)
            pe_k   = opt_sym(strike, "PE", expiry)
            rows.append({"offset": offset, "strike": strike,
                         "is_atm": offset == 0,
                         "ce_sym": ce_k, "pe_sym": pe_k})
            syms += [ce_k, pe_k]

        quotes   = kite.quote(syms)
        new_snap = {}
        result_rows = []
        total_ce = total_pe = 0
        ts_now = datetime.now().strftime("%d-%b %H:%M")

        for r in rows:
            ceq = quotes.get(r["ce_sym"], {})
            peq = quotes.get(r["pe_sym"], {})
            g = lambda q, k: q.get(k) or 0

            ce_oi  = g(ceq, "oi");          pe_oi  = g(peq, "oi")
            ce_tot = g(ceq, "oi_day_high"); pe_tot = g(peq, "oi_day_high")
            ce_ltp = g(ceq, "last_price");  pe_ltp = g(peq, "last_price")

            ce_diff, ce_pct = oi_change(r["ce_sym"], ce_oi)
            pe_diff, pe_pct = oi_change(r["pe_sym"], pe_oi)

            new_snap[r["ce_sym"]] = ce_oi
            new_snap[r["pe_sym"]] = pe_oi
            total_ce += ce_oi
            total_pe += pe_oi

            sk = int(r["strike"])
            if sk not in oi_history:
                oi_history[sk] = []
            if not oi_history[sk] or oi_history[sk][-1]["t"] != ts_now:
                oi_history[sk].append({"t": ts_now, "ce": ce_oi, "pe": pe_oi})

            # ── Compute IV using Black-Scholes ──────────────
            ce_iv = implied_vol(spot, sk, T, RISK_FREE, ce_ltp, 'c') if ce_ltp > 0.01 else None
            pe_iv = implied_vol(spot, sk, T, RISK_FREE, pe_ltp, 'p') if pe_ltp > 0.01 else None

            # Store IV history per strike
            if sk not in iv_history:
                iv_history[sk] = {
                    "ce": deque(maxlen=IV_HISTORY_MAXLEN),
                    "pe": deque(maxlen=IV_HISTORY_MAXLEN),
                }
            # Append or update last point if same timestamp
            for side, iv_val in [("ce", ce_iv), ("pe", pe_iv)]:
                buf = iv_history[sk][side]
                if buf and buf[-1]["t"] == ts_now:
                    buf[-1]["iv"] = iv_val
                else:
                    buf.append({"t": ts_now, "iv": iv_val})

            # Store price + OI history for buildup signals
            if sk not in price_oi_history:
                price_oi_history[sk] = deque(maxlen=IV_HISTORY_MAXLEN)
            poi_buf = price_oi_history[sk]
            if poi_buf and poi_buf[-1]["t"] == ts_now:
                poi_buf[-1].update({"ce_ltp": ce_ltp, "pe_ltp": pe_ltp,
                                    "ce_oi": ce_oi, "pe_oi": pe_oi})
            else:
                poi_buf.append({"t": ts_now, "ce_ltp": ce_ltp, "pe_ltp": pe_ltp,
                                "ce_oi": ce_oi, "pe_oi": pe_oi})

            result_rows.append({
                "offset": r["offset"], "strike": r["strike"], "is_atm": r["is_atm"],
                "ce_sym": r["ce_sym"], "pe_sym": r["pe_sym"],
                "ce_ltp": round(ce_ltp, 2), "pe_ltp": round(pe_ltp, 2),
                "ce_oi": ce_oi, "ce_oi_tot": ce_tot,
                "ce_oi_diff": ce_diff, "ce_oi_pct": ce_pct,
                "pe_oi": pe_oi, "pe_oi_tot": pe_tot,
                "pe_oi_diff": pe_diff, "pe_oi_pct": pe_pct,
                "strike_pcr": round(pe_oi / ce_oi, 4) if ce_oi else None,
                "ce_iv": ce_iv,   # current IV %
                "pe_iv": pe_iv,
            })

        _prev_oi = new_snap
        pcr = round(total_pe / total_ce, 4) if total_ce else 0
        oi_data = {
            "spot": round(spot, 2), "atm": atm,
            "expiry": expiry.strftime("%d %b %Y"),
            "rows": result_rows,
            "total_ce": total_ce, "total_pe": total_pe, "pcr": pcr,
            "ts_chart": ts_now,
        }
        last_oi_time = datetime.now().strftime("%H:%M:%S")
        error_msg    = None
        ltp_symbols  = ["NSE:NIFTY 50"] + syms
        print(f"  [OI] ATM={atm}  Spot={spot:.2f}  PCR={pcr}  ts={ts_now}")

    except Exception as e:
        import traceback; traceback.print_exc()
        error_msg = str(e)


def oi_loop():
    time.sleep(OI_INTERVAL)
    while True:
        fetch_oi()
        time.sleep(OI_INTERVAL)


# ═══════════════════════════════════════════════════════════
#  LTP FETCH
# ═══════════════════════════════════════════════════════════

def fetch_ltp():
    global ltp_data
    if not ltp_symbols: return
    try:
        raw = kite.ltp(ltp_symbols)
        ltp_data = {sym: round(v.get("last_price") or 0, 2)
                    for sym, v in raw.items()}
    except Exception:
        pass


def ltp_loop():
    while True:
        fetch_ltp()
        time.sleep(LTP_INTERVAL)


# ═══════════════════════════════════════════════════════════
#  PRICE / VWAP  (unchanged from v5)
# ═══════════════════════════════════════════════════════════

_pv_cache: dict  = {}
PV_CACHE_TTL     = 60


def _compute_vwap(candles: list) -> list:
    cum_tp_vol = 0.0; cum_vol = 0.0; out = []
    for c in candles:
        tp = (c["high"] + c["low"] + c["close"]) / 3.0
        cum_tp_vol += tp * (c["volume"] or 0); cum_vol += (c["volume"] or 0)
        out.append(round(cum_tp_vol / cum_vol, 2) if cum_vol else round(c["close"], 2))
    return out


# ═══════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/api/debug")
def api_debug():
    expiry   = _nearest_expiry
    atm      = oi_data.get("atm")
    exp_code = zerodha_expiry_code(expiry) if expiry else "?"
    sample_strikes = []
    if atm and expiry:
        for offset in range(-2, 3):
            sk = atm + offset * STRIKE_GAP
            ce_key = (int(sk), "CE", expiry); pe_key = (int(sk), "PE", expiry)
            sample_strikes.append({
                "strike": sk,
                "ce_sym": _instrument_map.get(ce_key, "NOT IN MAP"),
                "pe_sym": _instrument_map.get(pe_key, "NOT IN MAP"),
                "ce_token": _token_map.get(ce_key, "NO TOKEN"),
                "pe_token": _token_map.get(pe_key, "NO TOKEN"),
            })
    return jsonify({
        "nearest_expiry": str(expiry), "expiry_weekday": expiry.strftime("%A") if expiry else "?",
        "expiry_code": exp_code, "instrument_map_size": len(_instrument_map),
        "token_map_size": len(_token_map), "oi_error": error_msg, "atm": atm,
        "sample_strikes": sample_strikes,
    })


@app.route("/api/oi")
def api_oi():
    return jsonify({"data": oi_data, "updated_at": last_oi_time, "error": error_msg})


@app.route("/api/ltp")
def api_ltp():
    return jsonify(ltp_data)


# ── NEW: IV endpoint ──────────────────────────────────────

@app.route("/api/iv")
def api_iv():
    """
    Returns IV table for all current strikes with EMA(9) signal + buildup signal.
    Query param: ?tf=1|3|5|10  (default 5)

    Buildup signal compares current TF-bar vs previous TF-bar:
      price ≥+5% AND OI ≥+5%  → Long Build-up  ✅  Strong Buy & hold
      price ≥+5% AND OI ≤-5%  → Short Covering 🔥🔥 Very Strong Buy fast
      price ≤-5% AND OI ≥+5%  → Short Build-up 🔻  Bearish
      price ≤-5% AND OI ≤-5%  → Long Unwinding ⚠️   Caution
    """
    tf  = request.args.get("tf", default=5, type=int)
    if tf not in (1, 3, 5, 10):
        tf = 5

    rows_out = []
    for r in oi_data.get("rows", []):
        sk       = int(r["strike"])
        ce_ser   = resample_iv_history(sk, "ce", tf)
        pe_ser   = resample_iv_history(sk, "pe", tf)

        # Latest IV from the series (last element)
        cur_ce_iv   = ce_ser[-1]["iv"]   if ce_ser else r.get("ce_iv")
        cur_pe_iv   = pe_ser[-1]["iv"]   if pe_ser else r.get("pe_iv")
        cur_ce_ema9 = ce_ser[-1]["ema9"] if ce_ser else None
        cur_pe_ema9 = pe_ser[-1]["ema9"] if pe_ser else None

        def iv_signal(iv, ema9, threshold=0.5):
            if iv is None or ema9 is None:
                return "NEW"
            diff_pct = iv - ema9
            if diff_pct > threshold:
                return "HIGH"
            if diff_pct < -threshold:
                return "LOW"
            return "FLAT"

        # Buildup signal (uses price_oi_history + tf)
        bu = get_buildup_for_strike(sk, tf)

        rows_out.append({
            "strike":    sk,
            "is_atm":   r["is_atm"],
            "offset":   r["offset"],
            "ce_sym":   r.get("ce_sym", ""),   # ← for live LTP update
            "pe_sym":   r.get("pe_sym", ""),
            "ce_ltp":   r.get("ce_ltp"),
            "pe_ltp":   r.get("pe_ltp"),
            "ce_iv":    cur_ce_iv,
            "pe_iv":    cur_pe_iv,
            "ce_ema9":  cur_ce_ema9,
            "pe_ema9":  cur_pe_ema9,
            "ce_signal": iv_signal(cur_ce_iv, cur_ce_ema9),
            "pe_signal": iv_signal(cur_pe_iv, cur_pe_ema9),
            "ce_iv_pts": len(ce_ser),
            "pe_iv_pts": len(pe_ser),
            "ce_series": ce_ser[-50:],
            "pe_series": pe_ser[-50:],
            # Buildup
            "ce_buildup":       bu["ce"],
            "pe_buildup":       bu["pe"],
            "ce_price_pct":     bu["ce_price_pct"],
            "pe_price_pct":     bu["pe_price_pct"],
            "ce_oi_pct":        bu["ce_oi_pct"],
            "pe_oi_pct":        bu["pe_oi_pct"],
        })

    return jsonify({
        "rows": rows_out,
        "tf":   tf,
        "ts":   last_oi_time or "—",
        "spot": oi_data.get("spot"),
        "expiry": oi_data.get("expiry"),
    })


@app.route("/api/historical_oi")
def api_historical_oi():
    strike = request.args.get("strike", type=int)
    if not strike:
        return jsonify({"error": "strike param required"}), 400
    cached = _hist_cache.get(strike)
    if cached and (datetime.now() - cached["fetched_at"]).total_seconds() < HIST_CACHE_TTL:
        merged = _merge_hist_with_live(strike, cached)
        return jsonify({**merged, "source": "historical+live"})
    try:
        hist = _fetch_historical_for_strike(strike)
        _hist_cache[strike] = {**hist, "fetched_at": datetime.now()}
        merged = _merge_hist_with_live(strike, hist)
        return jsonify({**merged, "source": "historical+live"})
    except Exception as e:
        print(f"  [HIST] Failed for strike {strike}: {e}")
        live = oi_history.get(strike, [])
        return jsonify({
            "ts": [p["t"] for p in live], "ce": [p["ce"] for p in live],
            "pe": [p["pe"] for p in live], "source": "live_only", "error": str(e),
        })


@app.route("/api/historical_oi_total")
def api_historical_oi_total():
    cached = _hist_total_cache.get(HIST_TOTAL_CACHE_KEY)
    if cached and (datetime.now() - cached["fetched_at"]).total_seconds() < HIST_CACHE_TTL:
        merged = _merge_hist_total_with_live(cached)
        return jsonify({**merged, "source": "historical+live"})
    try:
        hist = _fetch_historical_total()
        _hist_total_cache[HIST_TOTAL_CACHE_KEY] = {**hist, "fetched_at": datetime.now()}
        merged = _merge_hist_total_with_live(hist)
        return jsonify({**merged, "source": "historical+live"})
    except Exception as e:
        import traceback; traceback.print_exc()
        merged = _merge_hist_total_with_live({"ts": [], "ce": [], "pe": []})
        return jsonify({**merged, "source": "live_only", "error": str(e)})


@app.route("/api/historical_oi_otm")
def api_historical_oi_otm():
    cached = _hist_otm_cache.get(HIST_OTM_CACHE_KEY)
    if cached and (datetime.now() - cached["fetched_at"]).total_seconds() < HIST_CACHE_TTL:
        merged = _merge_hist_otm_with_live(cached)
        return jsonify({**merged, "source": "historical+live"})
    try:
        hist = _fetch_historical_otm()
        _hist_otm_cache[HIST_OTM_CACHE_KEY] = {**hist, "fetched_at": datetime.now()}
        merged = _merge_hist_otm_with_live(hist)
        return jsonify({**merged, "source": "historical+live"})
    except Exception as e:
        import traceback; traceback.print_exc()
        merged = _merge_hist_otm_with_live({"ts": [], "ce": [], "pe": []})
        return jsonify({**merged, "source": "live_only", "error": str(e)})


@app.route("/api/debug_tokens")
def api_debug_tokens():
    strike = request.args.get("strike", type=int) or (oi_data.get("atm") if oi_data else None)
    result = {"nearest_expiry": str(_nearest_expiry),
              "token_map_size": len(_token_map),
              "instrument_map_size": len(_instrument_map)}
    if strike and _nearest_expiry:
        ce_key = (int(strike), "CE", _nearest_expiry)
        pe_key = (int(strike), "PE", _nearest_expiry)
        result.update({
            "strike": strike,
            "ce_token": _token_map.get(ce_key, "NOT FOUND"),
            "pe_token": _token_map.get(pe_key, "NOT FOUND"),
            "ce_sym": _instrument_map.get(ce_key, "NOT FOUND"),
            "pe_sym": _instrument_map.get(pe_key, "NOT FOUND"),
            "sample_keys": [(str(k), v) for k, v in list(_token_map.items())[:5]],
        })
    return jsonify(result)


@app.route("/api/price_vwap")
def api_price_vwap():
    tf = request.args.get("tf", default=1, type=int)
    if tf not in (1, 3, 5, 10):
        tf = 1
    ce_sym = request.args.get("ce_sym", "").strip()
    pe_sym = request.args.get("pe_sym", "").strip()
    if not ce_sym or not pe_sym:
        strike = request.args.get("strike", type=int)
        if not strike:
            return jsonify({"error": "Provide ce_sym & pe_sym (or strike)"}), 400
        expiry = _nearest_expiry
        if expiry is None:
            return jsonify({"error": "Expiry not loaded yet"}), 503
        ce_sym = _instrument_map.get((int(strike), "CE", expiry), "")
        pe_sym = _instrument_map.get((int(strike), "PE", expiry), "")
        if not ce_sym or not pe_sym:
            exp_code = zerodha_expiry_code(expiry)
            return jsonify({"error": f"Symbol not found for strike {strike} expiry {expiry} "
                                      f"(tried NIFTY{exp_code}{strike}CE/PE). "
                                      f"Check /api/debug_tokens?strike={strike}"}), 404
    ce_tok = _sym_to_token.get(ce_sym)
    pe_tok = _sym_to_token.get(pe_sym)
    if not ce_tok or not pe_tok:
        return jsonify({"error": f"Token not found for {ce_sym} / {pe_sym}. "
                                  f"Sym-to-token map has {len(_sym_to_token)} entries."}), 404
    cache_key = (ce_sym, pe_sym, tf)
    cached = _pv_cache.get(cache_key)
    if cached and (datetime.now() - cached["fetched_at"]).total_seconds() < PV_CACHE_TTL:
        return jsonify({**cached["data"], "tf": tf, "cached": True, "ce_sym": ce_sym, "pe_sym": pe_sym})
    today_d = date.today()
    tf_str = "minute" if tf == 1 else f"{tf}minute"
    try:
        ce_raw = kite.historical_data(ce_tok, today_d, today_d, tf_str)
        pe_raw = kite.historical_data(pe_tok, today_d, today_d, tf_str)

        def full_day_slots(tf_mins):
            from datetime import time as dtime
            slots = []; h, m = 9, 15
            while (h, m) <= (15, 30):
                slots.append(f"{h:02d}:{m:02d}")
                m += tf_mins
                if m >= 60:
                    h += m // 60; m = m % 60
            return slots

        def to_series(candles, tf_mins):
            candle_map = {c["date"].strftime("%H:%M"): c for c in candles}
            slots = full_day_slots(tf_mins)
            cum_tp_vol = 0.0; cum_vol = 0.0
            ts_out, price_out, vwap_out = [], [], []
            for slot in slots:
                ts_out.append(slot)
                if slot in candle_map:
                    c = candle_map[slot]
                    tp = (c["high"] + c["low"] + c["close"]) / 3.0
                    cum_tp_vol += tp * (c.get("volume") or 0); cum_vol += (c.get("volume") or 0)
                    price_out.append(round(c["close"], 2))
                    vwap_out.append(round(cum_tp_vol / cum_vol, 2) if cum_vol else round(c["close"], 2))
                else:
                    price_out.append(None); vwap_out.append(None)
            return {"ts": ts_out, "price": price_out, "vwap": vwap_out}

        data = {"ce": to_series(ce_raw, tf), "pe": to_series(pe_raw, tf)}
        _pv_cache[cache_key] = {"fetched_at": datetime.now(), "data": data}
        return jsonify({**data, "tf": tf, "ce_sym": ce_sym, "pe_sym": pe_sym})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
#  HTML  — v6 with IV Table
# ═══════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="night">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NIFTY OI Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Sora:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
/* ══ THEMES ══════════════════════════════════════════════ */
:root,[data-theme="night"]{
  --bg:#05080f; --s1:#090e1a; --s2:#0c1220; --brd:#162035;
  --ce:#00d4a0; --pe:#ff3f6c; --gold:#fbbf24;
  --text:#dce8f8; --muted:#4e6180;
  --hover:rgba(255,255,255,.018); --scan:rgba(0,0,0,.035);
  --atm-bg:rgba(251,191,36,.06); --atm-bd:rgba(251,191,36,.25);
  --shadow:0 2px 18px rgba(0,0,0,.55);
  --btn-bg:#0c1220; --btn-bd:#1e3a5f; --btn-cl:#6bacd6;
  --err-bg:#180508; --err-bd:#5a1020; --err-cl:#ff7a8a;
  --chart-bg:#07101e; --gc:rgba(22,32,53,.9);
  --tt:#090e1a; --tt-brd:#1e3a5f;
  --tt-title:#e8f4ff; --tt-body:#9bbcda;
  --ce-fill:rgba(0,212,160,.08); --pe-fill:rgba(255,63,108,.07);
  --sel-bg:#0c1220; --sel-bd:#1e3a5f; --sel-arr:#6bacd6;
  --iv-high-bg:rgba(255,63,108,.10); --iv-high-cl:#ff6b8e;
  --iv-low-bg:rgba(0,212,160,.10);   --iv-low-cl:#00d4a0;
  --iv-flat-bg:rgba(78,97,128,.12);  --iv-flat-cl:#7a90b0;
  --iv-new-bg:rgba(107,124,255,.10); --iv-new-cl:#8b9eff;
  /* buildup signals */
  --bu-long-bg:rgba(0,212,160,.12);    --bu-long-cl:#00d4a0;   --bu-long-bd:rgba(0,212,160,.35);
  --bu-cover-bg:rgba(255,140,0,.14);   --bu-cover-cl:#ffaa33;  --bu-cover-bd:rgba(255,140,0,.4);
  --bu-short-bg:rgba(255,63,108,.12);  --bu-short-cl:#ff6b8e;  --bu-short-bd:rgba(255,63,108,.35);
  --bu-unwind-bg:rgba(251,191,36,.10); --bu-unwind-cl:#fbbf24; --bu-unwind-bd:rgba(251,191,36,.3);
  --bu-watch-bg:rgba(107,124,255,.10); --bu-watch-cl:#8b9eff;  --bu-watch-bd:rgba(107,124,255,.3);
  --bu-neutral-bg:rgba(78,97,128,.08); --bu-neutral-cl:#4e6180;--bu-neutral-bd:rgba(78,97,128,.2);
}
[data-theme="day"]{
  --bg:#f0f4fa; --s1:#fff; --s2:#e8eef8; --brd:#c8d6e8;
  --ce:#007a5c; --pe:#b51532; --gold:#9a6700;
  --text:#0f1c2e; --muted:#6b82a0;
  --hover:rgba(0,0,0,.025); --scan:transparent;
  --atm-bg:rgba(154,103,0,.05); --atm-bd:rgba(154,103,0,.28);
  --shadow:0 2px 12px rgba(0,0,0,.09);
  --btn-bg:#e8eef8; --btn-bd:#b0c0d8; --btn-cl:#1a4a90;
  --err-bg:#fff0f2; --err-bd:#f5b8c2; --err-cl:#b51532;
  --chart-bg:#f7faff; --gc:rgba(200,214,232,.8);
  --tt:rgba(240,244,250,.98); --tt-brd:#b0c0d8;
  --tt-title:#0f1c2e; --tt-body:#2a4a6a;
  --ce-fill:rgba(0,122,92,.08); --pe-fill:rgba(181,21,50,.07);
  --sel-bg:#f0f4fa; --sel-bd:#b0c0d8; --sel-arr:#1a4a90;
  --iv-high-bg:rgba(181,21,50,.08);  --iv-high-cl:#b51532;
  --iv-low-bg:rgba(0,122,92,.08);    --iv-low-cl:#007a5c;
  --iv-flat-bg:rgba(107,130,160,.10);--iv-flat-cl:#4a6080;
  --iv-new-bg:rgba(26,74,144,.08);   --iv-new-cl:#1a4a90;
  --bu-long-bg:rgba(0,122,92,.10);    --bu-long-cl:#007a5c;   --bu-long-bd:rgba(0,122,92,.35);
  --bu-cover-bg:rgba(200,100,0,.10);  --bu-cover-cl:#c86400;  --bu-cover-bd:rgba(200,100,0,.35);
  --bu-short-bg:rgba(181,21,50,.10);  --bu-short-cl:#b51532;  --bu-short-bd:rgba(181,21,50,.35);
  --bu-unwind-bg:rgba(154,103,0,.10); --bu-unwind-cl:#9a6700; --bu-unwind-bd:rgba(154,103,0,.3);
  --bu-watch-bg:rgba(26,74,144,.08);  --bu-watch-cl:#1a4a90;  --bu-watch-bd:rgba(26,74,144,.25);
  --bu-neutral-bg:rgba(100,120,150,.06);--bu-neutral-cl:#6b82a0;--bu-neutral-bd:rgba(100,120,150,.2);
}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:var(--bg);color:var(--text);
  font-family:'Sora',sans-serif;min-height:100vh;overflow-x:hidden;
  transition:background .3s,color .3s;
}
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,var(--scan) 2px,var(--scan) 4px);
}
.wrap{max-width:1800px;margin:0 auto;padding:20px 22px}

/* ── TOPBAR ─────────────────────────────────────────────── */
.topbar{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
  gap:14px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--brd);
}
.brand{display:flex;align-items:center;gap:12px}
.brand-ico{
  width:40px;height:40px;background:linear-gradient(135deg,var(--s2),var(--s1));
  border:1px solid var(--btn-bd);border-radius:10px;
  display:grid;place-items:center;box-shadow:var(--shadow);
}
.brand-ico svg{width:20px;height:20px;stroke:var(--btn-cl)}
.brand-name{font-size:1.25rem;font-weight:800;letter-spacing:-.5px}
.brand-name em{font-style:normal;color:var(--ce)}
.topbar-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap}

.theme-btn{
  cursor:pointer;background:var(--btn-bg);border:1px solid var(--btn-bd);
  color:var(--btn-cl);border-radius:50px;padding:6px 14px;
  font-size:.7rem;font-family:'IBM Plex Mono',monospace;font-weight:600;
  display:flex;align-items:center;gap:6px;letter-spacing:.06em;
  transition:background .2s;user-select:none;
}
.theme-btn:hover{opacity:.8}
.chips{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.chip{
  font-family:'IBM Plex Mono',monospace;font-size:.66rem;
  padding:4px 11px;border-radius:50px;
  border:1px solid var(--brd);background:var(--s1);color:var(--muted);
  display:flex;align-items:center;gap:6px;
}
.chip b{color:var(--text)}
.dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.dot-l{background:var(--ce);box-shadow:0 0 7px var(--ce);animation:blink 1.4s ease infinite}
.dot-o{background:var(--gold)}.dot-e{background:#6d7eff}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

.ring-wrap{display:flex;align-items:center;gap:8px}
.ring{position:relative;width:40px;height:40px;flex-shrink:0}
.ring svg{transform:rotate(-90deg)}
.rg{fill:none;stroke:var(--brd);stroke-width:3.5}
.rf{fill:none;stroke:var(--ce);stroke-width:3.5;stroke-linecap:round;
  stroke-dasharray:100.5;stroke-dashoffset:0;transition:stroke-dashoffset 1s linear}
.rlbl{position:absolute;inset:0;display:grid;place-items:center;
  font-family:'IBM Plex Mono',monospace;font-size:.58rem;font-weight:700;color:var(--ce)}
.ring-txt{font-size:.65rem;color:var(--muted);font-family:'IBM Plex Mono',monospace}

/* ── CARDS ───────────────────────────────────────────────── */
.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:11px;margin-bottom:18px}
@media(max-width:1100px){.cards{grid-template-columns:repeat(3,1fr)}}
@media(max-width:600px) {.cards{grid-template-columns:repeat(2,1fr)}}
.card{
  background:var(--s1);border:1px solid var(--brd);border-radius:8px;
  padding:14px 15px;position:relative;overflow:hidden;box-shadow:var(--shadow);
  transition:background .3s;
}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:8px 8px 0 0;opacity:.8}
.c-spot::before{background:var(--gold)}.c-atm::before{background:var(--gold)}
.c-ce::before{background:var(--ce)}.c-pe::before{background:var(--pe)}
.c-pcr::before{background:linear-gradient(90deg,var(--ce),var(--pe))}
.c-exp::before{background:#6d7eff}
.card-lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  margin-bottom:5px;display:flex;align-items:center;gap:6px}
.card-val{font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:1.25rem;line-height:1.1}
.v-spot,.v-atm{color:var(--gold)}.v-ce{color:var(--ce)}.v-pe{color:var(--pe)}
.v-pcr{color:var(--gold)}.v-exp{color:#8b9eff;font-size:.92rem}
.card-sub{font-size:.6rem;color:var(--muted);margin-top:5px;font-family:'IBM Plex Mono',monospace}
.live-tag{
  font-size:.54rem;background:rgba(0,212,160,.1);border:1px solid rgba(0,212,160,.3);
  color:var(--ce);padding:1px 6px;border-radius:50px;font-family:'IBM Plex Mono',monospace;
}
[data-theme="day"] .live-tag{background:rgba(0,122,92,.1);border-color:rgba(0,122,92,.3)}

.flash-u{animation:fU .45s ease}
.flash-d{animation:fD .45s ease}
@keyframes fU{0%{background:rgba(0,212,160,.2)}100%{background:transparent}}
@keyframes fD{0%{background:rgba(255,63,108,.2)}100%{background:transparent}}

/* ── PCR BAR ─────────────────────────────────────────────── */
.pcr-wrap{
  background:var(--s1);border:1px solid var(--brd);border-radius:8px;
  padding:15px 18px;margin-bottom:18px;box-shadow:var(--shadow);transition:background .3s;
}
.pcr-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.pcr-ttl{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.pcr-num{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:700}
.pcr-track{height:7px;background:var(--s2);border-radius:50px;overflow:hidden;border:1px solid var(--brd)}
.pcr-fill{height:100%;border-radius:50px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.pcr-axis{display:flex;justify-content:space-between;margin-top:5px;
  font-size:.58rem;color:var(--muted);font-family:'IBM Plex Mono',monospace}
.pcr-sent{font-size:.76rem;font-weight:600;margin-top:8px;letter-spacing:.04em}

/* ── OI TABLE ────────────────────────────────────────────── */
.tbl-card{
  background:var(--s1);border:1px solid var(--brd);border-radius:8px;
  overflow:hidden;margin-bottom:18px;box-shadow:var(--shadow);transition:background .3s;
}
.tbl-hd{
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;
  gap:8px;padding:11px 16px;border-bottom:1px solid var(--brd);background:var(--s2);
}
.tbl-title{font-size:.8rem;font-weight:700;letter-spacing:.04em}
.tbl-legend{display:flex;gap:16px;font-size:.66rem;font-family:'IBM Plex Mono',monospace;flex-wrap:wrap}

table{width:100%;border-collapse:collapse}
thead tr th{
  padding:8px 10px;font-size:.6rem;text-transform:uppercase;letter-spacing:.09em;
  font-family:'IBM Plex Mono',monospace;font-weight:500;
  background:var(--s2);border-bottom:1px solid var(--brd);white-space:nowrap;
}
.grp-ce{background:rgba(0,212,160,.06)!important;color:var(--ce)!important;
  text-align:center;border-bottom:1px solid rgba(0,212,160,.2)!important;font-weight:700}
.grp-pe{background:rgba(255,63,108,.06)!important;color:var(--pe)!important;
  text-align:center;border-bottom:1px solid rgba(255,63,108,.2)!important;font-weight:700}
.grp-mid{background:rgba(251,191,36,.05)!important;color:var(--gold)!important;
  text-align:center;border-left:1px solid var(--brd);border-right:1px solid var(--brd);font-weight:700}
[data-theme="day"] .grp-ce{background:rgba(0,122,92,.06)!important}
[data-theme="day"] .grp-pe{background:rgba(181,21,50,.06)!important}
.sh-ce{text-align:right;color:var(--muted)!important;border-top:2px solid rgba(0,212,160,.3)}
.sh-pe{text-align:left; color:var(--muted)!important;border-top:2px solid rgba(255,63,108,.3)}
.sh-mid{text-align:center;color:var(--gold)!important;
  border-left:1px solid var(--brd);border-right:1px solid var(--brd);
  border-top:2px solid rgba(251,191,36,.3)}
[data-theme="day"] .sh-ce{border-top-color:rgba(0,122,92,.4)}
[data-theme="day"] .sh-pe{border-top-color:rgba(181,21,50,.4)}

tbody tr{border-bottom:1px solid var(--brd);transition:background .15s}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--hover)}
tbody tr.atm-row{
  background:var(--atm-bg)!important;
  border-top:1px solid var(--atm-bd)!important;
  border-bottom:1px solid var(--atm-bd)!important;
}
tbody td{padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:.78rem;
  white-space:nowrap;vertical-align:middle;color:var(--text)}
.td-ce{text-align:right}.td-pe{text-align:left}
.td-mid{text-align:center;border-left:1px solid var(--brd);
  border-right:1px solid var(--brd);background:rgba(251,191,36,.03)}
.strike-lbl{font-size:.88rem;font-weight:700;display:block}
.strike-offset{font-size:.58rem;color:var(--muted);display:block;margin-top:2px}
.atm-badge{
  display:inline-block;font-size:.52rem;font-weight:700;
  background:rgba(251,191,36,.15);color:var(--gold);
  border:1px solid rgba(251,191,36,.4);border-radius:50px;
  padding:1px 7px;vertical-align:middle;margin-left:5px;
  font-family:'IBM Plex Mono',monospace;
}
.pcr-cell{min-width:80px;text-align:center}
.pcr-s{display:inline-block;font-size:.68rem;font-weight:700;
  padding:3px 8px;border-radius:50px;font-family:'IBM Plex Mono',monospace}
.pcr-bull{background:rgba(0,212,160,.1);color:var(--ce);border:1px solid rgba(0,212,160,.3)}
.pcr-bear{background:rgba(255,63,108,.1);color:var(--pe);border:1px solid rgba(255,63,108,.3)}
.pcr-neut{background:rgba(251,191,36,.1);color:var(--gold);border:1px solid rgba(251,191,36,.3)}
[data-theme="day"] .pcr-bull{background:rgba(0,122,92,.1);border-color:rgba(0,122,92,.3)}
[data-theme="day"] .pcr-bear{background:rgba(181,21,50,.1);border-color:rgba(181,21,50,.3)}

.oi-cell{position:relative;min-width:90px}
.oi-bar{position:absolute;top:0;bottom:0;opacity:.13;border-radius:3px;
  pointer-events:none;transition:width .5s ease}
.oi-bar-ce{right:0;background:var(--ce)}.oi-bar-pe{left:0;background:var(--pe)}
.oi-num{position:relative;z-index:1}
.ltp-val{font-weight:600}.ltp-ce{color:var(--ce)}.ltp-pe{color:var(--pe)}
.diff-p{color:var(--ce);font-size:.73rem;font-weight:600}
.diff-n{color:var(--pe);font-size:.73rem;font-weight:600}
.diff-z{color:var(--muted);font-size:.73rem}
.pct-pill{display:inline-block;font-size:.58rem;font-weight:700;
  padding:2px 6px;border-radius:50px;font-family:'IBM Plex Mono',monospace}
.pill-p  {background:rgba(0,212,160,.12);color:var(--ce);border:1px solid rgba(0,212,160,.28)}
.pill-n  {background:rgba(255,63,108,.12);color:var(--pe);border:1px solid rgba(255,63,108,.28)}
.pill-z  {background:rgba(78,97,128,.12); color:var(--muted);border:1px solid rgba(78,97,128,.28)}
.pill-new{background:rgba(107,124,255,.12);color:#8b9eff;border:1px solid rgba(107,124,255,.28)}
[data-theme="day"] .pill-p{background:rgba(0,122,92,.1);border-color:rgba(0,122,92,.3)}
[data-theme="day"] .pill-n{background:rgba(181,21,50,.1);border-color:rgba(181,21,50,.3)}

tfoot td{padding:9px 10px;font-family:'IBM Plex Mono',monospace;font-size:.78rem;font-weight:700;
  background:var(--s2);border-top:1px solid var(--brd)}
.tot-ce{color:var(--ce);text-align:right}.tot-pe{color:var(--pe);text-align:left}
.tot-mid{text-align:center;font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  border-left:1px solid var(--brd);border-right:1px solid var(--brd)}
.tot-pcr{text-align:center;font-size:.72rem;font-weight:700}

/* ══════════════════════════════════════════════════════════
   IV TABLE  — Implied Volatility + EMA(9) + Signal
══════════════════════════════════════════════════════════ */
.iv-card{
  background:var(--s1);border:1px solid var(--brd);border-radius:8px;
  overflow:hidden;margin-bottom:18px;box-shadow:var(--shadow);transition:background .3s;
}
.iv-hd{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
  gap:10px;padding:12px 16px;border-bottom:1px solid var(--brd);background:var(--s2);
}
.iv-title{font-size:.86rem;font-weight:700;letter-spacing:.04em;
  display:flex;align-items:center;gap:8px;color:var(--text)}
.iv-title svg{width:16px;height:16px;stroke:#a78bfa;flex-shrink:0}
.iv-controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap}

/* Timeframe buttons for IV table */
.iv-tf-wrap{display:flex;align-items:center;gap:6px}
.iv-tf-lbl{font-size:.64rem;font-family:'IBM Plex Mono',monospace;color:var(--muted);margin-right:2px}
.iv-tf-btn{
  cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:.7rem;font-weight:600;
  padding:5px 12px;border-radius:6px;border:1px solid var(--brd);
  background:var(--s1);color:var(--muted);transition:all .18s;user-select:none;
}
.iv-tf-btn:hover{border-color:#a78bfa;color:var(--text)}
.iv-tf-btn.active{
  background:rgba(167,139,250,.14);border-color:#a78bfa;
  color:#c4b5fd;font-weight:700;box-shadow:0 0 0 2px rgba(167,139,250,.2);
}
[data-theme="day"] .iv-tf-btn.active{background:rgba(109,40,217,.08);border-color:#7c3aed;color:#6d28d9}

/* IV legend pills */
.iv-legend{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.ivl{display:flex;align-items:center;gap:5px;font-size:.62rem;font-family:'IBM Plex Mono',monospace;
  padding:3px 9px;border-radius:50px;}
.ivl-high{background:var(--iv-high-bg);color:var(--iv-high-cl);border:1px solid rgba(255,63,108,.25)}
.ivl-low {background:var(--iv-low-bg); color:var(--iv-low-cl); border:1px solid rgba(0,212,160,.25)}
.ivl-flat{background:var(--iv-flat-bg);color:var(--iv-flat-cl);border:1px solid rgba(78,97,128,.2)}
.ivl-new {background:var(--iv-new-bg); color:var(--iv-new-cl); border:1px solid rgba(107,124,255,.25)}

/* IV table column headers */
.iv-grp-ce{background:rgba(0,212,160,.05)!important;color:var(--ce)!important;
  text-align:center;border-bottom:1px solid rgba(0,212,160,.18)!important;font-weight:700}
.iv-grp-pe{background:rgba(255,63,108,.05)!important;color:var(--pe)!important;
  text-align:center;border-bottom:1px solid rgba(255,63,108,.18)!important;font-weight:700}
.iv-grp-mid{background:rgba(167,139,250,.05)!important;color:#a78bfa!important;
  text-align:center;border-left:1px solid var(--brd);border-right:1px solid var(--brd);font-weight:700}
.iv-sh-ce{text-align:right;color:var(--muted)!important;border-top:2px solid rgba(0,212,160,.25)}
.iv-sh-pe{text-align:left; color:var(--muted)!important;border-top:2px solid rgba(255,63,108,.25)}
.iv-sh-mid{text-align:center;color:#a78bfa!important;
  border-left:1px solid var(--brd);border-right:1px solid var(--brd);
  border-top:2px solid rgba(167,139,250,.25)}
[data-theme="day"] .iv-grp-ce{background:rgba(0,122,92,.05)!important}
[data-theme="day"] .iv-grp-pe{background:rgba(181,21,50,.05)!important}

/* IV table cells */
.iv-num{font-family:'IBM Plex Mono',monospace;font-size:.82rem;font-weight:700}
.iv-num-ce{color:var(--ce)}.iv-num-pe{color:var(--pe)}
.iv-ema{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted);margin-top:2px}
.iv-td-ce{text-align:right}.iv-td-pe{text-align:left}
.iv-td-mid{text-align:center;border-left:1px solid var(--brd);border-right:1px solid var(--brd);
  background:rgba(167,139,250,.02)}

/* IV signal badge */
.iv-sig{
  display:inline-flex;align-items:center;gap:4px;
  font-size:.64rem;font-weight:700;padding:3px 9px;border-radius:50px;
  font-family:'IBM Plex Mono',monospace;letter-spacing:.04em;white-space:nowrap;
}
.iv-sig-HIGH{background:var(--iv-high-bg);color:var(--iv-high-cl);border:1px solid rgba(255,63,108,.3)}
.iv-sig-LOW {background:var(--iv-low-bg); color:var(--iv-low-cl); border:1px solid rgba(0,212,160,.3)}
.iv-sig-FLAT{background:var(--iv-flat-bg);color:var(--iv-flat-cl);border:1px solid rgba(78,97,128,.22)}
.iv-sig-NEW {background:var(--iv-new-bg); color:var(--iv-new-cl); border:1px solid rgba(107,124,255,.28)}
[data-theme="day"] .iv-sig-HIGH{background:rgba(181,21,50,.08);color:#b51532}
[data-theme="day"] .iv-sig-LOW {background:rgba(0,122,92,.08);color:#007a5c}

/* IV sparkline canvas */
.iv-spark-wrap{display:inline-block;vertical-align:middle;margin-left:6px}
.iv-spark-wrap canvas{display:block}

/* IV diff sub-line */
.iv-diff{font-size:.62rem;font-family:'IBM Plex Mono',monospace;margin-top:2px;display:block}
.iv-diff-p{color:var(--iv-high-cl)}.iv-diff-n{color:var(--iv-low-cl)}.iv-diff-z{color:var(--muted)}

/* ── BUILDUP SIGNAL BADGE ──────────────────────────────── */
.bu-badge{
  display:inline-flex;align-items:center;gap:5px;
  font-size:.66rem;font-weight:700;padding:4px 10px;border-radius:6px;
  font-family:'IBM Plex Mono',monospace;letter-spacing:.02em;
  white-space:nowrap;line-height:1.3;
}
.bu-long   {background:var(--bu-long-bg);   color:var(--bu-long-cl);   border:1px solid var(--bu-long-bd)}
.bu-cover  {background:var(--bu-cover-bg);  color:var(--bu-cover-cl);  border:1px solid var(--bu-cover-bd);
            box-shadow:0 0 8px rgba(255,140,0,.25); animation:bu-pulse 1.8s ease infinite}
.bu-short  {background:var(--bu-short-bg);  color:var(--bu-short-cl);  border:1px solid var(--bu-short-bd)}
.bu-unwind {background:var(--bu-unwind-bg); color:var(--bu-unwind-cl); border:1px solid var(--bu-unwind-bd)}
.bu-watch  {background:var(--bu-watch-bg);  color:var(--bu-watch-cl);  border:1px solid var(--bu-watch-bd)}
.bu-neutral{background:var(--bu-neutral-bg);color:var(--bu-neutral-cl);border:1px solid var(--bu-neutral-bd)}
@keyframes bu-pulse{0%,100%{box-shadow:0 0 6px rgba(255,140,0,.3)}50%{box-shadow:0 0 14px rgba(255,140,0,.6)}}

/* strength label */
.bu-strength{display:block;font-size:.55rem;font-weight:600;letter-spacing:.06em;margin-top:2px;opacity:.8}
/* pct change sub-line */
.bu-pct{display:block;font-size:.58rem;font-family:'IBM Plex Mono',monospace;margin-top:3px;color:var(--muted)}
.bu-pct-up{color:var(--ce)}.bu-pct-dn{color:var(--pe)}

/* IV info footer */
.iv-foot{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
  padding:9px 16px;border-top:1px solid var(--brd);gap:8px;background:var(--s2);
}
.iv-foot span{font-size:.62rem;color:var(--muted);font-family:'IBM Plex Mono',monospace}
.iv-foot b{color:var(--text)}

/* ══════════════════════════════════════════════════════════
   OI CHART CARD
══════════════════════════════════════════════════════════ */
.chart-card{
  background:var(--s1);border:1px solid var(--brd);border-radius:8px;
  overflow:hidden;margin-bottom:18px;box-shadow:var(--shadow);transition:background .3s;
}
.chart-hd{
  padding:13px 18px 12px;border-bottom:1px solid var(--brd);background:var(--s2);
}
.chart-hd-top{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;
  margin-bottom:10px;
}
.chart-title{font-size:.86rem;font-weight:700;letter-spacing:.04em;
  display:flex;align-items:center;gap:8px;color:var(--text)}
.chart-title svg{width:16px;height:16px;stroke:var(--ce);flex-shrink:0}
.chart-meta{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.cmeta{
  display:flex;align-items:center;gap:5px;
  font-size:.64rem;font-family:'IBM Plex Mono',monospace;
  padding:4px 11px;border-radius:50px;
  border:1px solid var(--brd);background:var(--s1);color:var(--muted);
  white-space:nowrap;
}
.cmeta b{font-weight:700}
.cm-ce{color:var(--ce)!important;border-color:rgba(0,212,160,.3)!important;background:rgba(0,212,160,.06)!important}
.cm-pe{color:var(--pe)!important;border-color:rgba(255,63,108,.3)!important;background:rgba(255,63,108,.06)!important}
[data-theme="day"] .cm-ce{background:rgba(0,122,92,.06)!important;border-color:rgba(0,122,92,.3)!important}
[data-theme="day"] .cm-pe{background:rgba(181,21,50,.06)!important;border-color:rgba(181,21,50,.3)!important}
.src-badge{
  font-size:.58rem;font-family:'IBM Plex Mono',monospace;font-weight:700;
  padding:3px 9px;border-radius:50px;letter-spacing:.06em;
}
.src-hist{background:rgba(107,180,255,.1);color:#6bacd6;border:1px solid rgba(107,180,255,.3)}
.src-live{background:rgba(251,191,36,.1);color:var(--gold);border:1px solid rgba(251,191,36,.3)}
.chart-sel-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.chart-sel-lbl{font-size:.66rem;font-family:'IBM Plex Mono',monospace;color:var(--muted);white-space:nowrap;flex-shrink:0;}
.strike-btns{display:flex;align-items:center;gap:6px;flex-wrap:wrap;flex:1;}
.sk-btn{
  cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:.72rem;font-weight:600;
  padding:5px 13px;border-radius:6px;border:1px solid var(--brd);
  background:var(--s1);color:var(--muted);transition:all .18s;user-select:none;white-space:nowrap;
}
.sk-btn:hover{border-color:var(--ce);color:var(--text)}
.sk-btn.active{background:rgba(0,212,160,.12);border-color:var(--ce);color:var(--ce);box-shadow:0 0 0 2px rgba(0,212,160,.18);}
.sk-btn.sk-atm{background:rgba(251,191,36,.1);border-color:var(--gold);color:var(--gold);font-weight:700;}
.sk-btn.sk-atm.active{background:rgba(251,191,36,.2);box-shadow:0 0 0 2px rgba(251,191,36,.25);}
.sk-btn.sk-total{background:rgba(107,124,255,.08);border-color:rgba(107,124,255,.4);color:#8b9eff;font-weight:700;}
.sk-btn.sk-total.active{background:rgba(107,124,255,.18);border-color:#8b9eff;color:#c0caffff;box-shadow:0 0 0 2px rgba(107,124,255,.25);}
.sk-btn.sk-otm{background:rgba(251,191,36,.07);border-color:rgba(251,191,36,.35);color:#d4a800;font-weight:700;}
.sk-btn.sk-otm.active{background:rgba(251,191,36,.18);border-color:var(--gold);color:var(--gold);box-shadow:0 0 0 2px rgba(251,191,36,.25);}
[data-theme="day"] .sk-btn.sk-otm{background:rgba(154,103,0,.07);border-color:rgba(154,103,0,.4);color:var(--gold)}
[data-theme="day"] .sk-btn.sk-otm.active{background:rgba(154,103,0,.16);border-color:var(--gold)}
.sk-btn.sk-ce{border-color:rgba(0,212,160,.3)}.sk-btn.sk-pe{border-color:rgba(255,63,108,.3)}
[data-theme="day"] .sk-btn.active{background:rgba(0,122,92,.1);border-color:var(--ce);color:var(--ce)}
[data-theme="day"] .sk-btn.sk-atm{background:rgba(154,103,0,.07);border-color:var(--gold);color:var(--gold)}
.day-filter-row{
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  padding:10px 18px 12px;border-bottom:1px solid var(--brd);background:var(--s2);
}
.day-filter-lbl{font-size:.65rem;font-family:'IBM Plex Mono',monospace;color:var(--muted);white-space:nowrap;flex-shrink:0;margin-right:4px;}
.day-btn{
  cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:.72rem;font-weight:600;
  padding:5px 16px;border-radius:6px;border:1px solid var(--brd);
  background:var(--s1);color:var(--muted);transition:all .18s;user-select:none;white-space:nowrap;
}
.day-btn:hover{border-color:var(--gold);color:var(--text)}
.day-btn.active{background:rgba(251,191,36,.14);border-color:var(--gold);color:var(--gold);font-weight:700;box-shadow:0 0 0 2px rgba(251,191,36,.2);}
[data-theme="day"] .day-btn.active{background:rgba(154,103,0,.1);border-color:var(--gold);color:var(--gold)}
.day-btn-sep{width:1px;height:22px;background:var(--brd);margin:0 4px;flex-shrink:0}
.chart-canvas-wrap{position:relative;height:420px;min-height:420px}
.chart-body{padding:16px 20px 0;background:var(--chart-bg);transition:background .3s;position:relative;min-height:460px;}
.chart-spinner{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;justify-content:center;gap:14px;z-index:10;background:var(--chart-bg);}
.chart-spinner.show{display:flex}
.spinner-ring{width:42px;height:42px;border:3px solid var(--brd);border-top-color:var(--ce);border-radius:50%;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner-txt{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted)}
.chart-placeholder{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:.76rem;}
.chart-placeholder svg{opacity:.25;width:50px;height:50px;stroke:var(--muted)}
.chart-placeholder p{opacity:.5;text-align:center;line-height:1.6}
canvas#oi-chart{display:none;width:100%!important;height:420px!important}
.chart-legend{display:flex;align-items:center;justify-content:center;gap:24px;padding:11px 0 9px;flex-wrap:wrap;background:var(--chart-bg);}
.leg-item{display:flex;align-items:center;gap:7px;font-size:.65rem;font-family:'IBM Plex Mono',monospace;color:var(--muted)}
.leg-sw{width:28px;height:3px;border-radius:2px;flex-shrink:0}
.chart-foot{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;padding:9px 20px 11px;border-top:1px solid var(--brd);gap:10px;background:var(--s2);}
.chart-foot span{font-size:.62rem;color:var(--muted);font-family:'IBM Plex Mono',monospace}
.chart-foot b{color:var(--text)}

.err{background:var(--err-bg);border:1px solid var(--err-bd);border-radius:8px;
  padding:11px 16px;color:var(--err-cl);font-family:'IBM Plex Mono',monospace;
  font-size:.76rem;margin-bottom:14px}
footer{text-align:center;padding:14px 0 6px;color:var(--muted);
  font-size:.62rem;font-family:'IBM Plex Mono',monospace;
  border-top:1px solid var(--brd);margin-top:4px}

/* ══════════════════════════════════════════════════════════
   PRICE / VWAP CHART SECTION
══════════════════════════════════════════════════════════ */
.pv-card{background:var(--s1);border:1px solid var(--brd);border-radius:8px;overflow:hidden;margin-bottom:18px;box-shadow:var(--shadow);transition:background .3s;}
.pv-hd{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;padding:12px 18px;border-bottom:1px solid var(--brd);background:var(--s2);}
.pv-title{font-size:.86rem;font-weight:700;letter-spacing:.04em;display:flex;align-items:center;gap:8px}
.pv-title svg{width:16px;height:16px;stroke:var(--gold);flex-shrink:0}
.tf-btns{display:flex;align-items:center;gap:6px}
.tf-btn{cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:.72rem;font-weight:600;padding:5px 14px;border-radius:6px;border:1px solid var(--brd);background:var(--s1);color:var(--muted);transition:all .18s;user-select:none;}
.tf-btn:hover{border-color:var(--gold);color:var(--text)}
.tf-btn.active{background:rgba(251,191,36,.14);border-color:var(--gold);color:var(--gold);font-weight:700;box-shadow:0 0 0 2px rgba(251,191,36,.2);}
[data-theme="day"] .tf-btn.active{background:rgba(154,103,0,.1)}
.pv-body{display:grid;grid-template-columns:1fr 120px 1fr;gap:0;background:var(--chart-bg);transition:background .3s;min-height:400px;align-items:stretch;}
.pv-panel{padding:14px 14px 10px;display:flex;flex-direction:column;gap:8px;}
.pv-panel-title{font-size:.72rem;font-weight:700;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;gap:6px;padding:0 2px;}
.pv-panel-ce .pv-panel-title{color:var(--ce)}
.pv-panel-pe .pv-panel-title{color:var(--pe)}
.pv-panel-ce{border-right:1px solid var(--brd)}.pv-panel-pe{border-left:1px solid var(--brd)}
.pv-canvas-wrap{position:relative;flex:1;min-height:360px}
.pv-canvas-wrap canvas{display:none;width:100%!important;height:360px!important}
.pv-placeholder{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:.7rem;text-align:center;}
.pv-placeholder svg{opacity:.2;width:36px;height:36px;stroke:var(--muted)}
.pv-spinner{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:var(--chart-bg);z-index:5;}
.pv-spinner.show{display:flex}
.pv-strike-col{display:flex;flex-direction:column;align-items:stretch;border-left:1px solid var(--brd);border-right:1px solid var(--brd);background:var(--s2);padding:10px 6px;gap:4px;overflow-y:auto;align-self:stretch;}
.pv-sk-lbl{font-size:.56rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-family:'IBM Plex Mono',monospace;margin-bottom:4px;text-align:center;}
.pv-sk-btn{cursor:pointer;width:100%;font-family:'IBM Plex Mono',monospace;font-size:.65rem;font-weight:600;padding:7px 4px;border-radius:5px;border:1px solid var(--brd);background:var(--s1);color:var(--muted);transition:all .18s;user-select:none;text-align:center;white-space:normal;word-break:break-all;line-height:1.2;}
.pv-sk-btn:hover{border-color:var(--ce);color:var(--text)}
.pv-sk-btn.active{background:rgba(0,212,160,.12);border-color:var(--ce);color:var(--ce);font-weight:700;box-shadow:0 0 0 2px rgba(0,212,160,.15);}
.pv-sk-btn.pv-sk-atm{background:rgba(251,191,36,.1);border-color:var(--gold);color:var(--gold);}
.pv-sk-btn.pv-sk-atm.active{background:rgba(251,191,36,.2);box-shadow:0 0 0 2px rgba(251,191,36,.25);}
[data-theme="day"] .pv-sk-btn.active{background:rgba(0,122,92,.1)}
.pv-meta{display:flex;gap:8px;flex-wrap:wrap}
.pv-mp{font-size:.6rem;font-family:'IBM Plex Mono',monospace;padding:2px 8px;border-radius:50px;border:1px solid var(--brd);background:var(--s1);color:var(--muted);display:flex;align-items:center;gap:4px;}
.pv-mp b{font-weight:700}
.pv-mp-price-ce{color:var(--ce)!important;border-color:rgba(0,212,160,.3)!important;background:rgba(0,212,160,.06)!important}
.pv-mp-vwap-ce {color:#a0e8ff!important;border-color:rgba(160,232,255,.3)!important;background:rgba(160,232,255,.05)!important}
.pv-mp-price-pe{color:var(--pe)!important;border-color:rgba(255,63,108,.3)!important;background:rgba(255,63,108,.06)!important}
.pv-mp-vwap-pe {color:#ffb3c8!important;border-color:rgba(255,179,200,.3)!important;background:rgba(255,179,200,.05)!important}
.pv-foot{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;padding:8px 18px;border-top:1px solid var(--brd);gap:8px;background:var(--s2);}
.pv-foot span{font-size:.62rem;color:var(--muted);font-family:'IBM Plex Mono',monospace}
.pv-foot b{color:var(--text)}
</style>
</head>
<body>
<div class="wrap">

<!-- ── TOP BAR ── -->
<div class="topbar">
  <div class="brand">
    <div class="brand-ico">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>
        <polyline points="16 7 22 7 22 13"/>
      </svg>
    </div>
    <div class="brand-name">NIFTY <em>OI</em> Dashboard</div>
  </div>
  <div class="topbar-right">
    <button class="theme-btn" id="theme-btn" onclick="toggleTheme()">
      <span id="theme-ico">☀️</span><span id="theme-lbl">Day Mode</span>
    </button>
    <div class="chips">
      <div class="chip"><span class="dot dot-l"></span>LTP Live · 1s</div>
      <div class="chip"><span class="dot dot-o"></span>OI · 3 min</div>
      <div class="chip"><span class="dot dot-e"></span>Expiry: <b id="chip-exp">—</b></div>
      <div class="chip">OI at <b id="chip-time">—</b></div>
      <div class="ring-wrap">
        <div class="ring">
          <svg viewBox="0 0 36 36" width="40" height="40">
            <circle class="rg" cx="18" cy="18" r="16"/>
            <circle class="rf" cx="18" cy="18" r="16" id="ring-fg"/>
          </svg>
          <div class="rlbl" id="ring-lbl">—</div>
        </div>
        <div class="ring-txt">OI<br>refresh</div>
      </div>
    </div>
  </div>
</div>

<div class="err fade" id="err-box" style="display:none"></div>

<!-- ── STAT CARDS ── -->
<div class="cards fade">
  <div class="card c-spot">
    <div class="card-lbl">NIFTY Spot <span class="live-tag">LIVE</span></div>
    <div class="card-val v-spot" id="cv-spot">—</div>
    <div class="card-sub" id="cv-spot-t">—</div>
  </div>
  <div class="card c-atm">
    <div class="card-lbl">ATM Strike</div>
    <div class="card-val v-atm" id="cv-atm">—</div>
    <div class="card-sub">Rounded to 50</div>
  </div>
  <div class="card c-ce">
    <div class="card-lbl">Total CE OI</div>
    <div class="card-val v-ce" id="cv-tce">—</div>
    <div class="card-sub">11 strikes (ATM ± 5)</div>
  </div>
  <div class="card c-pe">
    <div class="card-lbl">Total PE OI</div>
    <div class="card-val v-pe" id="cv-tpe">—</div>
    <div class="card-sub">11 strikes (ATM ± 5)</div>
  </div>
  <div class="card c-pcr">
    <div class="card-lbl">Overall PCR</div>
    <div class="card-val v-pcr" id="cv-pcr">—</div>
    <div class="card-sub" id="cv-pcr-s">—</div>
  </div>
  <div class="card c-exp">
    <div class="card-lbl">Expiry</div>
    <div class="card-val v-exp" id="cv-exp">—</div>
    <div class="card-sub">Weekly / Monthly</div>
  </div>
</div>

<!-- ── PCR BAR ── -->
<div class="pcr-wrap fade">
  <div class="pcr-top">
    <div class="pcr-ttl">Overall Put / Call Ratio (PCR) — 11 Strikes</div>
    <div class="pcr-num" id="pcr-big">—</div>
  </div>
  <div class="pcr-track"><div class="pcr-fill" id="pcr-fill" style="width:50%"></div></div>
  <div class="pcr-axis">
    <span>Bearish &lt; 0.7</span><span>Neutral 0.7 – 1.2</span><span>Bullish &gt; 1.2</span>
  </div>
  <div class="pcr-sent" id="pcr-sent">—</div>
</div>

<!-- ── OI TABLE ── -->
<div class="tbl-card fade">
  <div class="tbl-hd">
    <div class="tbl-title">Open Interest Chain — 11 Strikes (ATM ± 5) · Live LTP · OI Δ vs prev 3-min · Per-Strike PCR</div>
    <div class="tbl-legend">
      <span style="color:var(--ce)">■ CE</span>
      <span style="color:var(--pe)">■ PE</span>
      <span style="color:var(--gold)">ATM highlighted</span>
      <span style="color:#8b9eff">NEW = first / prev OI was 0</span>
    </div>
  </div>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th class="grp-ce" colspan="5">◀  CALL (CE)</th>
        <th class="grp-mid" colspan="2">STRIKE · PCR</th>
        <th class="grp-pe" colspan="5">PUT (PE)  ▶</th>
      </tr>
      <tr>
        <th class="sh-ce">OI Day High</th>
        <th class="sh-ce">OI (Lots)</th>
        <th class="sh-ce">OI Δ Lots</th>
        <th class="sh-ce">OI Δ %</th>
        <th class="sh-ce">LTP <span class="live-tag">LIVE</span></th>
        <th class="sh-mid">Strike</th>
        <th class="sh-mid">Strike PCR</th>
        <th class="sh-pe">LTP <span class="live-tag">LIVE</span></th>
        <th class="sh-pe">OI Δ %</th>
        <th class="sh-pe">OI Δ Lots</th>
        <th class="sh-pe">OI (Lots)</th>
        <th class="sh-pe">OI Day High</th>
      </tr>
    </thead>
    <tbody id="oi-tbody"></tbody>
    <tfoot>
      <tr>
        <td class="tot-ce" colspan="2" id="foot-tce">—</td>
        <td colspan="3"></td>
        <td class="tot-mid">TOTAL OI</td>
        <td class="tot-pcr" id="foot-pcr">—</td>
        <td colspan="3"></td>
        <td class="tot-pe" colspan="2" id="foot-tpe">—</td>
      </tr>
    </tfoot>
  </table>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════
     IV TABLE  —  Implied Volatility + EMA(9) + Signal
══════════════════════════════════════════════════════════ -->
<div class="iv-card fade">
  <div class="iv-hd">
    <div class="iv-title">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 20h20M5 20V10l7-7 7 7v10"/>
        <path d="M9 20v-5h6v5"/>
      </svg>
      Implied Volatility — Black-Scholes · EMA(9) Signal
      <span style="color:var(--muted);font-size:.7rem;font-weight:400">Newton-Raphson IV · risk-free 6.5%</span>
    </div>
    <div class="iv-controls">
      <!-- Timeframe selector -->
      <div class="iv-tf-wrap">
        <span class="iv-tf-lbl">TF:</span>
        <button class="iv-tf-btn" data-tf="1"  onclick="setIvTf(1)">1 min</button>
        <button class="iv-tf-btn" data-tf="3"  onclick="setIvTf(3)">3 min</button>
        <button class="iv-tf-btn active" data-tf="5"  onclick="setIvTf(5)">5 min</button>
        <button class="iv-tf-btn" data-tf="10" onclick="setIvTf(10)">10 min</button>
      </div>
      <!-- Legend -->
      <div class="iv-legend">
        <div class="ivl ivl-high">🔴 HIGH — IV &gt; EMA(9)</div>
        <div class="ivl ivl-low" >🟢 LOW  — IV &lt; EMA(9)</div>
        <div class="ivl ivl-flat">⚪ FLAT — within ±0.5%</div>
        <div class="ivl ivl-new" >⚡ NEW  — building EMA</div>
        <div style="width:1px;height:18px;background:var(--brd);flex-shrink:0"></div>
        <div class="ivl" style="background:var(--bu-long-bg);color:var(--bu-long-cl);border:1px solid var(--bu-long-bd)">✅ Long Build-up</div>
        <div class="ivl" style="background:var(--bu-cover-bg);color:var(--bu-cover-cl);border:1px solid var(--bu-cover-bd)">🔥🔥 Short Covering</div>
        <div class="ivl" style="background:var(--bu-short-bg);color:var(--bu-short-cl);border:1px solid var(--bu-short-bd)">🔻 Short Build-up</div>
        <div class="ivl" style="background:var(--bu-unwind-bg);color:var(--bu-unwind-cl);border:1px solid var(--bu-unwind-bd)">⚠️ Long Unwinding</div>
      </div>
    </div>
  </div>
  <div style="overflow-x:auto">
  <table id="iv-table">
    <thead>
      <tr>
        <th class="iv-grp-ce" colspan="5">◀  CALL (CE) — IV · Buildup Signal</th>
        <th class="iv-grp-mid" colspan="2">STRIKE</th>
        <th class="iv-grp-pe" colspan="5">PUT (PE) — IV · Buildup Signal  ▶</th>
      </tr>
      <tr>
        <th class="iv-sh-ce">CE Buildup</th>
        <th class="iv-sh-ce">CE Signal vs EMA</th>
        <th class="iv-sh-ce">CE EMA(9) IV</th>
        <th class="iv-sh-ce">CE IV %</th>
        <th class="iv-sh-ce">CE LTP <span class="live-tag">LIVE</span></th>
        <th class="iv-sh-mid">Strike</th>
        <th class="iv-sh-mid">ATM / OTM</th>
        <th class="iv-sh-pe">PE LTP <span class="live-tag">LIVE</span></th>
        <th class="iv-sh-pe">PE IV %</th>
        <th class="iv-sh-pe">PE EMA(9) IV</th>
        <th class="iv-sh-pe">PE Signal vs EMA</th>
        <th class="iv-sh-pe">PE Buildup</th>
      </tr>
    </thead>
    <tbody id="iv-tbody">
      <tr><td colspan="12" style="text-align:center;padding:28px;color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:.76rem;">
        Waiting for first OI snapshot to compute IV…
      </td></tr>
    </tbody>
  </table>
  </div>
  <div class="iv-foot">
    <span>Method: <b>Black-Scholes Newton-Raphson</b></span>
    <span>IV Signal threshold: <b>±0.5% vs EMA(9)</b></span>
    <span>Buildup threshold: <b>Price ±5% &amp; OI ±5%</b></span>
    <span>Updated at: <b id="iv-updated">—</b></span>
    <span>TF: <b id="iv-tf-lbl">5 min</b></span>
    <span style="color:var(--bu-long-cl)">✅ Long Build-up</span>
    <span style="color:var(--bu-cover-cl)">🔥🔥 Short Covering</span>
    <span style="color:var(--bu-short-cl)">🔻 Short Build-up</span>
    <span style="color:var(--bu-unwind-cl)">⚠️ Long Unwinding</span>
    <span style="color:#a78bfa">LTP updates every 1s · IV/signals update every 3min</span>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════
     OI HISTORY CHART
══════════════════════════════════════════════════════════ -->
<div class="chart-card fade">
  <div class="chart-hd">
    <div class="chart-hd-top">
      <div class="chart-title">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        OI History Chart — CE &amp; PE  <span style="color:var(--muted);font-size:.7rem;font-weight:400">3 trading days + today live</span>
      </div>
      <div class="chart-meta">
        <div class="cmeta cm-ce">CE OI: <b id="ch-ce">—</b></div>
        <div class="cmeta cm-pe">PE OI: <b id="ch-pe">—</b></div>
        <div class="cmeta">PCR: <b id="ch-pcr">—</b></div>
        <div class="cmeta">Points: <b id="ch-pts">0</b></div>
        <span class="src-badge" id="ch-src" style="display:none"></span>
      </div>
    </div>
    <div class="chart-sel-row">
      <span class="chart-sel-lbl">Strike →</span>
      <div class="strike-btns" id="strike-btns">
        <span style="font-size:.66rem;color:var(--muted);font-family:'IBM Plex Mono',monospace">loading…</span>
      </div>
    </div>
  </div>
  <div class="day-filter-row">
    <span class="day-filter-lbl">Show:</span>
    <button class="day-btn" data-days="1" onclick="setDays(1)">1 Day</button>
    <button class="day-btn" data-days="2" onclick="setDays(2)">2 Days</button>
    <button class="day-btn" data-days="3" onclick="setDays(3)">3 Days</button>
    <button class="day-btn active" data-days="4" onclick="setDays(4)">4 Days</button>
    <div class="day-btn-sep"></div>
    <span style="font-size:.6rem;color:var(--muted);font-family:'IBM Plex Mono',monospace">
      Each day = market hours only (9:15 AM – 3:30 PM)
    </span>
  </div>
  <div class="chart-body">
    <div class="chart-canvas-wrap">
      <div class="chart-spinner" id="chart-spinner">
        <div class="spinner-ring"></div>
        <div class="spinner-txt">Fetching 4-day OI history…</div>
      </div>
      <div class="chart-placeholder" id="chart-ph">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
        <p>Waiting for first OI snapshot…<br>
           <span style="font-size:.62rem">Select a strike above · chart updates every 3 min</span></p>
      </div>
      <canvas id="oi-chart"></canvas>
    </div>
    <div class="chart-legend">
      <div class="leg-item"><div class="leg-sw" style="background:var(--ce)"></div><span style="color:var(--ce);font-weight:700">CE OI</span><span>Call Open Interest</span></div>
      <div class="leg-item"><div class="leg-sw" style="background:var(--pe)"></div><span style="color:var(--pe);font-weight:700">PE OI</span><span>Put Open Interest</span></div>
      <div class="leg-item" style="font-size:.6rem"><span style="color:var(--muted)">Hover over chart to see values · Each point = 3-min snapshot</span></div>
    </div>
  </div>
  <div class="chart-foot">
    <span>Strike: <b id="cf-strike">—</b></span>
    <span>From: <b id="cf-from">—</b></span>
    <span>To: <b id="cf-to">—</b></span>
    <span>Data points: <b id="cf-pts">0</b></span>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════
     PRICE + VWAP CHARTS
══════════════════════════════════════════════════════════ -->
<div class="pv-card fade">
  <div class="pv-hd">
    <div class="pv-title">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 2 9 10 13 4 17 10 21 6"/>
        <line x1="3" y1="20" x2="21" y2="20"/>
      </svg>
      Price &amp; VWAP
      <span style="color:var(--muted);font-size:.7rem;font-weight:400">CE on left · PE on right · today intraday</span>
    </div>
    <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:.64rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;margin-right:2px">TF:</span>
        <button class="tf-btn active" data-tf="1"  onclick="setPvTf(1)">1 min</button>
        <button class="tf-btn"        data-tf="3"  onclick="setPvTf(3)">3 min</button>
        <button class="tf-btn"        data-tf="5"  onclick="setPvTf(5)">5 min</button>
        <button class="tf-btn"        data-tf="10" onclick="setPvTf(10)">10 min</button>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap" id="pv-meta-wrap">
        <div class="pv-mp pv-mp-price-ce">CE LTP: <b id="pv-ce-ltp">—</b></div>
        <div class="pv-mp pv-mp-vwap-ce" >CE VWAP: <b id="pv-ce-vwap">—</b></div>
        <div class="pv-mp pv-mp-price-pe">PE LTP: <b id="pv-pe-ltp">—</b></div>
        <div class="pv-mp pv-mp-vwap-pe" >PE VWAP: <b id="pv-pe-vwap">—</b></div>
      </div>
    </div>
  </div>
  <div class="pv-body">
    <div class="pv-panel pv-panel-ce">
      <div class="pv-panel-title">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/></svg>
        CE — Price &amp; VWAP
      </div>
      <div class="pv-canvas-wrap">
        <div class="pv-spinner" id="pv-ce-spin"><div class="spinner-ring"></div><div class="spinner-txt">Loading CE data…</div></div>
        <div class="pv-placeholder" id="pv-ce-ph">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5" stroke-linecap="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/></svg>
          <p>Select a strike<br>to load CE price chart</p>
        </div>
        <canvas id="ce-chart"></canvas>
      </div>
    </div>
    <div class="pv-strike-col" id="pv-strike-col">
      <div class="pv-sk-lbl">Strike</div>
      <div style="font-size:.6rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;text-align:center">loading…</div>
    </div>
    <div class="pv-panel pv-panel-pe">
      <div class="pv-panel-title">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/></svg>
        PE — Price &amp; VWAP
      </div>
      <div class="pv-canvas-wrap">
        <div class="pv-spinner" id="pv-pe-spin"><div class="spinner-ring"></div><div class="spinner-txt">Loading PE data…</div></div>
        <div class="pv-placeholder" id="pv-pe-ph">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5" stroke-linecap="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/></svg>
          <p>Select a strike<br>to load PE price chart</p>
        </div>
        <canvas id="pe-chart"></canvas>
      </div>
    </div>
  </div>
  <div class="pv-foot">
    <span>Strike: <b id="pv-foot-strike">—</b></span>
    <span>Timeframe: <b id="pv-foot-tf">—</b></span>
    <span>CE candles: <b id="pv-foot-ce-pts">0</b></span>
    <span>PE candles: <b id="pv-foot-pe-pts">0</b></span>
    <span style="color:#8b9eff">Auto-refreshes every 60 sec</span>
  </div>
</div>

<footer>NIFTY OI Dashboard v6 &nbsp;·&nbsp; Kite Connect API &nbsp;·&nbsp; 11 Strikes (ATM±5) &nbsp;·&nbsp; IV Black-Scholes + EMA(9) &nbsp;·&nbsp; 4-day OI history &nbsp;·&nbsp; NSE/NFO</footer>
</div><!-- /wrap -->

<!-- ══════════════════════════════════════════════════════════
     JAVASCRIPT
══════════════════════════════════════════════════════════ -->
<script>
const OI_INT  = {{ oi_interval }};
const LTP_INT = {{ ltp_interval }};
let oiCountdown = OI_INT;
let cachedRows  = [];
let prevLTP     = {};

const localHistory = {};
let chartStrike   = null;
let activeDays    = 4;
let oiChart       = null;
let chartIsInit   = false;

/* IV table state */
let ivTf = 5;   // default 5-min TF
let ivLtpMap = {};  // sym → strike, populated by buildIvTable for live LTP updates

/* ── Theme ── */
let isDayMode = false;
function toggleTheme(){
  isDayMode = !isDayMode;
  document.documentElement.setAttribute('data-theme', isDayMode ? 'day' : 'night');
  document.getElementById('theme-ico').textContent = isDayMode ? '🌙' : '☀️';
  document.getElementById('theme-lbl').textContent = isDayMode ? 'Night Mode' : 'Day Mode';
  localStorage.setItem('oi-theme', isDayMode ? 'day' : 'night');
  if(oiChart) rebuildChartColors();
  if(ceChart) updatePvChartColors(ceChart, cssv('--ce'), '#a0e8ff');
  if(peChart) updatePvChartColors(peChart, cssv('--pe'), '#ffb3c8');
}
(function(){
  if(localStorage.getItem('oi-theme') === 'day'){
    isDayMode = true;
    document.documentElement.setAttribute('data-theme','day');
    document.getElementById('theme-ico').textContent = '🌙';
    document.getElementById('theme-lbl').textContent = 'Night Mode';
  }
})();

/* ── Formatters ── */
const fmtN  = n => n == null ? '—' : Number(n).toLocaleString('en-IN');
const fmtP  = n => n == null ? '—' : Number(n).toFixed(2);
const fmtK  = n => {
  if(n == null || n === '') return '—'; n = Number(n);
  if(Math.abs(n)>=1e7)  return (n/1e7).toFixed(2)+' Cr';
  if(Math.abs(n)>=1e5)  return (n/1e5).toFixed(2)+' L';
  if(Math.abs(n)>=1000) return n.toLocaleString('en-IN');
  return String(n);
};
const fmtKs = n => {
  if(n == null) return null;
  return (n>0?'+':'')+fmtK(n)+(n>0?' ▲':n<0?' ▼':'');
};

/* ── PCR helpers ── */
function pcrInfo(p){
  const pct=Math.min(100,Math.max(0,(p/2)*100));
  if(!p)    return ['—',           'var(--gold)',50];
  if(p<0.7) return ['Bearish — Heavy Call Writing','var(--pe)',  pct];
  if(p<=1.2)return ['Neutral — Market Balanced',   'var(--gold)',pct];
  return           ['Bullish — Heavy Put Writing',  'var(--ce)',  pct];
}
function spcrPill(pcr){
  if(pcr==null) return '<span class="pct-pill pill-new">—</span>';
  let cls,lbl;
  if(pcr<0.7)       {cls='pcr-bear';lbl='▼ '+pcr.toFixed(3);}
  else if(pcr<=1.2) {cls='pcr-neut';lbl='↔ '+pcr.toFixed(3);}
  else              {cls='pcr-bull';lbl='▲ '+pcr.toFixed(3);}
  return `<span class="pcr-s ${cls}">${lbl}</span>`;
}

/* ── Ring ── */
setInterval(()=>{
  oiCountdown=Math.max(0,oiCountdown-1);
  const C=2*Math.PI*16;
  document.getElementById('ring-fg').style.strokeDashoffset=C*(1-oiCountdown/OI_INT);
  document.getElementById('ring-lbl').textContent=oiCountdown+'s';
},1000);

/* ── Flash ── */
function flash(id,nv,pv){
  const el=document.getElementById(id);
  if(!el||pv===undefined)return;
  el.classList.remove('flash-u','flash-d'); void el.offsetWidth;
  if(nv>pv) el.classList.add('flash-u'); else if(nv<pv) el.classList.add('flash-d');
}

/* OI diff cell */
function diffCell(diff, pct, align){
  if(diff === null || diff === undefined){
    const f = align==='ce' ? 'float:right' : '';
    return '<span class="pct-pill pill-new" style="'+f+'">NEW</span>';
  }
  const dc = diff>0 ? 'diff-p' : diff<0 ? 'diff-n' : 'diff-z';
  const pctHtml = (pct===null || pct===undefined)
    ? '<span class="pct-pill pill-new">NEW OI</span>'
    : '<span class="pct-pill '+(pct>0?'pill-p':pct<0?'pill-n':'pill-z')+'">'
      + (pct>=0?'+':'')+Number(pct).toFixed(2)+'%</span>';
  const ds = '<span class="'+dc+'" style="display:block;text-align:'
    +(align==='ce'?'right':'left')+'">'+fmtKs(diff)+'</span>';
  return align==='ce' ? ds+pctHtml : pctHtml+ds;
}

/* ── Build OI table row ── */
function buildRow(r, mxCE, mxPE){
  const atm = r.is_atm;
  const bCE = mxCE ? Math.min(100,(r.ce_oi/mxCE)*100) : 0;
  const bPE = mxPE ? Math.min(100,(r.pe_oi/mxPE)*100) : 0;
  const off = r.offset===0 ? '' : (r.offset>0 ? 'ATM+'+r.offset : 'ATM'+r.offset);
  const atmbadge = atm ? '<span class="atm-badge">ATM</span>' : '';
  const offsetSpan = off ? '<span class="strike-offset">'+off+'</span>' : '';
  const sLbl = '<span class="strike-lbl" style="color:var(--gold)">'+r.strike+atmbadge+'</span>'+offsetSpan;
  return '<tr'+(atm?' class="atm-row"':'')+'>'+
    '<td class="td-ce" style="color:var(--muted)">'+fmtK(r.ce_oi_tot)+'</td>'+
    '<td class="td-ce oi-cell">'+
      '<div class="oi-bar oi-bar-ce" style="width:'+bCE+'%"></div>'+
      '<span class="oi-num" style="color:var(--ce)">'+fmtK(r.ce_oi)+'</span>'+
    '</td>'+
    '<td class="td-ce" style="min-width:80px">'+diffCell(r.ce_oi_diff,r.ce_oi_pct,'ce')+'</td>'+
    '<td class="td-ce" style="min-width:70px"></td>'+
    '<td class="td-ce" id="cetd'+r.strike+'">'+
      '<span class="ltp-val ltp-ce" id="celtp'+r.strike+'">'+fmtP(r.ce_ltp)+'</span>'+
    '</td>'+
    '<td class="td-mid" style="min-width:120px">'+sLbl+'</td>'+
    '<td class="td-mid pcr-cell">'+spcrPill(r.strike_pcr)+'</td>'+
    '<td class="td-pe" id="petd'+r.strike+'">'+
      '<span class="ltp-val ltp-pe" id="peltp'+r.strike+'">'+fmtP(r.pe_ltp)+'</span>'+
    '</td>'+
    '<td class="td-pe" style="min-width:70px"></td>'+
    '<td class="td-pe" style="min-width:80px">'+diffCell(r.pe_oi_diff,r.pe_oi_pct,'pe')+'</td>'+
    '<td class="td-pe oi-cell">'+
      '<div class="oi-bar oi-bar-pe" style="width:'+bPE+'%"></div>'+
      '<span class="oi-num" style="color:var(--pe)">'+fmtK(r.pe_oi)+'</span>'+
    '</td>'+
    '<td class="td-pe" style="color:var(--muted)">'+fmtK(r.pe_oi_tot)+'</td>'+
  '</tr>';
}

/* ════════════════════════════════════════════════════════════
   IV TABLE
════════════════════════════════════════════════════════════ */

function setIvTf(tf){
  ivTf = tf;
  document.querySelectorAll('.iv-tf-btn').forEach(b=>{
    b.classList.toggle('active', parseInt(b.dataset.tf)===tf);
  });
  document.getElementById('iv-tf-lbl').textContent = tf + ' min';
  fetchIV();
}

/* Signal badge HTML */
function ivSigBadge(signal, ivVal, ema9Val){
  const icons = {HIGH:'🔴', LOW:'🟢', FLAT:'⚪', NEW:'⚡'};
  const labels = {
    HIGH: 'HIGH IV',
    LOW:  'LOW IV',
    FLAT: 'FLAT',
    NEW:  'BUILDING',
  };
  const icon  = icons[signal]  || '⚡';
  const label = labels[signal] || 'NEW';
  return `<span class="iv-sig iv-sig-${signal}">${icon} ${label}</span>`;
}

/* IV number cell — shows IV% + sparkline + EMA diff */
function ivNumCell(ivVal, ema9Val, side){
  const cls = side === 'ce' ? 'iv-num iv-num-ce' : 'iv-num iv-num-pe';
  if(ivVal == null) return '<span class="iv-num" style="color:var(--muted)">—</span>';
  let diffHtml = '';
  if(ema9Val != null){
    const d = ivVal - ema9Val;
    const dc = d > 0.5 ? 'iv-diff-p' : d < -0.5 ? 'iv-diff-n' : 'iv-diff-z';
    const sign = d >= 0 ? '+' : '';
    diffHtml = `<span class="iv-diff ${dc}">${sign}${d.toFixed(2)}% vs EMA</span>`;
  }
  return `<span class="${cls}">${ivVal.toFixed(2)}%</span>${diffHtml}`;
}

/* EMA cell */
function ema9Cell(ema9Val){
  if(ema9Val == null) return '<span style="color:var(--muted);font-family:\'IBM Plex Mono\',monospace;font-size:.72rem">—</span>';
  return `<span class="iv-ema">${ema9Val.toFixed(2)}%</span>`;
}

/* Mini sparkline for IV history using Canvas 2D */
const sparkCanvases = {};   // sparkId → canvas el

function drawSparkline(canvasId, series, color){
  let cv = document.getElementById(canvasId);
  if(!cv) return;
  const W = 80, H = 28;
  cv.width = W; cv.height = H;
  const vals = series.map(p=>p.iv).filter(v=>v!=null);
  if(vals.length < 2) return;
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const range = mx - mn || 1;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.lineJoin    = 'round';
  ctx.beginPath();
  vals.forEach((v,i)=>{
    const x = (i / (vals.length-1)) * (W-2) + 1;
    const y = H - 2 - ((v-mn)/range) * (H-4);
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Draw EMA line in a lighter shade
  const emas = series.map(p=>p.ema9).filter(v=>v!=null);
  if(emas.length >= 2){
    ctx.strokeStyle = isDayMode ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.3)';
    ctx.lineWidth   = 1;
    ctx.setLineDash([3,3]);
    ctx.beginPath();
    emas.forEach((v,i)=>{
      const x = (i / (emas.length-1)) * (W-2) + 1;
      const y = H - 2 - ((v-mn)/range) * (H-4);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

/* ── Buildup signal badge ─────────────────────────────────
   bu       = {name, emoji, strength, action, cls}
   pricePct = option price % change over TF window
   oiPct    = option OI % change over TF window
─────────────────────────────────────────────────────────── */
function buildupBadge(bu, pricePct, oiPct){
  if(!bu || bu.cls === 'bu-neutral'){
    // Still show the pct data so trader can see "almost" signals
    const noData = (pricePct == null && oiPct == null);
    if(noData) return '<span class="bu-badge bu-neutral">— Neutral</span>';
    const pStr = pricePct != null
      ? `<span class="bu-pct">P:<span class="${pricePct>=0?'bu-pct-up':'bu-pct-dn'}">${pricePct>=0?'+':''}${pricePct.toFixed(2)}%</span></span>`
      : '';
    const oStr = oiPct != null
      ? `<span class="bu-pct">OI:<span class="${oiPct>=0?'bu-pct-up':'bu-pct-dn'}">${oiPct>=0?'+':''}${oiPct.toFixed(2)}%</span></span>`
      : '';
    return `<div class="bu-badge bu-neutral">— Neutral ${pStr}${oStr}</div>`;
  }

  const pStr = pricePct != null
    ? `<span class="bu-pct">P:<span class="${pricePct>=0?'bu-pct-up':'bu-pct-dn'}">${pricePct>=0?'+':''}${pricePct.toFixed(2)}%</span></span>`
    : '';
  const oStr = oiPct != null
    ? `<span class="bu-pct">OI:<span class="${oiPct>=0?'bu-pct-up':'bu-pct-dn'}">${oiPct>=0?'+':''}${oiPct.toFixed(2)}%</span></span>`
    : '';
  const strengthHtml = bu.strength
    ? `<span class="bu-strength">${bu.strength} · ${bu.action}</span>`
    : '';
  return `<div class="bu-badge ${bu.cls}">${bu.emoji} ${bu.name}${strengthHtml}${pStr}${oStr}</div>`;
}

/* Build the IV table body */
function buildIvTable(rows){
  if(!rows || rows.length === 0){
    document.getElementById('iv-tbody').innerHTML =
      '<tr><td colspan="12" style="text-align:center;padding:28px;color:var(--muted);font-family:\'IBM Plex Mono\',monospace;font-size:.76rem;">No IV data available yet — waiting for options LTP…</td></tr>';
    return;
  }

  const ceColor = getComputedStyle(document.documentElement).getPropertyValue('--ce').trim();
  const peColor = getComputedStyle(document.documentElement).getPropertyValue('--pe').trim();

  // Store sym→strike map for live LTP updates (built fresh each time)
  ivLtpMap = {};
  rows.forEach(r=>{ ivLtpMap[r.ce_sym] = r.strike; ivLtpMap[r.pe_sym] = r.strike; });

  const rowsHtml = rows.map(r=>{
    const atm  = r.is_atm;
    const off  = r.offset === 0 ? '' : (r.offset > 0 ? 'ATM+'+r.offset : 'ATM'+r.offset);
    const atmbadge  = atm ? '<span class="atm-badge">ATM</span>' : '';
    const sLbl      = `<span class="strike-lbl" style="color:var(--gold)">${r.strike}${atmbadge}</span>`;
    const offLbl    = off
      ? `<span style="font-size:.6rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;display:block;margin-top:2px">${off}</span>`
      : `<span style="font-size:.62rem;color:var(--muted);font-family:'IBM Plex Mono',monospace">ATM</span>`;
    const ceMoneyness = r.offset > 0 ? 'OTM Call' : r.offset < 0 ? 'ITM Call' : '★ ATM';
    const peMoneyness = r.offset < 0 ? 'OTM Put'  : r.offset > 0 ? 'ITM Put'  : '★ ATM';
    const ceSpark = `ce-spark-${r.strike}`;
    const peSpark = `pe-spark-${r.strike}`;

    return `<tr${atm?' class="atm-row"':''}>
      <td class="iv-td-ce" style="min-width:185px;padding:6px 8px">
        ${buildupBadge(r.ce_buildup, r.ce_price_pct, r.ce_oi_pct)}
      </td>
      <td class="iv-td-ce" style="min-width:140px">
        ${ivSigBadge(r.ce_signal, r.ce_iv, r.ce_ema9)}
      </td>
      <td class="iv-td-ce" style="min-width:100px">
        ${ema9Cell(r.ce_ema9)}
      </td>
      <td class="iv-td-ce" style="min-width:130px">
        ${ivNumCell(r.ce_iv, r.ce_ema9, 'ce')}
        <div class="iv-spark-wrap"><canvas id="${ceSpark}" width="80" height="28"></canvas></div>
      </td>
      <td class="iv-td-ce" style="min-width:80px">
        <span class="ltp-val ltp-ce" id="iv-ce-ltp-${r.strike}">${fmtP(r.ce_ltp)}</span>
        <span style="font-size:.58rem;color:var(--muted);display:block">${ceMoneyness}</span>
      </td>
      <td class="iv-td-mid">${sLbl}</td>
      <td class="iv-td-mid">${offLbl}</td>
      <td class="iv-td-pe" style="min-width:80px">
        <span class="ltp-val ltp-pe" id="iv-pe-ltp-${r.strike}">${fmtP(r.pe_ltp)}</span>
        <span style="font-size:.58rem;color:var(--muted);display:block">${peMoneyness}</span>
      </td>
      <td class="iv-td-pe" style="min-width:130px">
        ${ivNumCell(r.pe_iv, r.pe_ema9, 'pe')}
        <div class="iv-spark-wrap"><canvas id="${peSpark}" width="80" height="28"></canvas></div>
      </td>
      <td class="iv-td-pe" style="min-width:100px">
        ${ema9Cell(r.pe_ema9)}
      </td>
      <td class="iv-td-pe" style="min-width:140px">
        ${ivSigBadge(r.pe_signal, r.pe_iv, r.pe_ema9)}
      </td>
      <td class="iv-td-pe" style="min-width:185px;padding:6px 8px">
        ${buildupBadge(r.pe_buildup, r.pe_price_pct, r.pe_oi_pct)}
      </td>
    </tr>`;
  }).join('');

  document.getElementById('iv-tbody').innerHTML = rowsHtml;

  // Draw sparklines after DOM update
  requestAnimationFrame(()=>{
    rows.forEach(r=>{
      if(r.ce_series && r.ce_series.length >= 2)
        drawSparkline('ce-spark-'+r.strike, r.ce_series, ceColor);
      if(r.pe_series && r.pe_series.length >= 2)
        drawSparkline('pe-spark-'+r.strike, r.pe_series, peColor);
    });
  });
}

/* Fetch IV data from backend */
async function fetchIV(){
  try{
    const res  = await fetch('/api/iv?tf='+ivTf+'&_='+Date.now(), {cache:'no-store'});
    const data = await res.json();
    if(data.rows && data.rows.length > 0){
      buildIvTable(data.rows);
      document.getElementById('iv-updated').textContent = data.ts || '—';
    }
  } catch(e){
    console.warn('[IV] fetchIV error:', e);
  }
}

/* ════════════════════════════════════════════════════════════
   CHART ENGINE  (unchanged)
════════════════════════════════════════════════════════════ */
function cssv(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function hex2rgba(c, a){
  if(!c) return `rgba(0,0,0,${a})`;
  if(c.startsWith('#')){
    const r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
    return `rgba(${r},${g},${b},${a})`;
  }
  return c.replace('rgb(','rgba(').replace(')',`,${a})`);
}
function palette(){
  return {
    ce:cssv('--ce'),pe:cssv('--pe'),grid:cssv('--gc'),tick:cssv('--muted'),
    tip:cssv('--tt'),tipbrd:cssv('--tt-brd'),ttitle:cssv('--tt-title'),tbody:cssv('--tt-body'),
  };
}

function initChart(){
  const p   = palette();
  const ctx = document.getElementById('oi-chart').getContext('2d');
  const dayBoundaryPlugin = {
    id:'dayBoundary',
    afterDraw(chart){
      const labels = chart.data.labels || [];
      if(labels.length < 2) return;
      const ctx2   = chart.ctx;
      const xAxis  = chart.scales.x;
      const yAxis  = chart.scales.y;
      const top    = yAxis.top, bottom = yAxis.bottom;
      const lineCol= isDayMode?'rgba(0,0,0,0.30)':'rgba(255,255,255,0.30)';
      const lblCol = isDayMode?'rgba(0,0,0,0.55)':'rgba(255,255,255,0.55)';
      ctx2.save(); ctx2.strokeStyle=lineCol; ctx2.lineWidth=1; ctx2.setLineDash([4,3]);
      labels.forEach((lbl,i)=>{
        if(i===0) return;
        if(lbl.split(' ')[0]===labels[i-1].split(' ')[0]) return;
        const xPx=xAxis.getPixelForValue(i);
        ctx2.beginPath(); ctx2.moveTo(xPx,top); ctx2.lineTo(xPx,bottom); ctx2.stroke();
        ctx2.setLineDash([]);
        ctx2.fillStyle=lblCol; ctx2.font="bold 10px 'IBM Plex Mono', monospace";
        ctx2.textAlign='center'; ctx2.fillText(lbl.split(' ')[0],xPx,bottom+18);
        ctx2.setLineDash([4,3]);
      });
      ctx2.restore();
    }
  };
  oiChart = new Chart(ctx,{
    type:'line',plugins:[dayBoundaryPlugin],
    data:{labels:[],datasets:[
      {label:'CE OI',data:[],yAxisID:'y',borderColor:p.ce,borderWidth:2.5,pointRadius:0,pointHoverRadius:6,pointHoverBackgroundColor:p.ce,pointHoverBorderColor:'#fff',pointHoverBorderWidth:2,tension:0.35,fill:true,backgroundColor:hex2rgba(p.ce,0.09)},
      {label:'PE OI',data:[],yAxisID:'y',borderColor:p.pe,borderWidth:2.5,pointRadius:0,pointHoverRadius:6,pointHoverBackgroundColor:p.pe,pointHoverBorderColor:'#fff',pointHoverBorderWidth:2,tension:0.35,fill:true,backgroundColor:hex2rgba(p.pe,0.07)},
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,animation:{duration:400},
      layout:{padding:{bottom:24}},interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{display:false},
        tooltip:{
          backgroundColor:p.tip,borderColor:p.tipbrd,borderWidth:1,
          titleColor:p.ttitle,bodyColor:p.tbody,padding:14,
          titleFont:{family:"'IBM Plex Mono'",size:12,weight:'700'},
          bodyFont:{family:"'IBM Plex Mono'",size:11},
          filter:item=>item.dataset.label!=='_sep',
          callbacks:{
            title:items=>items[0].label,
            label:item=>'  '+item.dataset.label+': '+Number(item.raw).toLocaleString('en-IN')+' lots',
            afterBody:items=>{
              const ce=items.find(i=>i.dataset.label==='CE OI'||i.dataset.label==='Total CE OI')?.raw||0;
              const pe=items.find(i=>i.dataset.label==='PE OI'||i.dataset.label==='Total PE OI')?.raw||0;
              return ce?['  PCR: '+(pe/ce).toFixed(4)]:[];
            }
          }
        }
      },
      scales:{
        x:{grid:{display:false},ticks:{color:p.tick,font:{family:"'IBM Plex Mono'",size:10},maxRotation:0,autoSkip:true,maxTicksLimit:20,callback(val,idx){const lbl=this.getLabelForValue(val);if(!lbl)return'';return idx%20===0?lbl.split(' ')[1]||'':''}},border:{color:cssv('--brd'),display:true}},
        y:{grid:{display:false},ticks:{color:p.tick,font:{family:"'IBM Plex Mono'",size:11},callback:v=>{if(Math.abs(v)>=1e7)return(v/1e7).toFixed(1)+' Cr';if(Math.abs(v)>=1e5)return(v/1e5).toFixed(1)+' L';return Number(v).toLocaleString('en-IN')}},border:{color:cssv('--brd'),display:true}},
      }
    }
  });
  chartIsInit=true;
  oiChart.resize();
}

function rebuildChartColors(){
  if(!oiChart) return;
  const p=palette();
  const ds=oiChart.data.datasets;
  ds[0].borderColor=p.ce;ds[0].pointHoverBackgroundColor=p.ce;ds[0].backgroundColor=hex2rgba(p.ce,0.09);
  ds[1].borderColor=p.pe;ds[1].pointHoverBackgroundColor=p.pe;ds[1].backgroundColor=hex2rgba(p.pe,0.07);
  const o=oiChart.options;
  o.plugins.tooltip.backgroundColor=p.tip;o.plugins.tooltip.borderColor=p.tipbrd;
  o.plugins.tooltip.titleColor=p.ttitle;o.plugins.tooltip.bodyColor=p.tbody;
  o.scales.x.ticks.color=p.tick;o.scales.x.border.color=cssv('--brd');
  o.scales.y.ticks.color=p.tick;o.scales.y.border.color=cssv('--brd');
  oiChart.update('none');
}

function renderChart(strike){
  const isTotal=(strike==='TOTAL'), isOtm=(strike==='OTM');
  const hist=localHistory[strike];
  const ph=document.getElementById('chart-ph');
  const cv=document.getElementById('oi-chart');
  const spin=document.getElementById('chart-spinner');
  spin.classList.remove('show');
  if(!hist||hist.ts.length===0){ph.style.display='flex';cv.style.display='none';updateChartMeta(strike,null,null,0,isTotal);return;}
  const uniqueDates=[...new Set(hist.ts.map(t=>t.split(' ')[0]))].sort();
  const keepDates=new Set(uniqueDates.slice(-activeDays));
  const idxs=hist.ts.reduce((a,t,i)=>{if(keepDates.has(t.split(' ')[0]))a.push(i);return a;},[]);
  const labels=idxs.map(i=>hist.ts[i]);
  const ceData=idxs.map(i=>hist.ce[i]);
  const peData=idxs.map(i=>hist.pe[i]);
  ph.style.display='none';cv.style.display='block';cv.style.height='420px';
  if(!chartIsInit) initChart();
  oiChart.data.datasets[0].label=isTotal?'Total CE OI':(strike==='OTM'?'OTM CE OI':'CE OI');
  oiChart.data.datasets[1].label=isTotal?'Total PE OI':(strike==='OTM'?'OTM PE OI':'PE OI');
  oiChart.data.labels=labels;
  oiChart.data.datasets[0].data=ceData;
  oiChart.data.datasets[1].data=peData;
  oiChart.update();
  const n=labels.length-1;
  updateChartMeta(strike,ceData[n],peData[n],labels.length,isTotal);
  document.getElementById('cf-strike').textContent=isTotal?'All Strikes (Total)':(isOtm?'OTM Strikes (±1 to ±5)':strike);
  document.getElementById('cf-from').textContent=labels[0]||'—';
  document.getElementById('cf-to').textContent=labels[n]||'—';
  document.getElementById('cf-pts').textContent=labels.length;
  document.querySelectorAll('.day-btn').forEach(b=>{b.classList.toggle('active',parseInt(b.dataset.days)===activeDays);});
}

function updateChartMeta(strike,ce,pe,pts,isTotal){
  const isOtm=(strike==='OTM');
  const ceLabel=isTotal?'Total CE:':(isOtm?'OTM CE:':'CE OI:');
  const peLabel=isTotal?'Total PE:':(isOtm?'OTM PE:':'PE OI:');
  const ceEl=document.querySelector('.cmeta.cm-ce');
  const peEl=document.querySelector('.cmeta.cm-pe');
  if(ceEl)ceEl.innerHTML=ceLabel+' <b id="ch-ce">'+(ce!=null?fmtK(ce):'—')+'</b>';
  if(peEl)peEl.innerHTML=peLabel+' <b id="ch-pe">'+(pe!=null?fmtK(pe):'—')+'</b>';
  const pcrEl=document.getElementById('ch-pcr');if(pcrEl)pcrEl.textContent=(ce&&pe)?(pe/ce).toFixed(4):'—';
  const ptsEl=document.getElementById('ch-pts');if(ptsEl)ptsEl.textContent=pts||0;
}

function mergeIntoLocal(strike,ts_arr,ce_arr,pe_arr){
  if(!localHistory[strike])localHistory[strike]={ts:[],ce:[],pe:[]};
  const combined={};
  localHistory[strike].ts.forEach((t,i)=>{combined[t]={ce:localHistory[strike].ce[i],pe:localHistory[strike].pe[i]};});
  ts_arr.forEach((t,i)=>{combined[t]={ce:ce_arr[i],pe:pe_arr[i]};});
  const sorted=Object.keys(combined).sort();
  localHistory[strike]={ts:sorted,ce:sorted.map(t=>combined[t].ce),pe:sorted.map(t=>combined[t].pe)};
}

function appendLivePoint(rows,tsNow){
  let sumCe=0,sumPe=0,otmCe=0,otmPe=0;
  for(const r of rows){
    const sk=parseInt(r.strike);
    if(!localHistory[sk])localHistory[sk]={ts:[],ce:[],pe:[]};
    const last=localHistory[sk].ts.slice(-1)[0];
    if(last!==tsNow){localHistory[sk].ts.push(tsNow);localHistory[sk].ce.push(r.ce_oi);localHistory[sk].pe.push(r.pe_oi);}
    sumCe+=(r.ce_oi||0);sumPe+=(r.pe_oi||0);
    if(r.offset>0)otmCe+=(r.ce_oi||0);if(r.offset<0)otmPe+=(r.pe_oi||0);
  }
  if(!localHistory['TOTAL'])localHistory['TOTAL']={ts:[],ce:[],pe:[]};
  const lastT=localHistory['TOTAL'].ts.slice(-1)[0];
  if(lastT!==tsNow){localHistory['TOTAL'].ts.push(tsNow);localHistory['TOTAL'].ce.push(sumCe);localHistory['TOTAL'].pe.push(sumPe);}
  else{const n=localHistory['TOTAL'].ts.length-1;localHistory['TOTAL'].ce[n]=sumCe;localHistory['TOTAL'].pe[n]=sumPe;}
  if(!localHistory['OTM'])localHistory['OTM']={ts:[],ce:[],pe:[]};
  const lastO=localHistory['OTM'].ts.slice(-1)[0];
  if(lastO!==tsNow){localHistory['OTM'].ts.push(tsNow);localHistory['OTM'].ce.push(otmCe);localHistory['OTM'].pe.push(otmPe);}
  else{const n=localHistory['OTM'].ts.length-1;localHistory['OTM'].ce[n]=otmCe;localHistory['OTM'].pe[n]=otmPe;}
}

async function loadHistoricalOI(strike){
  if(strike==='TOTAL'){await loadHistoricalTotal();return;}
  if(strike==='OTM'){await loadHistoricalOtm();return;}
  document.getElementById('chart-spinner').classList.add('show');
  document.getElementById('chart-ph').style.display='none';
  document.getElementById('oi-chart').style.display='none';
  try{
    const res=await fetch('/api/historical_oi?strike='+strike+'&_='+Date.now(), {cache:'no-store'});
    const data=await res.json();
    if(data.ts&&data.ts.length>0){
      mergeIntoLocal(strike,data.ts,data.ce,data.pe);
      const sb=document.getElementById('ch-src');
      if(data.source){sb.textContent=data.source==='historical+live'?'4-Day Data':'Live Only';sb.className='src-badge '+(data.source==='historical+live'?'src-hist':'src-live');sb.style.display='inline-block';}
    }
  }catch(e){console.warn('Historical OI fetch failed:',e);}
  renderChart(strike);
}

async function loadHistoricalTotal(){
  document.getElementById('chart-spinner').classList.add('show');
  document.getElementById('chart-ph').style.display='none';
  document.getElementById('oi-chart').style.display='none';
  try{
    const res=await fetch('/api/historical_oi_total?_='+Date.now(), {cache:'no-store'});const data=await res.json();
    if(data.ts&&data.ts.length>0){
      const combined={};
      const existing=localHistory['TOTAL']||{ts:[],ce:[],pe:[]};
      existing.ts.forEach((t,i)=>{combined[t]={ce:existing.ce[i],pe:existing.pe[i]};});
      data.ts.forEach((t,i)=>{combined[t]={ce:data.ce[i],pe:data.pe[i]};});
      const sorted=Object.keys(combined).sort();
      localHistory['TOTAL']={ts:sorted,ce:sorted.map(t=>combined[t].ce),pe:sorted.map(t=>combined[t].pe)};
      const sb=document.getElementById('ch-src');
      if(data.source){sb.textContent=data.source==='historical+live'?'4-Day Total':'Live Total';sb.className='src-badge '+(data.source==='historical+live'?'src-hist':'src-live');sb.style.display='inline-block';}
    }
  }catch(e){console.warn('loadHistoricalTotal failed:',e);}
  renderChart('TOTAL');
}

async function loadHistoricalOtm(){
  document.getElementById('chart-spinner').classList.add('show');
  document.getElementById('chart-ph').style.display='none';
  document.getElementById('oi-chart').style.display='none';
  try{
    const res=await fetch('/api/historical_oi_otm?_='+Date.now(), {cache:'no-store'});const data=await res.json();
    if(data.ts&&data.ts.length>0){
      const combined={};
      const existing=localHistory['OTM']||{ts:[],ce:[],pe:[]};
      existing.ts.forEach((t,i)=>{combined[t]={ce:existing.ce[i],pe:existing.pe[i]};});
      data.ts.forEach((t,i)=>{combined[t]={ce:data.ce[i],pe:data.pe[i]};});
      const sorted=Object.keys(combined).sort();
      localHistory['OTM']={ts:sorted,ce:sorted.map(t=>combined[t].ce),pe:sorted.map(t=>combined[t].pe)};
      const sb=document.getElementById('ch-src');
      if(data.source){sb.textContent=data.source==='historical+live'?'4-Day OTM':'Live OTM';sb.className='src-badge '+(data.source==='historical+live'?'src-hist':'src-live');sb.style.display='inline-block';}
    }
  }catch(e){console.warn('loadHistoricalOtm failed:',e);}
  renderChart('OTM');
}

function setDays(n){activeDays=n;if(chartStrike!==null)renderChart(chartStrike);}

let strikeBtnATM=null;
function buildStrikeButtons(rows,atm){
  const wrap=document.getElementById('strike-btns');
  const curSk=chartStrike?String(chartStrike):null;
  const newKeys='TOTAL,OTM,'+rows.map(r=>r.strike).join(',');
  if(wrap.dataset.keys===newKeys)return;
  wrap.dataset.keys=newKeys;wrap.innerHTML='';strikeBtnATM=null;
  const totalBtn=document.createElement('button');
  totalBtn.textContent='Total OI';totalBtn.className='sk-btn sk-total'+(curSk==='TOTAL'?' active':'');
  totalBtn.dataset.strike='TOTAL';totalBtn.onclick=()=>selectStrike('TOTAL',totalBtn);wrap.appendChild(totalBtn);
  const otmBtn=document.createElement('button');
  otmBtn.textContent='OTM OI';otmBtn.className='sk-btn sk-otm'+(curSk==='OTM'?' active':'');
  otmBtn.dataset.strike='OTM';otmBtn.title='OTM Only: ATM+1..+5 CE + ATM-1..-5 PE (excludes ATM)';
  otmBtn.onclick=()=>selectStrike('OTM',otmBtn);wrap.appendChild(otmBtn);
  const sep=document.createElement('span');sep.style.cssText='width:1px;height:20px;background:var(--brd);flex-shrink:0;margin:0 2px';wrap.appendChild(sep);
  rows.forEach(r=>{
    const btn=document.createElement('button');
    const off=r.strike-atm;const isAtm=off===0;const isCE=off>0;
    const label=isAtm?r.strike+' ATM':isCE?'+'+off+' ('+r.strike+')':off+' ('+r.strike+')';
    btn.textContent=label;btn.className='sk-btn'+(isAtm?' sk-atm':isCE?' sk-ce':' sk-pe')+(String(r.strike)===curSk?' active':'');
    btn.dataset.strike=r.strike;btn.onclick=()=>selectStrike(r.strike,btn);
    wrap.appendChild(btn);if(isAtm)strikeBtnATM=btn;
  });
  if(chartStrike===null&&strikeBtnATM){selectStrike(atm,strikeBtnATM);}
}

function selectStrike(strike,btnEl){
  document.querySelectorAll('.sk-btn').forEach(b=>b.classList.remove('active'));
  if(btnEl)btnEl.classList.add('active');
  if(strike==='TOTAL'){chartStrike='TOTAL';if(localHistory['TOTAL']&&localHistory['TOTAL'].ts.length>0)renderChart('TOTAL');loadHistoricalTotal();return;}
  if(strike==='OTM'){chartStrike='OTM';if(localHistory['OTM']&&localHistory['OTM'].ts.length>0)renderChart('OTM');loadHistoricalOtm();return;}
  chartStrike=parseInt(strike);
  if(localHistory[chartStrike]&&localHistory[chartStrike].ts.length>0)renderChart(chartStrike);
  loadHistoricalOI(chartStrike);
}

/* ════════════════════════════════════════════════════════════
   OI FETCH
════════════════════════════════════════════════════════════ */
async function fetchOI(){
  try{
    const res=await fetch('/api/oi?_='+Date.now(), {cache:'no-store'});const j=await res.json();
    if(j.error){document.getElementById('err-box').style.display='block';document.getElementById('err-box').textContent='OI Error: '+j.error;return;}
    document.getElementById('err-box').style.display='none';
    const d=j.data;if(!d||!d.atm)return;
    document.getElementById('chip-time').textContent=j.updated_at||'—';
    document.getElementById('chip-exp').textContent=d.expiry||'—';
    oiCountdown=OI_INT;
    document.getElementById('cv-atm').textContent=fmtN(d.atm);
    document.getElementById('cv-tce').textContent=fmtK(d.total_ce);
    document.getElementById('cv-tpe').textContent=fmtK(d.total_pe);
    document.getElementById('cv-exp').textContent=d.expiry||'—';
    const pcr=d.pcr||0;
    const[sent,col,pct]=pcrInfo(pcr);
    const pcrEl1=document.getElementById('cv-pcr');const pcrEl2=document.getElementById('pcr-big');
    if(pcrEl1){pcrEl1.textContent=pcr.toFixed(4);pcrEl1.style.color=col;}
    if(pcrEl2){pcrEl2.textContent=pcr.toFixed(4);pcrEl2.style.color=col;}
    document.getElementById('cv-pcr-s').textContent=sent;
    document.getElementById('pcr-fill').style.width=pct+'%';
    document.getElementById('pcr-fill').style.background=col;
    document.getElementById('pcr-sent').textContent=sent;
    document.getElementById('pcr-sent').style.color=col;
    cachedRows=d.rows||[];
    const mxCE=Math.max(...cachedRows.map(r=>r.ce_oi||0),1);
    const mxPE=Math.max(...cachedRows.map(r=>r.pe_oi||0),1);
    document.getElementById('oi-tbody').innerHTML=cachedRows.map(r=>buildRow(r,mxCE,mxPE)).join('');
    document.getElementById('foot-tce').textContent='Total CE: '+fmtK(d.total_ce);
    document.getElementById('foot-tpe').textContent='Total PE: '+fmtK(d.total_pe);
    const fp=document.getElementById('foot-pcr');if(fp){fp.textContent='PCR: '+pcr.toFixed(4);fp.style.color=col;}
    const tsNow=d.ts_chart||(new Date().toLocaleDateString('en-IN',{day:'2-digit',month:'short'}).replace(' ','-')+' '+new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',hour12:false}));
    appendLivePoint(cachedRows,tsNow);
    buildStrikeButtons(cachedRows,d.atm);
    buildPvStrikeCol(cachedRows,d.atm);
    if(chartStrike!==null){
      if(chartStrike==='TOTAL')renderChart('TOTAL');
      else if(chartStrike==='OTM')renderChart('OTM');
      else if(localHistory[chartStrike])renderChart(chartStrike);
    }
    // Also refresh IV table on every OI update
    fetchIV();
  }catch(e){console.error('[OI] fetchOI error:',e);document.getElementById('err-box').style.display='block';document.getElementById('err-box').textContent='JS Error in fetchOI: '+e.message;}
}

async function fetchLTP(){
  try{
    const res=await fetch('/api/ltp?_='+Date.now(), {cache:'no-store'});const data=await res.json();
    const now=new Date().toLocaleTimeString('en-IN');
    const sv=data['NSE:NIFTY 50'];
    if(sv!==undefined){
      const el=document.getElementById('cv-spot');const pv=prevLTP['NSE:NIFTY 50'];
      el.textContent=fmtP(sv);
      if(pv!==undefined)el.style.color=sv>pv?'var(--ce)':sv<pv?'var(--pe)':'var(--gold)';
      document.getElementById('cv-spot-t').textContent='as of '+now;
      prevLTP['NSE:NIFTY 50']=sv;
    }
    for(const r of cachedRows){
      const cev=data[r.ce_sym];
      if(cev!==undefined){
        // Update OI table CE LTP
        const el=document.getElementById('celtp'+r.strike);
        if(el){flash('cetd'+r.strike,cev,prevLTP[r.ce_sym]);el.textContent=fmtP(cev);}
        // Update IV table CE LTP (live, every 1s)
        const ivEl=document.getElementById('iv-ce-ltp-'+r.strike);
        if(ivEl){
          const pv2=prevLTP[r.ce_sym];
          if(pv2!==undefined){
            ivEl.style.color=cev>pv2?'var(--ce)':cev<pv2?'var(--pe)':'var(--ce)';
          }
          ivEl.textContent=fmtP(cev);
        }
        prevLTP[r.ce_sym]=cev;
      }
      const pev=data[r.pe_sym];
      if(pev!==undefined){
        // Update OI table PE LTP
        const el=document.getElementById('peltp'+r.strike);
        if(el){flash('petd'+r.strike,pev,prevLTP[r.pe_sym]);el.textContent=fmtP(pev);}
        // Update IV table PE LTP (live, every 1s)
        const ivEl=document.getElementById('iv-pe-ltp-'+r.strike);
        if(ivEl){
          const pv2=prevLTP[r.pe_sym];
          if(pv2!==undefined){
            ivEl.style.color=pev>pv2?'var(--pe)':pev<pv2?'var(--pe)':'var(--pe)';
          }
          ivEl.textContent=fmtP(pev);
        }
        prevLTP[r.pe_sym]=pev;
      }
    }
  }catch(e){}
}

/* ════════════════════════════════════════════════════════════
   PRICE / VWAP CHARTS  (unchanged)
════════════════════════════════════════════════════════════ */
let pvStrike=null,pvTf=1,ceChart=null,peChart=null,pvInterval=null;
let pvCeSym='',pvPeSym='';

function setPvTf(tf){
  pvTf=tf;
  document.querySelectorAll('.tf-btn').forEach(b=>{b.classList.toggle('active',parseInt(b.dataset.tf)===tf);});
  document.getElementById('pv-foot-tf').textContent=tf+' min';
  if(pvStrike)loadPvData(pvStrike,pvCeSym,pvPeSym);
}

function buildPvStrikeCol(rows,atm){
  const col=document.getElementById('pv-strike-col');
  const keys=rows.map(r=>r.strike).join(',');
  if(col.dataset.keys!==keys){
    col.dataset.keys=keys;col.innerHTML='<div class="pv-sk-lbl">Strike</div>';
    rows.forEach(r=>{
      const btn=document.createElement('button');
      const off=r.strike-atm;const isAtm=off===0;
      btn.textContent=r.strike;btn.title=isAtm?'★ ATM':(off>0?`CE +${off}`:`PE ${off}`);
      btn.className='pv-sk-btn'+(isAtm?' pv-sk-atm':'');
      btn.dataset.strike=r.strike;btn.dataset.ceSym=r.ce_sym;btn.dataset.peSym=r.pe_sym;
      btn.onclick=()=>selectPvStrike(parseInt(r.strike),r.ce_sym,r.pe_sym,btn);
      col.appendChild(btn);
    });
    if(pvStrike===null){
      const atmBtn=col.querySelector('.pv-sk-atm');
      if(atmBtn){const r=rows.find(x=>x.strike===atm);if(r)selectPvStrike(atm,r.ce_sym,r.pe_sym,atmBtn);}
      return;
    }
  }
  col.querySelectorAll('.pv-sk-btn').forEach(b=>{b.classList.toggle('active',parseInt(b.dataset.strike)===pvStrike);});
}

function selectPvStrike(strike,ceSym,peSym,btnEl){
  document.querySelectorAll('#pv-strike-col .pv-sk-btn').forEach(b=>b.classList.remove('active'));
  if(btnEl)btnEl.classList.add('active');
  pvStrike=parseInt(strike);pvCeSym=ceSym;pvPeSym=peSym;
  loadPvData(pvStrike,ceSym,peSym);
}

function makePvChart(canvasId,priceColor,vwapColor){
  const p=palette();
  const ctx=document.getElementById(canvasId).getContext('2d');
  const nowLinePlugin={
    id:'nowLine',
    afterDraw(chart){
      const priceData=chart.data.datasets[0].data;if(!priceData||!priceData.length)return;
      let lastReal=-1;
      for(let i=priceData.length-1;i>=0;i--){if(priceData[i]!==null&&priceData[i]!==undefined){lastReal=i;break;}}
      if(lastReal<0)return;
      const xAxis=chart.scales.x;const yAxis=chart.scales.y;
      const ctx2=chart.ctx;const top=yAxis.top;const bottom=yAxis.bottom;const right=xAxis.right;
      if(lastReal<priceData.length-1){
        const xNow=xAxis.getPixelForValue(lastReal);
        ctx2.save();
        const futureAlpha=isDayMode?0.06:0.08;
        ctx2.fillStyle=isDayMode?'rgba(0,0,0,'+futureAlpha+')':'rgba(255,255,255,'+futureAlpha+')';
        ctx2.fillRect(xNow,top,right-xNow,bottom-top);
        ctx2.strokeStyle=isDayMode?'rgba(154,103,0,0.80)':'rgba(251,191,36,0.80)';
        ctx2.lineWidth=1.5;ctx2.setLineDash([]);ctx2.beginPath();ctx2.moveTo(xNow,top);ctx2.lineTo(xNow,bottom);ctx2.stroke();
        const timeLabel=chart.data.labels[lastReal]||'';
        ctx2.fillStyle=isDayMode?'rgba(154,103,0,0.90)':'rgba(251,191,36,0.90)';
        ctx2.font="bold 9px 'IBM Plex Mono', monospace";ctx2.textAlign='center';ctx2.fillText(timeLabel,xNow,bottom+12);
        ctx2.restore();
      }
    }
  };
  return new Chart(ctx,{
    type:'line',plugins:[nowLinePlugin],
    data:{labels:[],datasets:[
      {label:'Price',data:[],yAxisID:'y',borderColor:priceColor,borderWidth:2,pointRadius:0,pointHoverRadius:6,pointHoverBackgroundColor:priceColor,pointHoverBorderColor:'#fff',pointHoverBorderWidth:2,tension:0.3,fill:false,spanGaps:false},
      {label:'VWAP',data:[],yAxisID:'y',borderColor:vwapColor,borderWidth:1.8,borderDash:[5,4],pointRadius:0,pointHoverRadius:5,pointHoverBackgroundColor:vwapColor,tension:0.3,fill:false,spanGaps:false},
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,animation:{duration:300},layout:{padding:{bottom:18}},
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{backgroundColor:p.tip,borderColor:p.tipbrd,borderWidth:1,titleColor:p.ttitle,bodyColor:p.tbody,padding:12,titleFont:{family:"'IBM Plex Mono'",size:11,weight:'700'},bodyFont:{family:"'IBM Plex Mono'",size:11},filter:item=>item.raw!==null&&item.raw!==undefined,callbacks:{title:items=>items[0].label,label:item=>item.raw!=null?'  '+item.dataset.label+': Rs.'+Number(item.raw).toFixed(2):''}}},
      scales:{x:{grid:{display:false},ticks:{color:p.tick,font:{family:"'IBM Plex Mono'",size:10},maxRotation:0,autoSkip:false,callback(val,idx){const lbl=this.getLabelForValue(val);if(!lbl)return'';const[hh,mm]=lbl.split(':').map(Number);if(lbl==='09:15'||lbl==='15:30')return lbl;if(mm===0)return lbl;return'';}},border:{color:cssv('--brd')}},y:{position:'right',grid:{display:false},ticks:{color:p.tick,font:{family:"'IBM Plex Mono'",size:10},callback:v=>v!=null?'Rs.'+Number(v).toFixed(0):''},border:{color:cssv('--brd')}}}
    }
  });
}

function updatePvChartColors(chart,priceColor,vwapColor){
  if(!chart)return;
  const p=palette();
  chart.data.datasets[0].borderColor=priceColor;chart.data.datasets[0].pointHoverBackgroundColor=priceColor;
  chart.data.datasets[1].borderColor=vwapColor;
  chart.options.plugins.tooltip.backgroundColor=p.tip;chart.options.plugins.tooltip.borderColor=p.tipbrd;
  chart.options.plugins.tooltip.titleColor=p.ttitle;chart.options.plugins.tooltip.bodyColor=p.tbody;
  chart.options.scales.x.ticks.color=p.tick;chart.options.scales.x.border.color=cssv('--brd');
  chart.options.scales.y.ticks.color=p.tick;chart.options.scales.y.border.color=cssv('--brd');
  chart.update('none');
}

function renderPvChart(chartRef,canvasId,phId,spinId,series,latestPrice,latestVwap,metaPriceId,metaVwapId){
  const ph=document.getElementById(phId);const spin=document.getElementById(spinId);const cv=document.getElementById(canvasId);
  spin.classList.remove('show');
  if(!series||series.ts.length===0){ph.style.display='flex';cv.style.display='none';return null;}
  ph.style.display='none';cv.style.display='block';cv.style.height='360px';
  if(!chartRef){
    const isCe=canvasId==='ce-chart';
    chartRef=makePvChart(canvasId,isCe?cssv('--ce'):cssv('--pe'),isCe?'#a0e8ff':'#ffb3c8');
    chartRef.resize();
  }
  chartRef.data.labels=[...series.ts];
  chartRef.data.datasets[0].data=[...series.price];
  chartRef.data.datasets[1].data=[...series.vwap];
  chartRef.update();
  if(latestPrice!=null)document.getElementById(metaPriceId).textContent='₹'+latestPrice.toFixed(2);
  if(latestVwap!=null)document.getElementById(metaVwapId).textContent='₹'+latestVwap.toFixed(2);
  return chartRef;
}

async function loadPvData(strike,ceSym,peSym){
  ceSym=ceSym||pvCeSym;peSym=peSym||pvPeSym;
  if(!ceSym||!peSym){console.warn('PV: no symbols available yet for strike',strike);return;}
  document.getElementById('pv-ce-spin').classList.add('show');
  document.getElementById('pv-pe-spin').classList.add('show');
  document.getElementById('pv-ce-ph').style.display='none';
  document.getElementById('pv-pe-ph').style.display='none';
  try{
    const url=`/api/price_vwap?ce_sym=${encodeURIComponent(ceSym)}&pe_sym=${encodeURIComponent(peSym)}&tf=${pvTf}&_=${Date.now()}`;
    const res=await fetch(url, {cache:'no-store'});const data=await res.json();
    if(data.error){
      ['pv-ce-spin','pv-pe-spin'].forEach(id=>document.getElementById(id).classList.remove('show'));
      ['pv-ce-ph','pv-pe-ph'].forEach(id=>{const el=document.getElementById(id);el.style.display='flex';const p=el.querySelector('p');if(p)p.innerHTML=`<span style="font-size:.6rem;color:var(--err-cl)">${data.error}</span>`;});
      return;
    }
    const lastReal=arr=>{for(let i=arr.length-1;i>=0;i--){if(arr[i]!=null)return arr[i];}return null;};
    ceChart=renderPvChart(ceChart,'ce-chart','pv-ce-ph','pv-ce-spin',data.ce,lastReal(data.ce.price),lastReal(data.ce.vwap),'pv-ce-ltp','pv-ce-vwap');
    peChart=renderPvChart(peChart,'pe-chart','pv-pe-ph','pv-pe-spin',data.pe,lastReal(data.pe.price),lastReal(data.pe.vwap),'pv-pe-ltp','pv-pe-vwap');
    document.getElementById('pv-foot-strike').textContent=strike;
    document.getElementById('pv-foot-tf').textContent=pvTf+' min';
    document.getElementById('pv-foot-ce-pts').textContent=data.ce.ts.length;
    document.getElementById('pv-foot-pe-pts').textContent=data.pe.ts.length;
  }catch(e){console.warn('PV fetch error:',e);['pv-ce-spin','pv-pe-spin'].forEach(id=>document.getElementById(id).classList.remove('show'));}
}

function startPvRefresh(){
  if(pvInterval)clearInterval(pvInterval);
  pvInterval=setInterval(()=>{if(pvStrike&&pvCeSym)loadPvData(pvStrike,pvCeSym,pvPeSym);},60000);
}

/* ── Boot ── */
async function boot(){
  await fetchOI();
  setInterval(fetchOI, OI_INT * 1000);
  setTimeout(()=>{ fetchLTP(); setInterval(fetchLTP, LTP_INT * 1000); }, 900);
}

boot();
startPvRefresh();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, oi_interval=OI_INTERVAL, ltp_interval=LTP_INTERVAL)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


def start_dashboard_runtime():
    global _startup_done
    with _startup_lock:
        if _startup_done:
            return
        print("="*70)
        print("  NIFTY OI Dashboard  v7  (Long Buildup + Short Covering + LTP fix)")
        print(f"  OI refresh : {OI_INTERVAL}s   LTP refresh : {LTP_INTERVAL}s")
        print(f"  Strikes    : ATM±{OTM_DEPTH}  ({2*OTM_DEPTH+1} rows total)")
        print(f"  IV Method  : Black-Scholes Newton-Raphson (pure Python)")
        print(f"  Risk-Free  : {RISK_FREE*100:.1f}%  |  EMA Period: 9")
        print(f"  IV TF opts : 1 / 3 / 5 / 10 min")
        print(f"  Buildup    : Price ±5% & OI ±5% threshold (per TF bar)")
        print("="*70)
        print("  → Fetching instruments + initial OI …")
        fetch_oi()
        fetch_ltp()
        if error_msg:
            print(f"\n  ⚠  ERROR: {error_msg}\n")
        else:
            print(f"\n  ✓  Expiry   = {oi_data.get('expiry')}")
            print(f"  ✓  ATM      = {oi_data.get('atm')}")
            print(f"  ✓  Symbols  = {len(ltp_symbols)} tracked")
            print(f"  ✓  Tokens   = {len(_token_map)} loaded for historical")
            print(f"  ✓  IV rows  = {len(iv_history)} strikes computed")
            for sk, sides in list(iv_history.items())[:3]:
                ce_iv = sides['ce'][-1]['iv'] if sides['ce'] else None
                pe_iv = sides['pe'][-1]['iv'] if sides['pe'] else None
                print(f"     Strike {sk}:  CE IV={ce_iv}%  PE IV={pe_iv}%")
        threading.Thread(target=oi_loop, daemon=True, name="OI-Thread").start()
        threading.Thread(target=ltp_loop, daemon=True, name="LTP-Thread").start()
        print(f"\n  ▶  App ready on port {PORT}\n")
        _startup_done = True


@app.before_request
def ensure_runtime_started():
    if not _startup_done:
        start_dashboard_runtime()


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    start_dashboard_runtime()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
else:
    start_dashboard_runtime()
