from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import statistics

__all__ = [
    "calculate_dynamic_order_size",
    "apply_trailing_stop",
    "partial_take_profit",
    "limit_open_positions",
    "select_strategy",
    "dynamic_leverage",
    "reload_config",
    "cache_candles",
    "log_performance_metrics",
    "sma_crossover",
    "breakout",
    "mean_reversion",
    "rsi_strategy",
    "half_year_strategy",
    "apply_sl_tp_bounds",
]


def calculate_dynamic_order_size(balance: float, volatility: float, risk_pct: float = 1.0) -> float:
    """Return position size as percentage of balance adjusted by volatility.

    The higher the volatility, the smaller the resulting size. ``risk_pct`` is the
    percentage of the balance to use when volatility is zero.
    """

    if balance < 0:
        raise ValueError("balance must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if risk_pct <= 0:
        raise ValueError("risk_pct must be positive")
    return balance * (risk_pct / 100.0) / (1 + volatility)


def apply_trailing_stop(
    side: str,
    entry_price: float,
    current_price: float,
    trail_pct: float,
    current_sl: Optional[float] = None,
) -> float:
    """Return new stop-loss level applying a trailing percentage.

    ``side`` must be ``"Buy"`` for long positions or ``"Sell"`` for shorts.
    ``current_sl`` is the existing stop-loss (if any).
    """

    if trail_pct <= 0:
        raise ValueError("trail_pct must be positive")
    if side not in {"Buy", "Sell"}:
        raise ValueError("side must be 'Buy' or 'Sell'")

    if side == "Buy":
        base_sl = entry_price * (1 - trail_pct / 100)
        new_sl = current_price * (1 - trail_pct / 100)
        return max(current_sl or base_sl, new_sl)
    else:
        base_sl = entry_price * (1 + trail_pct / 100)
        new_sl = current_price * (1 + trail_pct / 100)
        return min(current_sl or base_sl, new_sl)


def partial_take_profit(
    position_qty: float, targets: Sequence[Tuple[float, float]]
) -> List[Dict[str, float]]:
    """Return list of orders closing parts of the position.

    ``targets`` is a sequence of ``(price, percent)`` pairs where ``percent`` is the
    percentage of the original position size to close at that price.
    """

    orders = []
    for price, percent in targets:
        if percent <= 0:
            raise ValueError("percent must be positive")
        qty = position_qty * percent / 100.0
        orders.append({"price": price, "qty": qty})
    return orders


def limit_open_positions(open_positions: Sequence[str], limit: int) -> bool:
    """Return ``True`` if a new position can be opened without exceeding ``limit``."""

    return len(open_positions) < limit


# Helper for strategy ATR calculations

def _atr_from_candles(candles: Sequence[Sequence[float]], period: int = 14) -> float:
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    trs: List[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs[-period:]) / period


def apply_sl_tp_bounds(
    price: float,
    side: str,
    stop: float,
    take: float,
    min_pct: float = 1.5,
    max_pct: float = 5.0,
) -> tuple[float, float]:
    """Clamp stop-loss and take-profit distances to ``min_pct``–``max_pct`` of price."""

    min_dist = price * (min_pct / 100)
    max_dist = price * (max_pct / 100)
    stop_dist = abs(price - stop)
    take_dist = abs(take - price)
    stop_dist = max(min_dist, min(stop_dist, max_dist))
    take_dist = max(min_dist, min(take_dist, max_dist))
    if side == "Buy":
        stop = price - stop_dist
        take = price + take_dist
    else:
        stop = price + stop_dist
        take = price - take_dist
    return stop, take


def sma_crossover(
    candles: Sequence[Sequence[float]], short: int = 5, long: int = 20, k: float = 2.0
) -> Tuple[str, float, float]:
    """Simple SMA crossover strategy returning signal, stop-loss and take-profit."""

    closes = [float(c[4]) for c in candles]
    if len(closes) < long:
        raise ValueError("not enough data")
    short_sma = sum(closes[-short:]) / short
    long_sma = sum(closes[-long:]) / long
    price = closes[-1]
    atr = _atr_from_candles(candles)
    if short_sma > long_sma:
        signal = "Buy"
        stop = price - atr
        take = price + k * atr
    elif short_sma < long_sma:
        signal = "Sell"
        stop = price + atr
        take = price - k * atr
    else:
        return "Hold", price, price
    stop, take = apply_sl_tp_bounds(price, signal, stop, take)
    return signal, stop, take


def breakout(
    candles: Sequence[Sequence[float]], lookback: int = 20, k: float = 2.0
) -> Tuple[str, float, float]:
    """Breakout strategy based on recent highs/lows."""

    if len(candles) < lookback + 1:
        raise ValueError("not enough data")
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    prev_high = max(highs[-(lookback + 1) : -1])
    prev_low = min(lows[-(lookback + 1) : -1])
    price = closes[-1]
    atr = _atr_from_candles(candles)
    if price > prev_high:
        signal = "Buy"
        stop = price - atr
        take = price + k * atr
    elif price < prev_low:
        signal = "Sell"
        stop = price + atr
        take = price - k * atr
    else:
        return "Hold", price, price
    stop, take = apply_sl_tp_bounds(price, signal, stop, take)
    return signal, stop, take


def mean_reversion(
    candles: Sequence[Sequence[float]], period: int = 20, k: float = 2.0
) -> Tuple[str, float, float]:
    """Mean reversion using Bollinger bands."""

    closes = [float(c[4]) for c in candles]
    if len(closes) < period:
        raise ValueError("not enough data")
    sma = sum(closes[-period:]) / period
    std = statistics.pstdev(closes[-period:])
    upper = sma + k * std
    lower = sma - k * std
    price = closes[-1]
    atr = _atr_from_candles(candles)
    if price < lower:
        signal = "Buy"
        stop = price - atr
        take = sma
    elif price > upper:
        signal = "Sell"
        stop = price + atr
        take = sma
    else:
        return "Hold", price, price
    stop, take = apply_sl_tp_bounds(price, signal, stop, take)
    return signal, stop, take


def rsi_strategy(
    candles: Sequence[Sequence[float]], period: int = 14, k: float = 2.0
) -> Tuple[str, float, float]:
    """RSI overbought/oversold strategy."""

    closes = [float(c[4]) for c in candles]
    if len(closes) < period + 1:
        raise ValueError("not enough data")
    gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, len(closes))]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    rsi = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    price = closes[-1]
    atr = _atr_from_candles(candles)
    if rsi < 30:
        signal = "Buy"
        stop = price - atr
        take = price + k * atr
    elif rsi > 70:
        signal = "Sell"
        stop = price + atr
        take = price - k * atr
    else:
        return "Hold", price, price
    stop, take = apply_sl_tp_bounds(price, signal, stop, take)
    return signal, stop, take


def select_strategy(
    config: Dict[str, str],
    strategies: Optional[Dict[str, Callable]] = None,
) -> Callable:
    """Select a trading strategy function based on configuration."""

    strategies = strategies or {
        "sma_crossover": sma_crossover,
        "breakout": breakout,
        "mean_reversion": mean_reversion,
        "rsi": rsi_strategy,
    }
    name = config.get("strategy")
    if name not in strategies:
        raise ValueError("Unknown strategy")
    return strategies[name]


def dynamic_leverage(
    base_leverage: int,
    volatility: float,
    max_leverage: int = 20,
    min_leverage: int = 1,
) -> int:
    """Adjust leverage based on volatility."""

    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    lev = int(base_leverage / (1 + volatility))
    return max(min_leverage, min(max_leverage, lev))


def reload_config(config_path: str | Path, current_config: Dict) -> Dict:
    """Reload configuration from JSON file and update ``current_config``."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        new_conf = json.load(f)
    current_config.update(new_conf)
    return current_config


def cache_candles(
    symbol: str, candles: Iterable, cache_dir: str | Path = "cache"
) -> Path:
    """Save candles for ``symbol`` to ``cache_dir`` as JSON and return the path."""

    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{symbol}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(list(candles), f)
    return file_path


def log_performance_metrics(
    api_latency: float,
    strategy_time: float,
    log_file: str | Path = "performance.log",
) -> None:
    """Append API and strategy timing metrics to ``log_file``."""

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("performance")
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info(
        "api_latency=%.3fs strategy_time=%.3fs", api_latency, strategy_time
    )
    handler.close()
    logger.handlers.clear()


def half_year_strategy(
    candles: Sequence[Sequence[float]], k: float = 2.0
) -> Tuple[str, float, float]:
    """Return trade signal with stop-loss and take-profit from 6 month data.

    ``candles`` should contain at least 200 entries of ``[ts, open, high, low, close, volume]``
    representing 4-hour bars. The function analyses trend using EMA50/EMA200 and RSI14

    and computes ATR14 for volatility. ATR is converted to an equivalent 5-minute value
    by dividing by ``48`` so that stop-loss uses ``±ATR_5m`` and take-profit uses
    ``±k*ATR_5m`` depending on the trade direction.

    """

    if len(candles) < 200:
        raise ValueError("need at least 200 candles")

    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]

    def ema(values: Sequence[float], period: int) -> float:
        factor = 2 / (period + 1)
        ema_val = values[0]
        for price in values[1:]:
            ema_val = price * factor + ema_val * (1 - factor)
        return ema_val

    def rsi(values: Sequence[float], period: int = 14) -> float:
        gains = []
        losses = []
        for i in range(1, len(values)):
            change = values[i] - values[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> float:
        trs = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            trs.append(tr)
        return sum(trs[-period:]) / period

    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    rsi_val = rsi(closes)
    atr_val = atr(highs, lows, closes)

    atr_5m = atr_val / 48  # convert 4h ATR to 5m equivalent

    price = closes[-1]

    if ema50 > ema200 and rsi_val > 50:
        signal = "Buy"
        stop = price - atr_5m
        take = price + k * atr_5m
    elif ema50 < ema200 and rsi_val < 50:
        signal = "Sell"
        stop = price + atr_5m
        take = price - k * atr_5m
    else:
        signal = "Hold"
        stop = take = price
        return signal, stop, take
    stop, take = apply_sl_tp_bounds(price, signal, stop, take)
    return signal, stop, take
