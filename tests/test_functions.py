import json
import logging
from pathlib import Path
import tempfile

import pytest

from functions import (
    calculate_dynamic_order_size,
    apply_trailing_stop,
    partial_take_profit,
    limit_open_positions,
    select_strategy,
    dynamic_leverage,
    reload_config,
    cache_candles,
    log_performance_metrics,
    sma_crossover,
    breakout,
    mean_reversion,
    rsi_strategy,
    half_year_strategy,
    apply_sl_tp_bounds,
)

def test_calculate_dynamic_order_size():
    size = calculate_dynamic_order_size(10000, 0.5, risk_pct=2)
    assert pytest.approx(size, rel=1e-5) == 133.3333333333


def test_apply_trailing_stop_buy():
    sl = apply_trailing_stop("Buy", 100, 120, 10, current_sl=105)
    assert sl == 108


def test_apply_trailing_stop_sell():
    sl = apply_trailing_stop("Sell", 100, 80, 10, current_sl=95)
    assert sl == 88


def test_partial_take_profit():
    orders = partial_take_profit(100, [(110, 50), (120, 25)])
    assert orders == [{"price": 110, "qty": 50}, {"price": 120, "qty": 25}]


def test_limit_open_positions():
    assert limit_open_positions(["BTC"], 2)
    assert not limit_open_positions(["BTC", "ETH"], 2)


def test_select_strategy():
    func = select_strategy({"strategy": "breakout"})
    assert func is breakout
    with pytest.raises(ValueError):
        select_strategy({"strategy": "unknown"})


def test_dynamic_leverage():
    assert dynamic_leverage(10, 0.5) == 6
    assert dynamic_leverage(10, 5) == 1


def test_reload_config():
    current = {"a": 1}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "cfg.json"
        cfg.write_text(json.dumps({"b": 2}))
        new = reload_config(cfg, current)
        assert new == {"a": 1, "b": 2}


def test_apply_sl_tp_bounds():
    price = 100
    stop, take = apply_sl_tp_bounds(price, "Buy", 99.9, 100.2)
    assert stop == pytest.approx(98.5)
    assert take == pytest.approx(101.5)
    stop, take = apply_sl_tp_bounds(price, "Buy", 50, 200)
    assert stop == pytest.approx(95)
    assert take == pytest.approx(105)


def test_cache_candles(tmp_path):
    path = cache_candles("BTCUSDT", [[1, 2, 3]], cache_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data == [[1, 2, 3]]


def test_log_performance_metrics(tmp_path):
    log_file = tmp_path / "perf.log"
    log_performance_metrics(0.1, 0.2, log_file)
    content = log_file.read_text()
    assert "api_latency=0.100s" in content
    assert "strategy_time=0.200s" in content


def _make_candles(prices):
    return [[0, p, p + 1, p - 1, p, 0] for p in prices]


def test_half_year_strategy_buy_sell():
    up_prices = list(range(1, 260))
    candles = _make_candles(up_prices)
    signal, stop, take = half_year_strategy(candles)
    assert signal == "Buy"
    assert stop < up_prices[-1] < take

    down_prices = list(range(260, 0, -1))
    candles = _make_candles(down_prices)
    signal, stop, take = half_year_strategy(candles)
    assert signal == "Sell"
    assert take < down_prices[-1] < stop



def test_half_year_strategy_scaling():
    prices = list(range(1, 260))
    candles = _make_candles(prices)
    signal, stop, take = half_year_strategy(candles, k=2)

    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]

    def _atr(high, low, close, period=14):
        trs = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            trs.append(tr)
        return sum(trs[-period:]) / period

    atr_val = _atr(highs, lows, closes) / 48
    price = closes[-1]

    assert signal == "Buy"
    expected = price * 0.015
    assert pytest.approx(price - stop, rel=1e-6) == expected
    assert pytest.approx(take - price, rel=1e-6) == expected


def test_sma_crossover_signals():
    up = _make_candles(range(1, 40))
    signal, _, _ = sma_crossover(up)
    assert signal == "Buy"
    down = _make_candles(range(40, 0, -1))
    signal, _, _ = sma_crossover(down)
    assert signal == "Sell"


def test_breakout_buy():
    prices = [1] * 20 + [5]
    signal, _, _ = breakout(_make_candles(prices))
    assert signal == "Buy"


def test_mean_reversion_sell():
    prices = [100] * 20 + [120]
    signal, _, target = mean_reversion(_make_candles(prices))
    assert signal == "Sell"
    assert target == pytest.approx(114.0)


def test_rsi_strategy_buy_sell():
    prices = list(range(100, 84, -1))
    signal, _, _ = rsi_strategy(_make_candles(prices))
    assert signal == "Buy"
    prices = list(range(84, 100))
    signal, _, _ = rsi_strategy(_make_candles(prices))
    assert signal == "Sell"

