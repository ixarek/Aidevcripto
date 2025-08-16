import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import logging
import tempfile

# Добавляем путь к родительской папке, чтобы импортировать bot.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot import BybitTradingBot, setup_logging, logger  # импорт бота и логгера


class TestBybitTradingBot(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock()
        self.session.get_tickers.return_value = {
            "result": {"list": [{"lastPrice": "100"}]}
        }
        self.session.get_instruments_info.return_value = {
            "result": {
                "list": [
                    {
                        "lotSizeFilter": {
                            "qtyStep": "0.001",
                            "minOrderQty": "0.001",
                        },
                        "priceFilter": {"tickSize": "0.5"},
                    }
                ]
            }
        }
        self.bot = BybitTradingBot(self.session)

    def test_place_order_calls_api(self):
        self.bot.place_order("BTCUSDT", "Buy", 100, 10, 95, 105)
        self.session.set_leverage.assert_called_once_with(
            category="linear",
            symbol="BTCUSDT",
            buyLeverage="10",
            sellLeverage="10",
        )
        self.session.get_tickers.assert_called_once_with(
            category="linear", symbol="BTCUSDT"
        )
        self.session.get_instruments_info.assert_called_once_with(
            category="linear", symbol="BTCUSDT"
        )
        self.session.place_order.assert_called_once()
        _, kwargs = self.session.place_order.call_args
        self.assertEqual(kwargs["category"], "linear")
        self.assertEqual(kwargs["symbol"], "BTCUSDT")
        self.assertEqual(kwargs["side"], "Buy")
        self.assertEqual(kwargs["orderType"], "Market")
        self.assertEqual(kwargs["timeInForce"], "ImmediateOrCancel")
        self.assertAlmostEqual(float(kwargs["stopLoss"]), 95.0)
        self.assertAlmostEqual(float(kwargs["takeProfit"]), 105.0)
        self.assertEqual(kwargs["tpslMode"], "Partial")
        self.assertAlmostEqual(float(kwargs["qty"]), 10.0)

    def test_close_position_calls_api(self):
        self.bot.close_position("BTCUSDT", "Buy", 100, 10)
        self.session.place_order.assert_called_once()
        _, kwargs = self.session.place_order.call_args
        self.assertEqual(kwargs["category"], "linear")
        self.assertEqual(kwargs["symbol"], "BTCUSDT")
        self.assertEqual(kwargs["side"], "Sell")
        self.assertEqual(kwargs["orderType"], "Market")
        self.assertEqual(kwargs["timeInForce"], "ImmediateOrCancel")
        self.assertTrue(kwargs["reduceOnly"])
        self.assertAlmostEqual(float(kwargs["qty"]), 10.0)

    def test_invalid_amount(self):
        with self.assertRaises(ValueError):
            self.bot.place_order("BTCUSDT", "Buy", 70, 10, 95, 105)

    def test_invalid_leverage(self):
        with self.assertRaises(ValueError):
            self.bot.place_order("BTCUSDT", "Buy", 100, 5, 95, 105)

    def test_invalid_symbol(self):
        with self.assertRaises(ValueError):
            self.bot.place_order("ADAUSDT", "Buy", 100, 10, 95, 105)

    def test_invalid_position_value(self):
        with self.assertRaises(ValueError):
            self.bot.place_order("BTCUSDT", "Buy", 120, 11, 95, 105)

    def test_log_market_trend_increase(self):
        candles = [[0, 0, 0, 0, "110", 0]] + [[0, 0, 0, 0, "100", 0]] * 49
        self.session.get_kline.return_value = {"result": {"list": candles}}
        with self.assertLogs("bot", level="INFO") as cm:
            self.bot.log_market_trend("BTCUSDT")
        self.session.get_kline.assert_called_once_with(
            category="linear", symbol="BTCUSDT", interval=5, limit=50
        )
        self.assertIn("BTCUSDT: actively bought, price increases", cm.output[0])

    def test_log_market_trend_decrease(self):
        candles = [[0, 0, 0, 0, "90", 0]] + [[0, 0, 0, 0, "100", 0]] * 49
        self.session.get_kline.return_value = {"result": {"list": candles}}
        with self.assertLogs("bot", level="INFO") as cm:
            self.bot.log_market_trend("BTCUSDT")
        self.session.get_kline.assert_called_once_with(
            category="linear", symbol="BTCUSDT", interval=5, limit=50
        )
        self.assertIn("BTCUSDT: actively sold, price decreases", cm.output[0])

    def test_log_market_trend_no_data(self):
        self.session.get_kline.return_value = {"result": {"list": []}}
        with self.assertLogs("bot", level="WARNING") as cm:
            self.bot.log_market_trend("BTCUSDT")
        self.assertIn("No kline data", cm.output[0])

    def test_log_market_trend_stable(self):
        candles = [[0, 0, 0, 0, "100", 0]] * 50
        self.session.get_kline.return_value = {"result": {"list": candles}}
        with self.assertLogs("bot", level="INFO") as cm:
            self.bot.log_market_trend("BTCUSDT")
        self.assertIn("stable, price unchanged", cm.output[0])

    def test_ma_crossover_signal_buy(self):
        candles = [[0, 0, 0, 0, "110", 0]] * 5 + [[0, 0, 0, 0, "100", 0]] * 45
        self.session.get_kline.return_value = {"result": {"list": candles}}
        signal = self.bot.ma_crossover_signal("BTCUSDT")
        self.assertEqual(signal, "Buy")

    def test_ma_crossover_signal_sell(self):
        closes = [100] * 45 + [90] * 5
        candles = [[0, 0, 0, 0, str(p), 0] for p in reversed(closes)]
        self.session.get_kline.return_value = {"result": {"list": candles}}
        signal = self.bot.ma_crossover_signal("BTCUSDT")
        self.assertEqual(signal, "Sell")

    def test_ma_crossover_signal_hold(self):
        candles = [[0, 0, 0, 0, "100", 0]] * 50
        self.session.get_kline.return_value = {"result": {"list": candles}}
        signal = self.bot.ma_crossover_signal("BTCUSDT")
        self.assertEqual(signal, "Hold")

    def test_ma_crossover_signal_not_enough_data(self):
        self.session.get_kline.return_value = {"result": {"list": [[0, 0, 0, 0, "100", 0]] * 10}}
        with self.assertLogs("bot", level="WARNING") as cm:
            signal = self.bot.ma_crossover_signal("BTCUSDT")
        self.assertIn("Not enough kline data", cm.output[0])
        self.assertEqual(signal, "Hold")

    def test_rsi_signal_buy(self):
        candles = [[0, 0, 0, 0, str(p), 0] for p in range(86, 101)]
        self.session.get_kline.return_value = {"result": {"list": candles}}
        signal = self.bot.rsi_signal("BTCUSDT")
        self.assertEqual(signal, "Buy")

    def test_rsi_signal_sell(self):
        candles = [[0, 0, 0, 0, str(p), 0] for p in range(100, 85, -1)]
        self.session.get_kline.return_value = {"result": {"list": candles}}
        signal = self.bot.rsi_signal("BTCUSDT")
        self.assertEqual(signal, "Sell")

    def test_rsi_signal_hold_value(self):
        closes = [100, 101] * 7 + [100]
        candles = [[0, 0, 0, 0, str(p), 0] for p in reversed(closes)]
        self.session.get_kline.return_value = {"result": {"list": candles}}
        signal = self.bot.rsi_signal("BTCUSDT")
        self.assertEqual(signal, "Hold")

    def test_rsi_signal_not_enough_data(self):
        self.session.get_kline.return_value = {"result": {"list": [[0, 0, 0, 0, "100", 0]] * 10}}
        with self.assertLogs("bot", level="WARNING") as cm:
            signal = self.bot.rsi_signal("BTCUSDT")
        self.assertIn("Not enough kline data", cm.output[0])
        self.assertEqual(signal, "Hold")

    def test_combined_signal_hold_on_disagreement(self):
        self.bot.ma_crossover_signal = MagicMock(return_value="Buy")
        self.bot.rsi_signal = MagicMock(return_value="Sell")
        with self.assertLogs("bot", level="INFO") as cm:
            signal, ma, rsi = self.bot.combined_signal("BTCUSDT")
        self.assertEqual(signal, "Hold")
        self.assertEqual(ma, "Buy")
        self.assertEqual(rsi, "Sell")
        self.assertIn("MA=Buy, RSI=Sell -> Hold", cm.output[0])

    def test_combined_signal_agree_sell(self):
        self.bot.ma_crossover_signal = MagicMock(return_value="Sell")
        self.bot.rsi_signal = MagicMock(return_value="Sell")
        with self.assertLogs("bot", level="INFO") as cm:
            signal, ma, rsi = self.bot.combined_signal("BTCUSDT")
        self.assertEqual(signal, "Sell")
        self.assertEqual(ma, "Sell")
        self.assertEqual(rsi, "Sell")
        self.assertIn("MA=Sell, RSI=Sell -> Sell", cm.output[0])

    def test_trade_with_signals_calls_place_order(self):
        self.bot.combined_signal = MagicMock(return_value=("Buy", "Buy", "Buy"))
        self.bot.place_order = MagicMock()
        self.bot._last_price = MagicMock(return_value=100)
        self.bot._calculate_sl_tp = MagicMock(return_value=(90, 110))
        self.bot.trade_with_signals("BTCUSDT", 100, 10)
        self.bot.place_order.assert_called_once_with(
            "BTCUSDT", "Buy", 100, 10, 90, 110, 100
        )

    def test_trade_with_signals_no_signal(self):
        self.bot.combined_signal = MagicMock(return_value=("Hold", "Buy", "Sell"))
        self.bot.place_order = MagicMock()
        with self.assertLogs("bot", level="INFO") as cm:
            result = self.bot.trade_with_signals("BTCUSDT", 100, 10)
        self.assertIsNone(result)
        self.bot.place_order.assert_not_called()
        self.assertIn("no trade signal", cm.output[0])

    def test_trade_strategy_passes_price(self):
        candles = [[0, 100, 101, 99, 100, 0]] * 50
        self.session.get_kline.return_value = {"result": {"list": candles}}

        def dummy_strategy(cs):
            assert len(cs[0]) == 6
            price = cs[-1][4]
            return "Buy", price - 1, price + 1

        self.bot.place_order = MagicMock()
        self.bot.trade_strategy("BTCUSDT", 100, 10, dummy_strategy)
        self.bot.place_order.assert_called_once_with(
            "BTCUSDT", "Buy", 100, 10, 99, 101, 100
        )

    def test_recent_trades_calls_api(self):
        self.session.get_executions.return_value = {"result": {"list": [{"a": 1}]}}
        trades = self.bot.recent_trades("BTCUSDT")
        self.session.get_executions.assert_called_once_with(
            category="linear", symbol="BTCUSDT", limit=10
        )
        self.assertEqual(trades, [{"a": 1}])

    def test_place_order_ignores_leverage_error(self):
        self.session.set_leverage.side_effect = Exception(
            "leverage not modified (ErrCode: 110043)"
        )
        self.bot.place_order("BTCUSDT", "Buy", 100, 10, 95, 105)
        self.session.place_order.assert_called_once()

    def test_qty_is_rounded_to_step(self):
        self.session.get_tickers.return_value = {
            "result": {"list": [{"lastPrice": "114000"}]}
        }
        self.bot.place_order("BTCUSDT", "Buy", 100, 10, 95000, 120000)
        _, kwargs = self.session.place_order.call_args
        self.assertEqual(kwargs["qty"], "0.008")

    def test_format_qty_uses_minimum(self):
        qty = self.bot._format_qty("BTCUSDT", 0.0004)
        self.assertEqual(qty, 0.001)

    def test_place_order_requires_sl_tp(self):
        with self.assertRaises(ValueError):
            self.bot.place_order("BTCUSDT", "Buy", 100, 10, None, 105)
        with self.assertRaises(ValueError):
            self.bot.place_order("BTCUSDT", "Buy", 100, 10, 95, None)

    def test_place_order_invalid_sl_tp_buy(self):
        with self.assertRaises(ValueError):
            self.bot.place_order("BTCUSDT", "Buy", 100, 10, 105, 95)

    def test_place_order_invalid_sl_tp_sell(self):
        with self.assertRaises(ValueError):
            self.bot.place_order("BTCUSDT", "Sell", 100, 10, 95, 105)

    def test_sl_tp_rounded_to_tick(self):
        self.session.place_order.reset_mock()
        self.bot.place_order("BTCUSDT", "Buy", 100, 10, 94.23, 105.87)
        _, kwargs = self.session.place_order.call_args
        self.assertEqual(kwargs["stopLoss"], "94.0")
        self.assertEqual(kwargs["takeProfit"], "106.0")

    def test_last_price_no_data(self):
        self.session.get_tickers.return_value = {"result": {"list": [{}]}}
        with self.assertRaises(ValueError):
            self.bot._last_price("BTCUSDT")

    def test_calculate_sl_tp_buy(self):
        closes = list(range(100, 115))
        candles = [[0, p, p + 1, p - 1, p, 0] for p in reversed(closes)]
        self.session.get_kline.return_value = {"result": {"list": candles}}
        self.session.get_tickers.return_value = {
            "result": {"list": [{"lastPrice": "114"}]}
        }
        price = 114
        stop, take = self.bot._calculate_sl_tp("BTCUSDT", "Buy", price)
        self.assertAlmostEqual(stop, 111)
        self.assertAlmostEqual(take, 119)

    def test_calculate_sl_tp_sell(self):
        closes = list(range(100, 115))
        candles = [[0, p, p + 1, p - 1, p, 0] for p in reversed(closes)]
        self.session.get_kline.return_value = {"result": {"list": candles}}
        self.session.get_tickers.return_value = {
            "result": {"list": [{"lastPrice": "114"}]}
        }
        price = 114
        stop, take = self.bot._calculate_sl_tp("BTCUSDT", "Sell", price)
        self.assertAlmostEqual(stop, 117)
        self.assertAlmostEqual(take, 109)

    def test_calculate_sl_tp_no_data(self):
        self.session.get_kline.return_value = {"result": {"list": []}}
        with self.assertRaises(ValueError):
            self.bot._calculate_sl_tp("BTCUSDT", "Buy", 100)

    def test_calculate_sl_tp_bounds(self):
        candles_small = [[0, 100, 100.005, 99.995, 100, 0]] * 15
        self.session.get_kline.return_value = {"result": {"list": candles_small}}
        self.session.get_tickers.return_value = {
            "result": {"list": [{"lastPrice": "100"}]}
        }
        price = 100
        stop, take = self.bot._calculate_sl_tp("BTCUSDT", "Buy", price)
        self.assertAlmostEqual(stop, 98.5)
        self.assertAlmostEqual(take, 101.5)
        candles_large = [[0, 100, 110, 90, 100, 0]] * 15
        self.session.get_kline.return_value = {"result": {"list": candles_large}}
        stop, take = self.bot._calculate_sl_tp("BTCUSDT", "Buy", price)
        self.assertAlmostEqual(stop, 95)
        self.assertAlmostEqual(take, 105)

    def test_log_all_trends_invokes_log_market_trend(self):
        self.bot.log_market_trend = MagicMock()
        self.bot.log_all_trends()
        self.assertEqual(
            self.bot.log_market_trend.call_count, len(self.bot.ALLOWED_SYMBOLS)
        )

    def test_setup_logging_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.txt"
            setup_logging(log_path)
            logger.info("file logging works")
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                handler.flush()
                handler.close()
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("file logging works", content)
            root_logger.handlers.clear()


if __name__ == "__main__":
    unittest.main()
