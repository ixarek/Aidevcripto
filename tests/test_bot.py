import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import logging
import tempfile

# Добавляем путь к родительской папке, чтобы импортировать bot.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot import BybitTradingBot, setup_logging, logger, PositionManager  # импорт бота и логгера
from config import BybitConfig


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
        # Создаем mock конфигурацию
        self.config = BybitConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            demo=True
        )
        self.bot = BybitTradingBot(self.session, self.config)

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
        # Сначала добавляем позицию
        self.bot.position_manager.add_position("BTCUSDT", "Buy", 100, 50000, 45000, 55000)
        
        # Мокаем успешный ответ API
        self.session.place_order.return_value = {"retCode": 0, "retMsg": "OK"}
        
        self.bot.close_position("BTCUSDT", "Buy", 100)
        self.session.place_order.assert_called_once()
        _, kwargs = self.session.place_order.call_args
        self.assertEqual(kwargs["category"], "linear")
        self.assertEqual(kwargs["symbol"], "BTCUSDT")
        self.assertEqual(kwargs["side"], "Sell")
        self.assertEqual(kwargs["orderType"], "Market")
        self.assertEqual(kwargs["timeInForce"], "ImmediateOrCancel")
        self.assertTrue(kwargs["reduceOnly"])
        self.assertAlmostEqual(float(kwargs["qty"]), 100.0)

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
        self.assertIn("BTCUSDT:", cm.output[0])
        self.assertIn("UP", cm.output[0])

    def test_log_market_trend_decrease(self):
        candles = [[0, 0, 0, 0, "90", 0]] + [[0, 0, 0, 0, "100", 0]] * 49
        self.session.get_kline.return_value = {"result": {"list": candles}}
        with self.assertLogs("bot", level="INFO") as cm:
            self.bot.log_market_trend("BTCUSDT")
        self.session.get_kline.assert_called_once_with(
            category="linear", symbol="BTCUSDT", interval=5, limit=50
        )
        self.assertIn("BTCUSDT:", cm.output[0])
        self.assertIn("DOWN", cm.output[0])

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
        self.assertIn("FLAT", cm.output[0])

    def test_smart_trade_functionality(self):
        """Test smart_trade method basic functionality"""
        # Mock market conditions
        self.bot.calculate_market_conditions = MagicMock(return_value={
            'volatility': 1.5, 'trend_strength': 0.7, 'volume_ratio': 1.2
        })
        
        # Mock balance update
        self.bot.update_balance = MagicMock(return_value=1000.0)
        
        # Mock strategy selector
        self.bot.strategy_selector.select_best_strategy = MagicMock(return_value=None)
        
        result = self.bot.smart_trade("BTCUSDT")
        self.assertIsNone(result)  # Should return None when no strategy available
    
    def test_position_manager_functionality(self):
        """Test position manager basic functionality"""
        pm = PositionManager(max_positions=2, max_risk_per_trade=1.0)
        
        # Test can_open_position
        self.assertTrue(pm.can_open_position("BTCUSDT"))
        
        # Add position
        pm.add_position("BTCUSDT", "Buy", 100, 50000, 45000, 55000)
        self.assertFalse(pm.can_open_position("BTCUSDT"))  # Already exists
        
        # Test get_position
        position = pm.get_position("BTCUSDT")
        self.assertIsNotNone(position)
        self.assertEqual(position['side'], "Buy")
        
        # Test remove_position
        pm.remove_position("BTCUSDT")
        self.assertIsNone(pm.get_position("BTCUSDT"))

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

    def test_calculate_market_conditions(self):
        """Test market conditions calculation"""
        candles = [[0, 100, 101, 99, 100, 1000]] * 50  # Добавляем volume
        self.session.get_kline.return_value = {"result": {"list": candles}}
        
        conditions = self.bot.calculate_market_conditions("BTCUSDT")
        
        self.assertIn('volatility', conditions)
        self.assertIn('trend_strength', conditions)
        self.assertIn('volume_ratio', conditions)
        
        # Проверяем, что значения в разумных пределах
        self.assertGreaterEqual(conditions['volatility'], 0)
        self.assertGreaterEqual(conditions['trend_strength'], 0)
        self.assertLessEqual(conditions['trend_strength'], 1.0)
        self.assertGreaterEqual(conditions['volume_ratio'], 0)

    def test_update_balance_api_error(self):
        """Test balance update with API error"""
        # Мокаем ошибку API
        self.session.get_wallet_balance.return_value = {
            "retCode": 10001,
            "retMsg": "API error"
        }
        
        # Устанавливаем кэшированный баланс
        self.bot.account_balance = 1000.0
        
        balance = self.bot.update_balance()
        self.assertEqual(balance, 1000.0)  # Должно вернуть кэшированное значение

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
