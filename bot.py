from config import BybitConfig
from pybit.unified_trading import HTTP
import urllib3
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List
from functools import lru_cache
from decimal import Decimal, ROUND_HALF_UP
import time
import json
from datetime import datetime, timedelta

from functions import (
    sma_crossover,
    breakout,
    mean_reversion,
    rsi_strategy,
    half_year_strategy,
    apply_sl_tp_bounds,
    calculate_dynamic_order_size,
    dynamic_leverage,
    limit_open_positions,
)


logger = logging.getLogger(__name__)


def setup_logging(log_file: str | Path = "trade_log.txt") -> None:
    """Configure logging to console and append to a text file."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )


class PositionManager:
    """Управление открытыми позициями и рисками
    
    Атрибуты:
        positions: Словарь открытых позиций
        max_positions: Максимальное количество одновременных позиций
        max_risk_per_trade: Максимальный риск на одну сделку в процентах
        last_trade_time: Время последней сделки по каждому символу
        min_trade_interval: Минимальный интервал между сделками для одного символа
    """
    
    def __init__(self, max_positions: int = 3, max_risk_per_trade: float = 2.0, min_trade_interval_minutes: int = 5):
        """Инициализация менеджера позиций
        
        Args:
            max_positions: Максимальное количество одновременных позиций
            max_risk_per_trade: Максимальный риск на одну сделку в процентах
            min_trade_interval_minutes: Минимальный интервал между сделками в минутах
        """
        self.positions: Dict[str, Dict] = {}
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        self.last_trade_time: Dict[str, datetime] = {}
        self.min_trade_interval = timedelta(minutes=min_trade_interval_minutes)
        
    def can_open_position(self, symbol: str) -> bool:
        """Проверка возможности открытия новой позиции
        
        Args:
            symbol: Торговая пара
            
        Returns:
            bool: True если можно открыть позицию, False если нельзя
        """
        if len(self.positions) >= self.max_positions:
            logger.info(f"Cannot open {symbol}: max positions reached ({self.max_positions})")
            return False
            
        if symbol in self.positions:
            logger.info(f"Cannot open {symbol}: position already exists")
            return False
            
        if symbol in self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time[symbol]
            if time_since_last < self.min_trade_interval:
                logger.debug(f"Cannot open {symbol}: too soon since last trade ({time_since_last.seconds}s)")
                return False
                
        return True
    
    def add_position(self, symbol: str, side: str, amount: float, price: float, stop_loss: float, take_profit: float):
        """Добавление новой позиции
        
        Args:
            symbol: Торговая пара
            side: Сторона сделки (Buy/Sell)
            amount: Размер позиции
            price: Цена входа
            stop_loss: Уровень стоп-лосс
            take_profit: Уровень тейк-профит
        """
        self.positions[symbol] = {
            'side': side,
            'amount': amount,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now()
        }
        self.last_trade_time[symbol] = datetime.now()
        logger.info(f"Position added: {symbol} {side} amount={amount} price={price} SL={stop_loss} TP={take_profit}")
    
    def remove_position(self, symbol: str):
        """Удаление закрытой позиции
        
        Args:
            symbol: Торговая пара
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Position removed: {symbol}")
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Получение информации о позиции
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Optional[Dict]: Информация о позиции или None если позиция не найдена
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Получение всех открытых позиций
        
        Returns:
            Dict[str, Dict]: Словарь всех открытых позиций
        """
        return self.positions.copy()
    
    def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """Обновление позиции и проверка достижения SL/TP
        
        Args:
            symbol: Торговая пара
            current_price: Текущая цена
            
        Returns:
            Optional[str]: "stop_loss" если достигнут стоп-лосс, 
                          "take_profit" если достигнут тейк-профит,
                          None если уровни не достигнуты
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        side = position['side']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Проверяем достижение уровней SL/TP
        if side == "Buy":
            if current_price <= stop_loss:
                return "stop_loss"
            elif current_price >= take_profit:
                return "take_profit"
        else:  # Sell
            if current_price >= stop_loss:
                return "stop_loss"
            elif current_price <= take_profit:
                return "take_profit"
        
        return None


class StrategySelector:
    """Интеллектуальный выбор стратегии на основе рыночных условий"""
    
    def __init__(self, strategy_cooldown_minutes: int = 30):
        """Инициализация селектора стратегий
        
        Args:
            strategy_cooldown_minutes: Период кулдауна для стратегий в минутах
        """
        self.strategy_performance: Dict[str, Dict] = {}
        self.strategy_cooldown: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=strategy_cooldown_minutes)
        
    def select_best_strategy(self, symbol: str, market_conditions: Dict) -> Optional[tuple]:
        """Выбор лучшей стратегии для текущих рыночных условий
        
        Args:
            symbol: Торговая пара
            market_conditions: Рыночные условия (волатильность, сила тренда, объемы)
            
        Returns:
            Optional[tuple]: Кортеж (название стратегии, функция стратегии, вес) или None
        """
        
        volatility = market_conditions.get('volatility', 0)
        trend_strength = market_conditions.get('trend_strength', 0)
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        
        strategies = []
        
        # Выбираем стратегии на основе рыночных условий
        if trend_strength > 0.6:
            strategies.append(('sma_crossover', sma_crossover, 1.0))
            strategies.append(('breakout', breakout, 0.8))
        elif volatility > 2.0:
            strategies.append(('mean_reversion', mean_reversion, 1.0))
            strategies.append(('rsi_strategy', rsi_strategy, 0.7))
        else:
            strategies.append(('sma_crossover', sma_crossover, 0.5))
            strategies.append(('breakout', breakout, 0.5))
            strategies.append(('mean_reversion', mean_reversion, 0.5))
            strategies.append(('rsi_strategy', rsi_strategy, 0.5))
        
        if volume_ratio > 0.8:
            strategies.append(('half_year_strategy', half_year_strategy, 0.3))
        
        # Фильтруем стратегии в кулдауне
        available_strategies = []
        current_time = datetime.now()
        for name, func, weight in strategies:
            if name in self.strategy_cooldown:
                if current_time - self.strategy_cooldown[name] < self.cooldown_period:
                    continue
            available_strategies.append((name, func, weight))
        
        if not available_strategies:
            return None
            
        best_strategy = max(available_strategies, key=lambda x: x[2])
        logger.debug(f"Selected strategy for {symbol}: {best_strategy[0]} (weight={best_strategy[2]:.2f})")
        return best_strategy  # Возвращаем кортеж (name, func, weight)
    
    def record_result(self, strategy_name: str, success: bool):
        """Запись результата работы стратегии
        
        Args:
            strategy_name: Название стратегии
            success: Результат работы стратегии (True - успех, False - неудача)
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {'success': 0, 'fail': 0}
        
        if success:
            self.strategy_performance[strategy_name]['success'] += 1
        else:
            self.strategy_performance[strategy_name]['fail'] += 1
            self.strategy_cooldown[strategy_name] = datetime.now()


class BybitTradingBot:
    """Enhanced Bybit trading bot with risk management
    
    Атрибуты:
        session: Сессия API Bybit
        config: Конфигурация бота
        position_manager: Менеджер позиций
        strategy_selector: Селектор стратегий
        account_balance: Баланс аккаунта
        last_balance_update: Время последнего обновления баланса
    """

    ALLOWED_SYMBOLS = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "BNBUSDT",
    ]

    def __init__(self, session: HTTP, config: BybitConfig):
        """Инициализация торгового бота
        
        Args:
            session: Сессия API Bybit
            config: Конфигурация бота
        """
        self.session = session
        self.config = config
        self.position_manager = PositionManager(
            max_positions=config.max_positions,
            max_risk_per_trade=config.max_risk_per_trade,
            min_trade_interval_minutes=config.min_trade_interval_minutes
        )
        self.strategy_selector = StrategySelector(
            strategy_cooldown_minutes=self.config.strategy_cooldown_minutes
        )
        self.account_balance = 0
        self.last_balance_update = datetime.min

    def update_balance(self) -> float:
        """Обновление баланса аккаунта
        
        Returns:
            float: Текущий баланс аккаунта
        """
        try:
            if datetime.now() - self.last_balance_update > timedelta(minutes=self.config.balance_update_interval_minutes):
                result = self.session.get_wallet_balance(accountType="UNIFIED")
                
                # Проверяем результат API
                ret_code = result.get("retCode", -1)
                if ret_code != 0:
                    logger.error(f"Failed to get wallet balance - code: {ret_code}, message: {result.get('retMsg', 'Unknown error')}")
                    return self.account_balance  # Возвращаем кэшированное значение
                
                balance_data = result.get("result", {}).get("list", [{}])[0]
                coins = balance_data.get("coin", [])
                for coin in coins:
                    if coin.get("coin") == "USDT":
                        self.account_balance = float(coin.get("walletBalance", 0))
                        self.last_balance_update = datetime.now()
                        logger.info(f"Account balance updated: ${self.account_balance:.2f}")
                        break
        except Exception as e:
            logger.error(f"Failed to update balance: {e}")
        return self.account_balance

    def _validate(self, symbol: str, amount: float, leverage: int) -> None:
        """Валидация параметров сделки
        
        Args:
            symbol: Торговая пара
            amount: Размер сделки
            leverage: Плечо
            
        Raises:
            ValueError: Если параметры не проходят валидацию
        """
        if symbol not in self.ALLOWED_SYMBOLS:
            raise ValueError("Unsupported trading pair")
        if not self.config.min_trade_amount <= amount <= self.config.max_trade_amount:
            raise ValueError(f"Amount must be between {self.config.min_trade_amount} and {self.config.max_trade_amount} USD")
        if not self.config.min_leverage <= leverage <= self.config.max_leverage:
            raise ValueError(f"Leverage must be between {self.config.min_leverage} and {self.config.max_leverage}")

    def _last_price(self, symbol: str) -> float:
        """Return last traded price for symbol"""
        try:
            result = self.session.get_tickers(category="linear", symbol=symbol)
            
            # Проверяем результат API
            ret_code = result.get("retCode", -1)
            if ret_code != 0:
                raise ValueError(f"API error for {symbol}: code {ret_code}, message: {result.get('retMsg', 'Unknown error')}")
            
            price = result.get("result", {}).get("list", [{}])[0].get("lastPrice")
            if price is None:
                raise ValueError(f"No price data for {symbol}")
            return float(price)
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            raise

    @lru_cache(maxsize=128)
    def _instrument_info(self, symbol: str) -> dict:
        """Return instrument info used for lot and price steps"""
        try:
            result = self.session.get_instruments_info(category="linear", symbol=symbol)
            
            # Проверяем результат API
            ret_code = result.get("retCode", -1)
            if ret_code != 0:
                raise ValueError(f"API error getting instrument info for {symbol}: code {ret_code}")
            
            instrument_list = result.get("result", {}).get("list", [])
            if not instrument_list:
                raise ValueError(f"No instrument info found for {symbol}")
            
            return instrument_list[0]
        except Exception as e:
            logger.error(f"Failed to get instrument info for {symbol}: {e}")
            # Возвращаем базовые значения в качестве fallback
            return {
                "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001"},
                "priceFilter": {"tickSize": "0.01"}
            }

    def _lot_step(self, symbol: str) -> tuple[float, float]:
        """Return quantity step and minimum order size for symbol"""
        info = self._instrument_info(symbol).get("lotSizeFilter", {})
        step = float(info.get("qtyStep", 1))
        min_qty = float(info.get("minOrderQty", step))
        return step, min_qty

    def _price_tick(self, symbol: str) -> float:
        """Return price tick size for symbol"""
        info = self._instrument_info(symbol).get("priceFilter", {})
        return float(info.get("tickSize", 0.01))

    def _format_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to exchange step size"""
        step, min_qty = self._lot_step(symbol)
        d_step = Decimal(str(step))
        d_qty = Decimal(str(qty))
        d_min = Decimal(str(min_qty))
        adjusted = (d_qty // d_step) * d_step
        if adjusted < d_min:
            adjusted = d_min
        return float(adjusted)

    def _format_price(self, symbol: str, price: float) -> float:
        """Round price to the instrument's tick size"""
        tick = Decimal(str(self._price_tick(symbol)))
        d_price = Decimal(str(price))
        return float((d_price / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick)
    
    def _validate_sl_tp_levels(self, symbol: str, price: float, stop_loss: float, take_profit: float, side: str) -> bool:
        """Проверка доступности уровней SL/TP через API"""
        try:
            # Получаем информацию о минимальных и максимальных расстояниях для SL/TP
            info = self._instrument_info(symbol)
            price_filter = info.get("priceFilter", {})
            min_price = float(price_filter.get("minPrice", 0))
            max_price = float(price_filter.get("maxPrice", 1000000))
            
            # Проверяем, что уровни находятся в допустимом диапазоне цен
            if not (min_price <= stop_loss <= max_price):
                logger.warning(f"{symbol}: Stop-loss level {stop_loss} is out of range [{min_price}, {max_price}]")
                return False
                
            if not (min_price <= take_profit <= max_price):
                logger.warning(f"{symbol}: Take-profit level {take_profit} is out of range [{min_price}, {max_price}]")
                return False
            
            # Проверяем правильность расположения уровней относительно цены входа
            if side == "Buy":
                if not (stop_loss < price < take_profit):
                    logger.warning(f"{symbol}: Invalid Buy SL/TP levels: SL={stop_loss}, Price={price}, TP={take_profit}")
                    return False
            else:  # Sell
                if not (take_profit < price < stop_loss):
                    logger.warning(f"{symbol}: Invalid Sell SL/TP levels: TP={take_profit}, Price={price}, SL={stop_loss}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"{symbol}: Error validating SL/TP levels: {e}")
            return False

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        """Safely set leverage ignoring non-modification errors"""
        try:
            logger.debug(f"Setting leverage for {symbol} to {leverage}x")
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            logger.debug(f"Leverage set successfully for {symbol}")
        except Exception as exc:
            msg = str(exc)
            logger.error(f"Error setting leverage for {symbol}: {msg}")
            if "110043" in msg or "leverage not modified" in msg.lower():
                logger.debug(f"{symbol}: leverage already set to {leverage}")
            else:
                # Логируем traceback для отладки
                import traceback
                logger.error(f"{symbol}: Traceback: {traceback.format_exc()}")
                raise

    def calculate_market_conditions(self, symbol: str) -> Dict:
        """Анализ рыночных условий для выбора стратегии
        
        Args:
            symbol: Торговая пара для анализа
            
        Returns:
            Dict: Словарь с параметрами рынка:
                - volatility: Волатильность в процентах
                - trend_strength: Сила тренда (0-1)
                - volume_ratio: Соотношение текущего объема к среднему
        """
        try:
            # Получаем свечи для анализа
            result = self.session.get_kline(
                category="linear", symbol=symbol, interval=5, limit=50
            )
            candles = result.get("result", {}).get("list", [])
            if len(candles) < 20:
                return {'volatility': 1.0, 'trend_strength': 0.5, 'volume_ratio': 1.0}
            
            closes = [float(c[4]) for c in reversed(candles)]
            volumes = [float(c[5]) for c in reversed(candles)]
            
            # Расчет волатильности (процентное изменение)
            returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 
                      for i in range(1, len(closes))]
            volatility = sum(abs(r) for r in returns) / len(returns) if returns else 1.0
            
            # Расчет силы тренда
            sma20 = sum(closes[-20:]) / 20
            price_to_sma = (closes[-1] - sma20) / sma20
            trend_strength = abs(price_to_sma)
            
            # Расчет объема
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'volatility': volatility,
                'trend_strength': min(trend_strength, 1.0),
                'volume_ratio': volume_ratio
            }
        except Exception as e:
            logger.error(f"Failed to calculate market conditions: {e}")
            return {'volatility': 1.0, 'trend_strength': 0.5, 'volume_ratio': 1.0}

    def calculate_position_size(self, symbol: str, balance: float, volatility: float) -> float:
        """Расчет размера позиции с учетом риска"""
        # Используем функцию из functions.py
        base_risk = self.position_manager.max_risk_per_trade
        position_size = calculate_dynamic_order_size(balance, volatility, base_risk)
        
        # Ограничиваем размер позиции
        min_size = 10  # Минимум $10
        max_size = min(balance * 0.1, 500)  # Максимум 10% от баланса или $500
        
        position_size = max(min_size, min(position_size, max_size))
        logger.debug(f"{symbol}: calculated position size=${position_size:.2f} (volatility={volatility:.2f})")
        return position_size

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
        price: float | None = None,
    ) -> dict:
        """Размещение рыночного ордера с SL/TP
        
        Args:
            symbol: Торговая пара
            side: Сторона сделки (Buy/Sell)
            amount: Размер позиции в USDT
            leverage: Плечо
            stop_loss: Уровень стоп-лосс
            take_profit: Уровень тейк-профит
            price: Цена входа (если не указана, используется текущая рыночная цена)
            
        Returns:
            dict: Результат размещения ордера
            
        Raises:
            ValueError: Если параметры не проходят валидацию
        """
        if stop_loss is None or take_profit is None:
            raise ValueError("Stop loss and take profit required")
        
        self._validate(symbol, amount, leverage)
        price = self._last_price(symbol) if price is None else price
        stop_loss = self._format_price(symbol, stop_loss)
        take_profit = self._format_price(symbol, take_profit)
        
        # Проверяем доступность уровней SL/TP
        if not self._validate_sl_tp_levels(symbol, price, stop_loss, take_profit, side):
            raise ValueError(f"Invalid SL/TP levels for {symbol}")
        
        qty = self._format_qty(symbol, amount * leverage / price)
        logger.info(
            f"{symbol}: placing {side} order: qty={qty:.4f}, price=${price:.2f}, "
            f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}, leverage={leverage}x"
        )
        
        self._set_leverage(symbol, leverage)
        
        try:
            result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                timeInForce="ImmediateOrCancel",
                stopLoss=str(stop_loss),
                takeProfit=str(take_profit),
                tpslMode="Partial",
            )
            
            # Проверяем результат и логируем подробную информацию
            ret_code = result.get("retCode", -1)
            ret_msg = result.get("retMsg", "Unknown error")
            
            if ret_code == 0:
                logger.info(f"{symbol}: Order placed successfully")
                # Добавляем позицию в менеджер только при успешном размещении ордера
                self.position_manager.add_position(symbol, side, amount, price, stop_loss, take_profit)
            else:
                logger.error(f"{symbol}: Failed to place order - code: {ret_code}, message: {ret_msg}")
                # Логируем дополнительную информацию об ошибке
                if "error" in result:
                    logger.error(f"{symbol}: Error details: {result['error']}")
            
            return result
        except Exception as e:
            logger.error(f"{symbol}: Exception during order placement: {str(e)}")
            # Логируем traceback для отладки
            import traceback
            logger.error(f"{symbol}: Traceback: {traceback.format_exc()}")
            return {"retCode": -1, "retMsg": f"Exception: {str(e)}"}

    def smart_trade(self, symbol: str) -> Optional[dict]:
        """Интеллектуальная торговля с выбором стратегии и управлением рисками
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Optional[dict]: Результат размещения ордера или None
        """
        
        # Проверяем, можем ли открыть позицию
        if not self.position_manager.can_open_position(symbol):
            return None
        
        # Обновляем баланс
        balance = self.update_balance()
        min_balance = self.config.min_trade_amount * 5  # Минимальный баланс должен быть в 5 раз больше минимальной суммы сделки
        if balance < min_balance:
            logger.warning(f"Insufficient balance: ${balance:.2f} (minimum required: ${min_balance:.2f})")
            return None
        
        # Анализируем рыночные условия
        market_conditions = self.calculate_market_conditions(symbol)
        logger.info(f"{symbol} market: volatility={market_conditions['volatility']:.2f}, "
                   f"trend={market_conditions['trend_strength']:.2f}, "
                   f"volume_ratio={market_conditions['volume_ratio']:.2f}")
        
        # Выбираем стратегию
        strategy_info = self.strategy_selector.select_best_strategy(symbol, market_conditions)
        if not strategy_info:
            logger.info(f"{symbol}: no suitable strategy available")
            return None
        
        strategy_name, strategy_func, _ = strategy_info
        
        # Получаем данные для стратегии
        interval = 240 if strategy_name == 'half_year_strategy' else 5
        limit = 200 if strategy_name == 'half_year_strategy' else 50
        
        result = self.session.get_kline(
            category="linear", symbol=symbol, interval=interval, limit=limit
        )
        candles = result.get("result", {}).get("list", [])
        if len(candles) < limit:
            logger.warning(f"{symbol}: not enough data for {strategy_name}")
            return None
        
        candles = [list(map(float, c[:6])) for c in reversed(candles)]
        
        # Получаем сигнал от стратегии
        try:
            signal, stop, take = strategy_func(candles)
        except Exception as e:
            logger.error(f"{symbol}: strategy {strategy_name} failed: {e}")
            self.strategy_selector.record_result(strategy_name, False)
            return None
        
        if signal == "Hold":
            logger.info(f"{symbol}: {strategy_name} -> no signal")
            return None
        
        logger.info(f"{symbol}: {strategy_name} -> {signal}, SL=${stop:.2f}, TP=${take:.2f}")
        
        # Рассчитываем размер позиции
        position_size = self.calculate_position_size(
            symbol, balance, market_conditions['volatility']
        )
        
        # Рассчитываем динамическое плечо
        base_leverage = (self.config.min_leverage + self.config.max_leverage) // 2
        leverage = dynamic_leverage(
            base_leverage, 
            market_conditions['volatility'],
            max_leverage=self.config.max_leverage,
            min_leverage=self.config.min_leverage
        )
        
        # Открываем позицию
        try:
            price = candles[-1][4]
            order_result = self.place_order(
                symbol, signal, position_size, leverage, stop, take, price
            )
            
            if order_result.get("retCode") == 0:
                logger.info(f"{symbol}: order placed successfully via {strategy_name}")
                self.strategy_selector.record_result(strategy_name, True)
                return order_result
            else:
                logger.error(f"{symbol}: order failed: {order_result}")
                self.strategy_selector.record_result(strategy_name, False)
                return None
                
        except Exception as e:
            logger.error(f"{symbol}: failed to place order: {e}")
            self.strategy_selector.record_result(strategy_name, False)
            return None

    def log_market_trend(self, symbol: str) -> None:
        """Log market trend information"""
        result = self.session.get_kline(
            category="linear", symbol=symbol, interval=5, limit=50
        )
        candles = result.get("result", {}).get("list", [])
        if not candles:
            logger.warning(f"No kline data for {symbol}")
            return
        closes = [float(c[4]) for c in reversed(candles)]
        start, end = closes[0], closes[-1]
        change_pct = ((end - start) / start) * 100
        if end > start:
            trend = f"UP {change_pct:.2f}%"
        elif end < start:
            trend = f"DOWN {change_pct:.2f}%"
        else:
            trend = "FLAT"
        logger.info(f"{symbol}: {trend} (${start:.2f} -> ${end:.2f})")

    def log_all_trends(self) -> None:
        for symbol in self.ALLOWED_SYMBOLS:
            self.log_market_trend(symbol)

    def get_strategy_stats(self) -> str:
        """Получение статистики по стратегиям"""
        stats = []
        for name, perf in self.strategy_selector.strategy_performance.items():
            total = perf['success'] + perf['fail']
            if total > 0:
                win_rate = (perf['success'] / total) * 100
                stats.append(f"{name}: {perf['success']}/{total} ({win_rate:.1f}%)")
        return " | ".join(stats) if stats else "No trades yet"
    
    def close_position(self, symbol: str, side: str, amount: float) -> dict:
        """Закрытие позиции
        
        Args:
            symbol: Торговая пара
            side: Сторона позиции (Buy/Sell)
            amount: Размер позиции
            
        Returns:
            dict: Результат закрытия позиции
        """
        try:
            close_side = "Sell" if side == "Buy" else "Buy"
            qty = self._format_qty(symbol, amount)
            
            logger.info(f"Closing position: {symbol} {close_side} qty={qty}")
            
            result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=str(qty),
                timeInForce="ImmediateOrCancel",
                reduceOnly=True,
            )
            
            # Проверяем результат и логируем подробную информацию
            ret_code = result.get("retCode", -1)
            ret_msg = result.get("retMsg", "Unknown error")
            
            if ret_code == 0:
                self.position_manager.remove_position(symbol)
                logger.info(f"Position closed successfully: {symbol} {close_side} qty={qty}")
            else:
                logger.error(f"Failed to close position {symbol} - code: {ret_code}, message: {ret_msg}")
                # Логируем дополнительную информацию об ошибке
                if "error" in result:
                    logger.error(f"{symbol}: Error details: {result['error']}")
                # Удаляем позицию из менеджера даже при ошибке API (позиция могла быть закрыта)
                self.position_manager.remove_position(symbol)
            
            return result
        except Exception as e:
            logger.error(f"Exception closing position {symbol}: {e}")
            # Логируем traceback для отладки
            import traceback
            logger.error(f"{symbol}: Traceback: {traceback.format_exc()}")
            return {"retCode": -1, "retMsg": f"Exception: {str(e)}"}
    
    def check_and_close_positions(self) -> None:
        """Проверка и закрытие позиций при достижении SL/TP
        
        Проверяет все открытые позиции и закрывает те, где достигнуты уровни SL/TP
        """
        positions = self.position_manager.get_all_positions()
        for symbol in list(positions.keys()):  # Используем list() для избежания изменения словаря во время итерации
            try:
                current_price = self._last_price(symbol)
                result = self.position_manager.update_position(symbol, current_price)
                
                if result == "stop_loss":
                    position = self.position_manager.get_position(symbol)
                    if position:
                        logger.info(f"Stop-loss reached for {symbol} at price {current_price}")
                        self.close_position(symbol, position['side'], position['amount'])
                elif result == "take_profit":
                    position = self.position_manager.get_position(symbol)
                    if position:
                        logger.info(f"Take-profit reached for {symbol} at price {current_price}")
                        self.close_position(symbol, position['side'], position['amount'])
            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")


def main() -> None:
    """Main trading loop with enhanced features"""
    cfg = BybitConfig.from_env()
    session = HTTP(
        testnet=cfg.testnet,
        api_key=cfg.api_key,
        api_secret=cfg.api_secret,
        demo=cfg.demo,
    )
    if cfg.ignore_ssl:
        session.client.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    setup_logging()
    bot = BybitTradingBot(session, cfg)
    
    logger.info("=" * 60)
    logger.info("Starting Enhanced Bybit Trading Bot")
    logger.info("=" * 60)
    
    # Начальная проверка баланса
    try:
        balance = bot.update_balance()
        logger.info(f"Initial balance: ${balance:.2f}")
    except Exception as e:
        logger.error(f"Failed to fetch initial balance: {e}")
        return
    
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n--- Iteration {iteration} ---")
        
        # Логируем тренды
        bot.log_all_trends()
        
        # Логируем статистику стратегий
        stats = bot.get_strategy_stats()
        if stats:
            logger.info(f"Strategy performance: {stats}")
        
        # Логируем открытые позиции
        if bot.position_manager.positions:
            logger.info(f"Open positions: {list(bot.position_manager.positions.keys())}")
        
        # Проверяем и закрываем позиции при достижении SL/TP
        bot.check_and_close_positions()
        
        # Торгуем по каждому символу
        for symbol in bot.ALLOWED_SYMBOLS:
            try:
                result = bot.smart_trade(symbol)
                if result:
                    logger.info(f"{symbol}: trade executed successfully")
            except Exception as e:
                logger.error(f"{symbol}: trade failed: {e}")
        
        # Пауза между итерациями
        logger.info(f"Waiting 60 seconds before next iteration...")
        time.sleep(60)


if __name__ == "__main__":
    main()