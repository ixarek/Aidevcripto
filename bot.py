from config import BybitConfig
from pybit.unified_trading import HTTP
import urllib3
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, List, Tuple
from functools import lru_cache
from decimal import Decimal, ROUND_HALF_UP
import time
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
    
    def __init__(self, max_positions_per_symbol: int = 3, max_risk_per_trade: float = 2.0, min_trade_interval_minutes: int = 5):
        """Инициализация менеджера позиций
        
        Args:
            max_positions_per_symbol: Максимальное количество одновременных позиций на символ
            max_risk_per_trade: Максимальный риск на одну сделку в процентах
            min_trade_interval_minutes: Минимальный интервал между сделками в минутах
        """
        self.positions: Dict[str, List[Dict]] = {}  # Теперь список позиций для каждого символа
        self.max_positions_per_symbol = max_positions_per_symbol
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
        # Проверяем количество позиций для конкретного символа
        symbol_positions = self.positions.get(symbol, [])
        if len(symbol_positions) >= self.max_positions_per_symbol:
            logger.info(f"Cannot open {symbol}: max positions for symbol reached ({len(symbol_positions)}/{self.max_positions_per_symbol})")
            return False
            
        if symbol in self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time[symbol]
            if time_since_last < self.min_trade_interval:
                logger.debug(f"Cannot open {symbol}: too soon since last trade ({time_since_last.seconds}s)")
                return False
                
        return True
    
    def add_position(self, symbol: str, side: str, amount: float, price: float, stop_loss: float, take_profit: float, strategy_used: str = None, position_id: str = None):
        """Добавление новой позиции с информацией о стратегии
        
        Args:
            symbol: Торговая пара
            side: Сторона сделки (Buy/Sell)
            amount: Размер позиции
            price: Цена входа
            stop_loss: Уровень стоп-лосс
            take_profit: Уровень тейк-профит
            strategy_used: Название использованной стратегии
            position_id: Уникальный ID позиции
        """
        if symbol not in self.positions:
            self.positions[symbol] = []
            
        # Генерируем ID позиции если не предоставлен
        if position_id is None:
            position_id = f"{symbol}_{len(self.positions[symbol])}_{int(datetime.now().timestamp())}"
            
        position = {
            'id': position_id,
            'side': side,
            'amount': amount,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now(),
            'strategy_used': strategy_used,
            'trade_start_time': datetime.now()
        }
        
        self.positions[symbol].append(position)
        self.last_trade_time[symbol] = datetime.now()
        logger.info(f"Position added: {symbol} {side} amount={amount} price={price} SL={stop_loss} TP={take_profit} strategy={strategy_used} (total: {len(self.positions[symbol])}/{self.max_positions_per_symbol})")
    
    def remove_position(self, symbol: str, position_id: str = None):
        """Удаление закрытой позиции
        
        Args:
            symbol: Торговая пара
            position_id: ID конкретной позиции (если None, удаляется первая)
        """
        if symbol not in self.positions or not self.positions[symbol]:
            logger.warning(f"No positions found for {symbol}")
            return
            
        if position_id is None:
            # Удаляем первую позицию если ID не указан
            removed_position = self.positions[symbol].pop(0)
            logger.info(f"Position removed: {symbol} (ID: {removed_position['id']})")
        else:
            # Удаляем конкретную позицию по ID
            for i, position in enumerate(self.positions[symbol]):
                if position['id'] == position_id:
                    removed_position = self.positions[symbol].pop(i)
                    logger.info(f"Position removed: {symbol} (ID: {position_id})")
                    break
            else:
                logger.warning(f"Position ID {position_id} not found for {symbol}")
                
        # Удаляем символ из словаря если позиций больше нет
        if not self.positions[symbol]:
            del self.positions[symbol]
    
    def get_position(self, symbol: str, position_id: str = None) -> Optional[Dict]:
        """Получение информации о позиции
        
        Args:
            symbol: Торговая пара
            position_id: ID конкретной позиции (если None, возвращается первая)
            
        Returns:
            Optional[Dict]: Информация о позиции или None если позиция не найдена
        """
        if symbol not in self.positions or not self.positions[symbol]:
            return None
            
        if position_id is None:
            return self.positions[symbol][0] if self.positions[symbol] else None
        else:
            for position in self.positions[symbol]:
                if position['id'] == position_id:
                    return position
            return None
    
    def get_all_positions(self) -> Dict[str, List[Dict]]:
        """Получение всех открытых позиций
        
        Returns:
            Dict[str, List[Dict]]: Словарь всех открытых позиций
        """
        return self.positions.copy()
        
    def get_total_positions_count(self) -> int:
        """Получение общего количества открытых позиций
        
        Returns:
            int: Общее количество открытых позиций
        """
        return sum(len(positions) for positions in self.positions.values())
        
    def get_positions_for_symbol(self, symbol: str) -> List[Dict]:
        """Получение всех позиций для конкретной торговой пары
        
        Args:
            symbol: Торговая пара
            
        Returns:
            List[Dict]: Список позиций для символа
        """
        return self.positions.get(symbol, [])
    
    def update_position(self, symbol: str, current_price: float) -> List[tuple]:
        """Обновление позиций и проверка достижения SL/TP
        
        Args:
            symbol: Торговая пара
            current_price: Текущая цена
            
        Returns:
            List[tuple]: Список кортежей (position_id, trigger_type) для позиций, достигших SL/TP
        """
        if symbol not in self.positions or not self.positions[symbol]:
            return []
            
        triggered_positions = []
        
        for position in self.positions[symbol]:
            side = position['side']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            position_id = position['id']
            
            # Проверяем достижение уровней SL/TP
            if side == "Buy":
                if current_price <= stop_loss:
                    triggered_positions.append((position_id, "stop_loss"))
                elif current_price >= take_profit:
                    triggered_positions.append((position_id, "take_profit"))
            else:  # Sell
                if current_price >= stop_loss:
                    triggered_positions.append((position_id, "stop_loss"))
                elif current_price <= take_profit:
                    triggered_positions.append((position_id, "take_profit"))
        
        return triggered_positions


class StrategySelector:
    """Интеллектуальный выбор стратегии на основе рыночных условий и исторической эффективности"""
    
    def __init__(self, strategy_cooldown_minutes: int = 30, data_file: str = "strategy_performance.json"):
        """Инициализация селектора стратегий
        
        Args:
            strategy_cooldown_minutes: Период кулдауна для стратегий в минутах
            data_file: Путь к файлу для сохранения данных
        """
        self.data_file = Path(data_file)
        self.strategy_performance: Dict[str, Dict] = {}
        self.strategy_cooldown: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=strategy_cooldown_minutes)
        
        # Загружаем сохраненные данные или инициализируем новые
        self._load_performance_data()
        
        # Инициализируем базовые метрики для всех стратегий (если не загружены)
        self._init_strategy_metrics()
        
        # История сигналов для анализа конфликтов
        self.signal_history: List[Dict] = []
    
    def _load_performance_data(self):
        """Загружает сохраненные данные о производительности стратегий"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.strategy_performance = data.get('strategy_performance', {})
                    
                    # Преобразуем datetime объекты из строк
                    for strategy_name, perf in self.strategy_performance.items():
                        if 'last_10_trades' in perf:
                            for trade in perf['last_10_trades']:
                                if 'timestamp' in trade and isinstance(trade['timestamp'], str):
                                    trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                    
                    logger.info(f"Загружены данные о производительности стратегий из {self.data_file}")
                    
                    # Логируем краткую сводку
                    for name, perf in self.strategy_performance.items():
                        if perf.get('total_trades', 0) > 0:
                            win_rate = (perf['winning_trades'] / perf['total_trades']) * 100
                            logger.info(f"  {name}: {perf['winning_trades']}/{perf['total_trades']} ({win_rate:.1f}%), PnL: ${perf['total_pnl']:.2f}")
            else:
                logger.info(f"Файл {self.data_file} не найден, создаем новые данные")
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            logger.info("Используем новые данные")
    
    def _save_performance_data(self):
        """Сохраняет данные о производительности стратегий в файл"""
        try:
            # Преобразуем datetime объекты в строки для JSON
            serializable_data = {}
            for strategy_name, perf in self.strategy_performance.items():
                serializable_perf = perf.copy()
                if 'last_10_trades' in serializable_perf:
                    for trade in serializable_perf['last_10_trades']:
                        if 'timestamp' in trade and hasattr(trade['timestamp'], 'isoformat'):
                            trade['timestamp'] = trade['timestamp'].isoformat()
                serializable_data[strategy_name] = serializable_perf
            
            data_to_save = {
                'strategy_performance': serializable_data,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Создаем директорию если не существует
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Данные о производительности сохранены в {self.data_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных: {e}")
        
    def _init_strategy_metrics(self):
        """Инициализация метрик для всех стратегий (только если они еще не существуют)"""
        strategies = ['sma_crossover', 'breakout', 'mean_reversion', 'rsi_strategy', 'half_year_strategy']
        
        for strategy in strategies:
            # Только создаем новые метрики, если они не загружены
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0.0,
                    'avg_trade_duration': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'last_10_trades': [],  # Для расчета краткосрочной эффективности
                    'base_weight': 1.0,  # Базовый вес стратегии
                    'volatility_preference': 1.0,  # Предпочтение волатильности (0.5-2.0)
                    'trend_preference': 1.0,  # Предпочтение тренда (0.5-2.0)
                }
            
        # Настройка предпочтений стратегий (всегда обновляем)
        self.strategy_performance['sma_crossover']['trend_preference'] = 1.8
        self.strategy_performance['sma_crossover']['volatility_preference'] = 0.7
        
        self.strategy_performance['breakout']['trend_preference'] = 1.5
        self.strategy_performance['breakout']['volatility_preference'] = 1.3
        
        self.strategy_performance['mean_reversion']['trend_preference'] = 0.6
        self.strategy_performance['mean_reversion']['volatility_preference'] = 1.8
        
        self.strategy_performance['rsi_strategy']['trend_preference'] = 0.8
        self.strategy_performance['rsi_strategy']['volatility_preference'] = 1.4
        
        self.strategy_performance['half_year_strategy']['trend_preference'] = 1.9
        self.strategy_performance['half_year_strategy']['volatility_preference'] = 0.5
    
    def calculate_strategy_score(self, strategy_name: str, market_conditions: Dict) -> float:
        """Рассчитывает комплексный рейтинг стратегии
        
        Args:
            strategy_name: Название стратегии
            market_conditions: Текущие рыночные условия
            
        Returns:
            float: Итоговый рейтинг стратегии (0-10)
        """
        if strategy_name not in self.strategy_performance:
            return 0.0
            
        perf = self.strategy_performance[strategy_name]
        
        # 1. Базовая эффективность (40% веса)
        efficiency_score = self._calculate_efficiency_score(perf)
        
        # 2. Соответствие рыночным условиям (30% веса)
        market_fit_score = self._calculate_market_fit_score(perf, market_conditions)
        
        # 3. Краткосрочная форма (20% веса)
        recent_form_score = self._calculate_recent_form_score(perf)
        
        # 4. Стабильность (10% веса)
        stability_score = self._calculate_stability_score(perf)
        
        # Итоговый рейтинг
        total_score = (
            efficiency_score * 0.4 +
            market_fit_score * 0.3 +
            recent_form_score * 0.2 +
            stability_score * 0.1
        )
        
        logger.debug(f"{strategy_name} score breakdown: efficiency={efficiency_score:.2f}, "
                    f"market_fit={market_fit_score:.2f}, recent={recent_form_score:.2f}, "
                    f"stability={stability_score:.2f}, total={total_score:.2f}")
        
        return min(total_score, 10.0)
    
    def _calculate_efficiency_score(self, perf: Dict) -> float:
        """Рассчитывает базовую эффективность стратегии"""
        if perf['total_trades'] == 0:
            return 5.0  # Нейтральный рейтинг для новых стратегий
            
        # Win rate (0-4 балла)
        win_rate = perf['winning_trades'] / perf['total_trades']
        win_score = min(win_rate * 8, 4.0)  # 50% = 4 балла, 100% = 4 балла
        
        # PnL score (0-3 балла)
        avg_pnl_per_trade = perf['total_pnl'] / perf['total_trades'] if perf['total_trades'] > 0 else 0
        pnl_score = min(max(avg_pnl_per_trade / 10, -1), 3.0)  # $10 прибыли = 3 балла
        
        # Sharpe ratio (0-3 балла)
        sharpe_score = min(max(perf['sharpe_ratio'], 0), 3.0)
        
        return win_score + pnl_score + sharpe_score
    
    def _calculate_market_fit_score(self, perf: Dict, market_conditions: Dict) -> float:
        """Рассчитывает соответствие стратегии текущим рыночным условиям"""
        volatility = market_conditions.get('volatility', 1.0)
        trend_strength = market_conditions.get('trend_strength', 0.5)
        
        # Оценка по волатильности
        vol_optimal = perf['volatility_preference']
        vol_score = 5.0 - abs(volatility - vol_optimal) * 2
        vol_score = max(0, min(vol_score, 5.0))
        
        # Оценка по силе тренда
        trend_optimal = perf['trend_preference']
        trend_score = 5.0 - abs(trend_strength - trend_optimal) * 5
        trend_score = max(0, min(trend_score, 5.0))
        
        return (vol_score + trend_score) / 2
    
    def _calculate_recent_form_score(self, perf: Dict) -> float:
        """Рассчитывает краткосрочную форму стратегии (последние 10 сделок)"""
        recent_trades = perf['last_10_trades']
        
        if len(recent_trades) < 3:
            return 5.0  # Нейтральный рейтинг
            
        recent_wins = sum(1 for trade in recent_trades if trade['success'])
        recent_win_rate = recent_wins / len(recent_trades)
        
        recent_pnl = sum(trade['pnl'] for trade in recent_trades)
        avg_recent_pnl = recent_pnl / len(recent_trades)
        
        # Комбинированная оценка краткосрочной формы
        form_score = recent_win_rate * 6 + min(max(avg_recent_pnl / 5, -2), 4)
        return max(0, min(form_score, 10.0))
    
    def _calculate_stability_score(self, perf: Dict) -> float:
        """Рассчитывает стабильность стратегии"""
        if perf['total_trades'] < 10:
            return 5.0
            
        # Чем меньше максимальная просадка, тем выше стабильность
        drawdown_score = max(0, 10 - perf['max_drawdown'])
        
        # Стабильность последних сделок
        recent_trades = perf['last_10_trades']
        if len(recent_trades) >= 5:
            pnl_variance = sum((trade['pnl'] - sum(t['pnl'] for t in recent_trades) / len(recent_trades)) ** 2 
                             for trade in recent_trades) / len(recent_trades)
            variance_score = max(0, 5 - pnl_variance / 10)
        else:
            variance_score = 5.0
            
        return (drawdown_score + variance_score) / 2
    
    def get_strategy_rankings(self, market_conditions: Dict) -> List[Tuple[str, float]]:
        """Получает ранжированный список всех стратегий
        
        Args:
            market_conditions: Текущие рыночные условия
            
        Returns:
            List[Tuple[str, float]]: Список кортежей (название_стратегии, рейтинг)
        """
        rankings = []
        current_time = datetime.now()
        
        for strategy_name in self.strategy_performance.keys():
            # Проверяем кулдаун
            if strategy_name in self.strategy_cooldown:
                if current_time - self.strategy_cooldown[strategy_name] < self.cooldown_period:
                    continue
                    
            score = self.calculate_strategy_score(strategy_name, market_conditions)
            rankings.append((strategy_name, score))
        
        # Сортируем по убыванию рейтинга
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def resolve_signal_conflicts(self, signals: Dict[str, Tuple[str, float, float]], 
                                market_conditions: Dict) -> Optional[Tuple[str, str, float, float]]:
        """Разрешает конфликты между сигналами разных стратегий
        
        Args:
            signals: Словарь {strategy_name: (signal, stop_loss, take_profit)}
            market_conditions: Текущие рыночные условия
            
        Returns:
            Optional[Tuple]: (winning_strategy, signal, stop_loss, take_profit) или None
        """
        if not signals:
            return None
            
        # Фильтруем только торговые сигналы (не Hold)
        trade_signals = {name: signal for name, signal in signals.items() 
                        if signal[0] != "Hold"}
        
        if not trade_signals:
            return None
            
        if len(trade_signals) == 1:
            # Только один сигнал - возвращаем его
            strategy_name, (signal, sl, tp) = next(iter(trade_signals.items()))
            return (strategy_name, signal, sl, tp)
        
        # Анализируем конфликты
        buy_signals = {name: sig for name, sig in trade_signals.items() if sig[0] == "Buy"}
        sell_signals = {name: sig for name, sig in trade_signals.items() if sig[0] == "Sell"}
        
        logger.info(f"Signal conflict detected: {len(buy_signals)} BUY, {len(sell_signals)} SELL")
        
        # Если все сигналы в одном направлении - выбираем лучшую стратегию
        if buy_signals and not sell_signals:
            return self._select_best_signal(buy_signals, market_conditions)
        elif sell_signals and not buy_signals:
            return self._select_best_signal(sell_signals, market_conditions)
        else:
            # Конфликт между Buy и Sell - выбираем стратегию с наивысшим рейтингом
            logger.warning("Buy/Sell conflict detected - choosing highest rated strategy")
            all_rankings = self.get_strategy_rankings(market_conditions)
            
            for strategy_name, score in all_rankings:
                if strategy_name in trade_signals:
                    signal, sl, tp = trade_signals[strategy_name]
                    logger.info(f"Conflict resolved: {strategy_name} wins with score {score:.2f} -> {signal}")
                    return (strategy_name, signal, sl, tp)
                    
        return None
    
    def _select_best_signal(self, signals: Dict[str, Tuple[str, float, float]], 
                           market_conditions: Dict) -> Optional[Tuple[str, str, float, float]]:
        """Выбирает лучший сигнал из однонаправленных сигналов"""
        rankings = self.get_strategy_rankings(market_conditions)
        
        for strategy_name, score in rankings:
            if strategy_name in signals:
                signal, sl, tp = signals[strategy_name]
                logger.info(f"Best signal selected: {strategy_name} (score: {score:.2f}) -> {signal}")
                return (strategy_name, signal, sl, tp)
                
        return None
    
    def select_best_strategy(self, symbol: str, market_conditions: Dict) -> Optional[tuple]:
        """Выбор лучшей стратегии для текущих рыночных условий
        
        Args:
            symbol: Торговая пара
            market_conditions: Рыночные условия (волатильность, сила тренда, объемы)
            
        Returns:
            Optional[tuple]: Кортеж (название стратегии, функция стратегии, рейтинг) или None
        """
        rankings = self.get_strategy_rankings(market_conditions)
        
        if not rankings:
            logger.warning(f"{symbol}: no strategies available (all in cooldown)")
            return None
            
        # Выбираем стратегию с наивысшим рейтингом
        best_strategy_name, best_score = rankings[0]
        
        # Маппинг названий на функции
        strategy_functions = {
            'sma_crossover': sma_crossover,
            'breakout': breakout,
            'mean_reversion': mean_reversion,
            'rsi_strategy': rsi_strategy,
            'half_year_strategy': half_year_strategy,
        }
        
        if best_strategy_name not in strategy_functions:
            logger.error(f"Unknown strategy: {best_strategy_name}")
            return None
            
        strategy_func = strategy_functions[best_strategy_name]
        
        logger.info(f"{symbol}: selected strategy {best_strategy_name} with score {best_score:.2f}")
        return (best_strategy_name, strategy_func, best_score)
    
    def record_trade_result(self, strategy_name: str, success: bool, pnl: float, 
                           trade_duration_minutes: int = 0):
        """Запись результата сделки для обновления рейтинга стратегии
        
        Args:
            strategy_name: Название стратегии
            success: Успешность сделки
            pnl: Прибыль/убыток по сделке
            trade_duration_minutes: Длительность сделки в минутах
        """
        if strategy_name not in self.strategy_performance:
            return
            
        perf = self.strategy_performance[strategy_name]
        
        # Обновляем основные метрики
        perf['total_trades'] += 1
        if success:
            perf['winning_trades'] += 1
        else:
            perf['losing_trades'] += 1
            # Устанавливаем кулдаун для неуспешных стратегий
            self.strategy_cooldown[strategy_name] = datetime.now()
            
        perf['total_pnl'] += pnl
        
        # Обновляем средние значения
        if perf['total_trades'] > 0:
            avg_duration = perf['avg_trade_duration']
            perf['avg_trade_duration'] = ((avg_duration * (perf['total_trades'] - 1) + 
                                         trade_duration_minutes) / perf['total_trades'])
        
        # Добавляем в историю последних сделок
        trade_record = {
            'success': success,
            'pnl': pnl,
            'duration': trade_duration_minutes,
            'timestamp': datetime.now()
        }
        
        perf['last_10_trades'].append(trade_record)
        if len(perf['last_10_trades']) > 10:
            perf['last_10_trades'].pop(0)
            
        # Обновляем максимальную просадку
        if pnl < 0:
            drawdown = abs(pnl) / max(abs(perf['total_pnl']), 100) * 100
            perf['max_drawdown'] = max(perf['max_drawdown'], drawdown)
            
        # Пересчитываем Sharpe ratio
        self._update_sharpe_ratio(strategy_name)
        
        logger.info(f"Updated {strategy_name}: {perf['winning_trades']}/{perf['total_trades']} "
                   f"wins, total PnL: ${perf['total_pnl']:.2f}")
    
    def _update_sharpe_ratio(self, strategy_name: str):
        """Обновляет Sharpe ratio для стратегии"""
        perf = self.strategy_performance[strategy_name]
        recent_trades = perf['last_10_trades']
        
        if len(recent_trades) < 5:
            return
            
        returns = [trade['pnl'] for trade in recent_trades]
        avg_return = sum(returns) / len(returns)
        
        if len(returns) > 1:
            variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = variance ** 0.5
            
            if std_dev > 0:
                # Предполагаем безрисковую ставку 0
                perf['sharpe_ratio'] = avg_return / std_dev
            else:
                perf['sharpe_ratio'] = 0
    
    def get_performance_summary(self) -> str:
        """Возвращает сводку по эффективности всех стратегий"""
        summary = ["\n=== STRATEGY PERFORMANCE SUMMARY ==="]
        
        # Сортируем стратегии по винрейту
        strategies = [(name, perf) for name, perf in self.strategy_performance.items()]
        strategies.sort(key=lambda x: x[1]['winning_trades'] / max(x[1]['total_trades'], 1), reverse=True)
        
        for name, perf in strategies:
            if perf['total_trades'] > 0:
                win_rate = (perf['winning_trades'] / perf['total_trades']) * 100
                avg_pnl = perf['total_pnl'] / perf['total_trades']
                summary.append(
                    f"{name:15} | {perf['winning_trades']:3}/{perf['total_trades']:3} "
                    f"({win_rate:5.1f}%) | PnL: ${perf['total_pnl']:8.2f} "
                    f"(${avg_pnl:6.2f}/trade) | Sharpe: {perf['sharpe_ratio']:5.2f}"
                )
            else:
                summary.append(f"{name:15} | No trades yet")
                
        return "\n".join(summary)


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
            max_positions_per_symbol=config.max_positions_per_symbol,
            max_risk_per_trade=config.max_risk_per_trade,
            min_trade_interval_minutes=config.min_trade_interval_minutes
        )
        self.max_total_positions = config.max_total_positions
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
                # Проверяем, что передан strategy_name в качестве последнего параметра
                strategy_name = getattr(self, '_current_strategy', None)
                self.position_manager.add_position(symbol, side, amount, price, stop_loss, take_profit, strategy_name)
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
        """Интеллектуальная торговля с множественными стратегиями и разрешением конфликтов
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Optional[dict]: Результат размещения ордера или None
        """
        
        # Проверяем, можем ли открыть позицию
        if not self.position_manager.can_open_position(symbol):
            return None
            
        # Проверяем общее количество позиций
        total_positions = self.position_manager.get_total_positions_count()
        if total_positions >= self.max_total_positions:
            logger.info(f"Cannot open {symbol}: max total positions reached ({total_positions}/{self.max_total_positions})")
            return None
        
        # Обновляем баланс
        balance = self.update_balance()
        min_balance = self.config.min_trade_amount * 5
        if balance < min_balance:
            logger.warning(f"Insufficient balance: ${balance:.2f} (minimum required: ${min_balance:.2f})")
            return None
        
        # Анализируем рыночные условия
        market_conditions = self.calculate_market_conditions(symbol)
        logger.info(f"{symbol} market: volatility={market_conditions['volatility']:.2f}, "
                   f"trend={market_conditions['trend_strength']:.2f}, "
                   f"volume_ratio={market_conditions['volume_ratio']:.2f}")
        
        # Получаем сигналы от всех стратегий
        all_signals = self._get_all_strategy_signals(symbol, market_conditions)
        
        if not all_signals:
            logger.info(f"{symbol}: no strategy signals available")
            return None
        
        # Логируем все сигналы
        signal_summary = ", ".join([f"{name}:{sig[0]}" for name, sig in all_signals.items()])
        logger.info(f"{symbol} signals: {signal_summary}")
        
        # Разрешаем конфликты и выбираем лучший сигнал
        final_decision = self.strategy_selector.resolve_signal_conflicts(all_signals, market_conditions)
        
        if not final_decision:
            logger.info(f"{symbol}: no clear trading signal after conflict resolution")
            return None
        
        winning_strategy, signal, stop_loss, take_profit = final_decision
        logger.info(f"{symbol}: final decision - {winning_strategy} -> {signal}, SL=${stop_loss:.2f}, TP=${take_profit:.2f}")
        
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
            # Запоминаем время начала сделки и стратегию
            trade_start_time = datetime.now()
            self._current_strategy = winning_strategy  # Сохраняем для use в place_order
            
            price = self._last_price(symbol)
            order_result = self.place_order(
                symbol, signal, position_size, leverage, stop_loss, take_profit, price
            )
            
            if order_result.get("retCode") == 0:
                logger.info(f"{symbol}: order placed successfully via {winning_strategy}")
                return order_result
            else:
                logger.error(f"{symbol}: order failed: {order_result}")
                self.strategy_selector.record_trade_result(winning_strategy, False, 0)
                return None
                
        except Exception as e:
            logger.error(f"{symbol}: failed to place order: {e}")
            return None
        finally:
            # Очищаем временную переменную
            if hasattr(self, '_current_strategy'):
                delattr(self, '_current_strategy')
    
    def _get_all_strategy_signals(self, symbol: str, market_conditions: Dict) -> Dict[str, Tuple[str, float, float]]:
        """Получает сигналы от всех доступных стратегий
        
        Args:
            symbol: Торговая пара
            market_conditions: Рыночные условия
            
        Returns:
            Dict[str, Tuple]: Словарь {strategy_name: (signal, stop_loss, take_profit)}
        """
        strategies_to_test = {
            'sma_crossover': {'interval': 5, 'limit': 50},
            'breakout': {'interval': 5, 'limit': 50},
            'mean_reversion': {'interval': 5, 'limit': 50},
            'rsi_strategy': {'interval': 5, 'limit': 50},
            'half_year_strategy': {'interval': 240, 'limit': 200},
        }
        
        strategy_functions = {
            'sma_crossover': sma_crossover,
            'breakout': breakout,
            'mean_reversion': mean_reversion,
            'rsi_strategy': rsi_strategy,
            'half_year_strategy': half_year_strategy,
        }
        
        all_signals = {}
        
        for strategy_name, params in strategies_to_test.items():
            try:
                # Проверяем кулдаун
                if strategy_name in self.strategy_selector.strategy_cooldown:
                    time_since_cooldown = datetime.now() - self.strategy_selector.strategy_cooldown[strategy_name]
                    if time_since_cooldown < self.strategy_selector.cooldown_period:
                        logger.debug(f"{symbol}: {strategy_name} in cooldown")
                        continue
                
                # Получаем данные для стратегии
                result = self.session.get_kline(
                    category="linear", 
                    symbol=symbol, 
                    interval=params['interval'], 
                    limit=params['limit']
                )
                
                candles = result.get("result", {}).get("list", [])
                if len(candles) < params['limit']:
                    logger.warning(f"{symbol}: not enough data for {strategy_name} ({len(candles)}/{params['limit']})")
                    continue
                
                # Преобразуем данные
                candles = [list(map(float, c[:6])) for c in reversed(candles)]
                
                # Получаем сигнал от стратегии
                strategy_func = strategy_functions[strategy_name]
                signal, stop, take = strategy_func(candles)
                
                # Применяем ограничения на SL/TP
                price = candles[-1][4]
                stop, take = apply_sl_tp_bounds(price, signal, stop, take)
                
                all_signals[strategy_name] = (signal, stop, take)
                
                logger.debug(f"{symbol}: {strategy_name} -> {signal} (SL: ${stop:.2f}, TP: ${take:.2f})")
                
            except Exception as e:
                logger.error(f"{symbol}: error getting signal from {strategy_name}: {e}")
                continue
        
        return all_signals

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
        """Получение подробной статистики по всем стратегиям"""
        return self.strategy_selector.get_performance_summary()
    
    def close_position(self, symbol: str, side: str, amount: float, position_id: str = None) -> dict:
        """Закрытие позиции с отслеживанием результатов стратегии
        
        Args:
            symbol: Торговая пара
            side: Сторона позиции (Buy/Sell)
            amount: Размер позиции
            position_id: ID конкретной позиции
            
        Returns:
            dict: Результат закрытия позиции
        """
        # Получаем информацию о позиции до закрытия
        position_info = self.position_manager.get_position(symbol, position_id)
        strategy_used = None
        trade_start_time = None
        entry_price = None
        
        if position_info:
            strategy_used = position_info.get('strategy_used')
            trade_start_time = position_info.get('trade_start_time')
            entry_price = position_info.get('entry_price')
        
        try:
            close_side = "Sell" if side == "Buy" else "Buy"
            qty = self._format_qty(symbol, amount)
            
            logger.info(f"Closing position: {symbol} {close_side} qty={qty}")
            
            # Получаем текущую цену для расчета PnL
            current_price = self._last_price(symbol)
            
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
                logger.info(f"Position closed successfully: {symbol} {close_side} qty={qty}")
                
                # Рассчитываем PnL и обновляем статистику стратегии
                if strategy_used and entry_price:
                    # Рассчитываем PnL
                    if side == "Buy":
                        actual_pnl_usd = (current_price - entry_price) * amount / entry_price
                    else:  # Sell
                        actual_pnl_usd = (entry_price - current_price) * amount / entry_price
                    
                    # Определяем успешность сделки
                    success = actual_pnl_usd > 0
                    
                    # Рассчитываем длительность сделки
                    trade_duration = 0
                    if trade_start_time:
                        trade_duration = int((datetime.now() - trade_start_time).total_seconds() / 60)
                    
                    # Обновляем статистику стратегии
                    self.strategy_selector.record_trade_result(
                        strategy_used, success, actual_pnl_usd, trade_duration
                    )
                    
                    # Автоматически сохраняем данные
                    self.strategy_selector._save_performance_data()
                    
                    pnl_pct = (actual_pnl_usd / amount) * 100
                    logger.info(f"Trade completed: {strategy_used} -> {'SUCCESS' if success else 'LOSS'} "
                               f"PnL: ${actual_pnl_usd:.2f} ({pnl_pct:+.2f}%) Duration: {trade_duration}min")
                
                # Удаляем позицию из менеджера
                self.position_manager.remove_position(symbol, position_id)
            else:
                logger.error(f"Failed to close position {symbol} - code: {ret_code}, message: {ret_msg}")
                # Логируем дополнительную информацию об ошибке
                if "error" in result:
                    logger.error(f"{symbol}: Error details: {result['error']}")
                # Удаляем позицию из менеджера даже при ошибке API
                self.position_manager.remove_position(symbol, position_id)
            
            return result
        except Exception as e:
            logger.error(f"Exception closing position {symbol}: {e}")
            import traceback
            logger.error(f"{symbol}: Traceback: {traceback.format_exc()}")
            return {"retCode": -1, "retMsg": f"Exception: {str(e)}"}
    
    def check_and_close_positions(self) -> None:
        """Проверка и закрытие позиций при достижении SL/TP
        
        Проверяет все открытые позиции и закрывает те, где достигнуты уровни SL/TP
        """
        all_positions = self.position_manager.get_all_positions()
        for symbol in list(all_positions.keys()):
            try:
                current_price = self._last_price(symbol)
                triggered_positions = self.position_manager.update_position(symbol, current_price)
                
                for position_id, trigger_type in triggered_positions:
                    position = self.position_manager.get_position(symbol, position_id)
                    if position:
                        if trigger_type == "stop_loss":
                            logger.info(f"Stop-loss reached for {symbol} (ID: {position_id}) at price {current_price}")
                            self.close_position(symbol, position['side'], position['amount'], position_id)
                        elif trigger_type == "take_profit":
                            logger.info(f"Take-profit reached for {symbol} (ID: {position_id}) at price {current_price}")
                            self.close_position(symbol, position['side'], position['amount'], position_id)
            except Exception as e:
                logger.error(f"Error checking positions for {symbol}: {e}")


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
    logger.info(f"Max positions per symbol: {cfg.max_positions_per_symbol}")
    logger.info(f"Max total positions: {cfg.max_total_positions}")
    logger.info(f"Allowed symbols: {', '.join(bot.ALLOWED_SYMBOLS)}")
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
        total_positions = bot.position_manager.get_total_positions_count()
        if total_positions > 0:
            position_summary = []
            for symbol, positions in bot.position_manager.get_all_positions().items():
                position_summary.append(f"{symbol}:{len(positions)}")
            logger.info(f"Open positions ({total_positions}/{bot.max_total_positions}): {', '.join(position_summary)}")
        
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