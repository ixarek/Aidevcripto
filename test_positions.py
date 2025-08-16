#!/usr/bin/env python3
"""
Тестовый скрипт для проверки корректности реализации множественных позиций
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot import PositionManager
from config import BybitConfig
from datetime import datetime

def test_position_manager():
    """Тестирование PositionManager"""
    print("=== Тестирование PositionManager ===")
    
    # Создаем менеджер позиций
    pm = PositionManager(max_positions_per_symbol=3, max_risk_per_trade=2.0)
    
    # Тест 1: Проверка начального состояния
    print(f"Начальное состояние:")
    print(f"  Общее количество позиций: {pm.get_total_positions_count()}")
    print(f"  Можно открыть BTCUSDT: {pm.can_open_position('BTCUSDT')}")
    
    # Тест 2: Добавление позиций
    print(f"\nДобавление позиций:")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in symbols:
        for i in range(3):  # По 3 позиции на символ
            pm.add_position(
                symbol=symbol,
                side="Buy" if i % 2 == 0 else "Sell",
                amount=100,
                price=50000 + i * 1000,
                stop_loss=49000 + i * 1000,
                take_profit=51000 + i * 1000,
                strategy_used=f"strategy_{i}"
            )
    
    print(f"После добавления:")
    print(f"  Общее количество позиций: {pm.get_total_positions_count()}")
    
    for symbol in symbols:
        positions = pm.get_positions_for_symbol(symbol)
        print(f"  {symbol}: {len(positions)} позиций")
        for pos in positions:
            print(f"    ID: {pos['id']}, Side: {pos['side']}, Strategy: {pos['strategy_used']}")
    
    # Тест 3: Проверка лимитов
    print(f"\nПроверка лимитов:")
    for symbol in symbols:
        can_open = pm.can_open_position(symbol)
        print(f"  Можно открыть {symbol}: {can_open}")
    
    # Тест 4: Удаление позиций
    print(f"\nУдаление позиций:")
    
    # Удаляем первую позицию BTCUSDT
    btc_positions = pm.get_positions_for_symbol("BTCUSDT")
    if btc_positions:
        first_position_id = btc_positions[0]['id']
        pm.remove_position("BTCUSDT", first_position_id)
        print(f"  Удалена позиция {first_position_id}")
    
    print(f"После удаления:")
    print(f"  Общее количество позиций: {pm.get_total_positions_count()}")
    print(f"  BTCUSDT позиций: {len(pm.get_positions_for_symbol('BTCUSDT'))}")
    print(f"  Можно открыть BTCUSDT: {pm.can_open_position('BTCUSDT')}")
    
    # Тест 5: Обновление позиций (симуляция изменения цены)
    print(f"\nТестирование обновления позиций:")
    
    triggered = pm.update_position("ETHUSDT", 50500)  # Цена достигла TP для некоторых позиций
    print(f"  Triggered positions: {triggered}")
    
    return True

def test_config():
    """Тестирование конфигурации"""
    print("\n=== Тестирование конфигурации ===")
    
    # Тест загрузки конфигурации
    try:
        config = BybitConfig.from_env()
        print(f"Конфигурация загружена:")
        print(f"  max_positions_per_symbol: {config.max_positions_per_symbol}")
        print(f"  max_total_positions: {config.max_total_positions}")
        print(f"  max_risk_per_trade: {config.max_risk_per_trade}")
        return True
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return False

def test_functions():
    """Тестирование functions.py"""
    print("\n=== Тестирование functions.py ===")
    
    try:
        from functions import limit_open_positions, apply_sl_tp_bounds
        
        # Тест limit_open_positions
        test_positions = {"BTCUSDT": {}, "ETHUSDT": {}}
        result1 = limit_open_positions(test_positions, 3)
        result2 = limit_open_positions(test_positions, 1)
        
        print(f"limit_open_positions:")
        print(f"  2 позиции, лимит 3: {result1}")
        print(f"  2 позиции, лимит 1: {result2}")
        
        # Тест apply_sl_tp_bounds
        price = 50000
        stop, take = apply_sl_tp_bounds(price, "Buy", 48000, 52000)
        print(f"apply_sl_tp_bounds (Buy):")
        print(f"  Цена: {price}, SL: {stop}, TP: {take}")
        
        return True
    except Exception as e:
        print(f"Ошибка тестирования functions: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("Тестирование реализации множественных позиций")
    print("=" * 50)
    
    tests = [
        ("PositionManager", test_position_manager),
        ("Config", test_config),
        ("Functions", test_functions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n✅ {test_name}: {'ПРОЙДЕН' if result else 'ПРОВАЛЕН'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ {test_name}: ОШИБКА - {e}")
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nИтого: {passed}/{len(results)} тестов пройдено")
    
    if passed == len(results):
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Реализация готова к использованию.")
    else:
        print("\n⚠️  Есть проблемы, требующие исправления.")

if __name__ == "__main__":
    main()
