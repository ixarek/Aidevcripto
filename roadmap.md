**Создание криптобота для работы на фьючерсном рынке Bybit**  
*Полноценный roadmap, методы торговли, сигналы, ограничения и практические рекомендации.*  

---  

## 1. Введение  

Криптобот – это программный автомат, который получает рыночные данные, генерирует торговые сигналы и отправляет заявки в биржу без участия человека. На Bybit (фьючерсы BTC, ETH, альткоины, а также USDT‑перпетуалы) доступны мощные API, позволяющие реализовать любую стратегию от простых скользящих‑средних до сложных моделей машинного обучения.  

В этом документе мы разберём:  

1. **Ключевые методы и типы стратегий**.  
2. **Сигналы и индикаторы, которые можно использовать**.  
3. **Ограничения биржи и риск‑менеджмент**.  
4. **Поэтапный roadmap** — что делаем, какие задачи можно делегировать ИИ, какие требуют ручного контроля.  

---  

## 2. Архитектурный обзор

| Слой | Описание | Технологии / Инструменты |
|------|----------|--------------------------|
| **Data Ingestion** | Получение тикеров, стакана, исторических свечей, позиции и баланса | REST API (GET / v2/public/tickers), WebSocket (order‑book, trades, kline) |
| **Storage** | Хранение исторических данных для бэктестинга и онлайн‑аналитики | PostgreSQL / ClickHouse, TimeSeries DB (InfluxDB) |
| **Signal Engine** | Вычисление индикаторов, генерация торговых сигналов | Python (pandas, TA‑Lib, numpy), R, Julia, или **AI‑модели** (TensorFlow/PyTorch) |
| **Risk & Money Management** | Расчёт позиции, лимитов, SL/TP, контроль маржи | Python (risk‑libs), кастомные правила |
| **Order Execution** | Формирование, отправка и отслеживание ордеров | REST (POST / v2/private/order/create), WebSocket (order‑update) |
| **Monitoring & Alerting** | Логи, метрики, алерты, UI‑дашборд | Grafana + Prometheus, Telegram Bot, Slack, Discord |
| **CI/CD & Deployment** | Автоматическое тестирование, выкладка в продакшн | Docker, Kubernetes, GitHub Actions, Terraform |

---  

## 3. Методы торговли (стратегии)

| Категория | Краткое описание | Пример сигнала | Преимущества | Недостатки |
|-----------|------------------|----------------|--------------|------------|
| **Trend‑following** (трендовые) | Открывать позиции в направлении текущего тренда. | Пересечение 50‑MA и 200‑MA (golden cross) → LONG | Простота, хорошо работает в сильных трендах | Потери в боковом рынке |
| **Mean reversion** (рецессия к среднему) | Ожидание возврата цены к уровню среднего. | RSI > 70 → SHORT, RSI < 30 → LONG | Подходит для колебаний, может давать быстрые прибыль | Риск в сильных трендах |
| **Market‑making / Grid** | Размещение лимит‑ордеров по сетке цен, получение прибыли от спреда. | Стакан: разместить BUY на -0.5 % и SELL на +0.5 % от цены | Пассивный доход, высокая частота сделок | Требует большого капитала и контроля маржи |
| **Scalping** | Микросделки в течение нескольких секунд‑минут. | Imbalance (Bid > Ask by 5 %) → BUY, затем SELL в‑ближайшие 10 сек | Высокий теоретический APR | Чувствительность к комиссии и задержкам |
| **Arbitrage (прямая)** | Использовать ценовые различия между биржами или сегментами (Spot ↔ Futures). | Spot‑price = 55 000, Futures‑price = 55 500 → купить Spot, продать Futures | Практически без риска | Нужен быстрый кросс‑биржевой трансфер и большие лоты |
| **Funding‑rate trading** | Играть на разнице между фьючерсным и спотовым рынком, используя Funding Rate. | Funding > 0, цена Futures > Spot → SHORT Futures | Пассивный доход от финансирования | Выборка ограничена, может быть неустойчивой |
| **ML‑based / Predictive** | Прогноз цены/волатильности на основе исторических данных, новостей, соцсетей. | LSTM предсказывает рост цены → LONG | Потенциально высокая эффективность | Требует датасетов, вычислительных ресурсов, риск переобучения |

> **Tip для ИИ**: При выборе стратегии подберите набор индикаторов, которые будем вычислять в `Signal Engine`. ИИ‑модель может сгенерировать оптимальные параметры (периоды MA, пороги RSI) на основе исторических данных.  

---  

## 4. Сигналы и индикаторы

| Индикатор | Применение | Параметры (по умолчанию) | Как реализовать (Python) |
|-----------|------------|--------------------------|--------------------------|
| **EMA / SMA** | Тренд, уровни поддержки/сопротивления | 20, 50, 200 | `ta.ema(close, length=20)` |
| **MACD** | Перекрёсток быстрой/медленной линии | fast=12, slow=26, signal=9 | `ta.macd(close, fast=12, slow=26, signal=9)` |
| **RSI** | Перепроданность/перекупленность | period=14 | `ta.rsi(close, length=14)` |
| **Bollinger Bands** | Волатильность, breakout | period=20, std=2 | `ta.bbands(close, length=20, std=2)` |
| **Order‑book imbalance** | Оценка давления покупателей/продавцов | (bid_vol‑ask_vol)/(bid_vol+ask_vol) | `imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)` |
| **Funding Rate** | Оценка будущего направления (отрицательный → LONG) | текущий период | `funding_rate = client.get_funding_rate(symbol)` |
| **Open Interest Δ** | Увеличение открытого интереса → подтверждение тренда | ΔOI за 1 h | `oi_change = oi_current - oi_prev` |
| **Volume Weighted Average Price (VWAP)** | Средняя цена, используемая институционалами | – | `vwap = (price * volume).cumsum() / volume.cumsum()` |
| **News / Sentiment** | Фундаментальные драйверы | Тональность в % | **AI**: использовать GPT‑4/ChatGPT в качестве классификатора новостей. |

> **AI‑задача**: собрать исторические данные по этим индикаторам (цены, OI, funding) и построить dataframe‑тесты, чтобы проверить их корреляцию со следующей свечой (1 h).  

---  

## 5. Ограничения и риски (Bybit)

| Ограничение | Описание | Как соблюдать |
|-------------|----------|----------------|
| **API‑rate‑limit** | HTTP → ≈ 120 req/мин, WS → ≈ 100 msg/сек. | Пул запросов, лимитирование (`asyncio.Semaphore`). |
| **Минимальный размер ордера** | 0.001 BTC (может отличаться от актива). | Проверять `min_trade_amount` через `/v2/public/symbols`. |
| **Максимальное кредитное плечо** | До 100× (для BTC) – 50× (для ETH) и ниже для альткоинов. | Ограничить `max_leverage` в `Risk Engine`. |
| **Требования к марже** | Initial Margin ≈ 1/Leverage, Maintenance ≈ 0.5 × Initial. | При расчёте позиции проверять текущий `available_margin`. |
| **Funding‑rate schedule** | Каждые 8 часов, может быть ± 0.01 % – 0.1 %. | Добавить задачу, получающую предсказание будущего funding‑rate. |
| **Блокировка по IP / 2FA** | При больших объёмов требуется дополнительный KYC и IP‑whitelist. | Хранить API‑ключи в безопасном Vault, добавить IP в `IP‑Whitelist`. |
| **Системные сбои / ликвидность** | В периоды высокой волатильности могут происходить проскальзывания. | Ограничить `max_slippage`, использовать `postOnly`‑ордера, ставить `cancelOnDisconnect`. |
| **Регуляторные ограничения** | В некоторых юрисдикциях фьючерсы запрещены. | Проводите KYC, проверяйте, что пользователи находятся в разрешённых регионах. |

---  

## 6. Roadmap (поэтапный план)

> **Пометка**: задачи, которые удобно реализовать или проверить с помощью ИИ, отмечены **[AI]**.  

### Этап 0 – ПОДГОТОВКА
| Шаг | Описание | Инструменты | Ответственный |
|-----|----------|--------------|----------------|
| 0.1 | Оформление аккаунта, включение API‑ключей (read‑only + trade) | Bybit UI | Человек |
| 0.2 | Настройка безопасного хранилища (HashiCorp Vault, AWS Secrets Manager) | Vault | Человек |
| 0.3 | Выбор языка/фреймворка (Python 3.11 + asyncio) | PyCharm, VS Code | Человек |

### Этап 1 – СБОР ДАННЫХ
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 1.1 **[AI]** | Написать скрипт‑парсер исторических свечей (1 min – 1 day) для выбранных символов | `client.query_kline(symbol, interval, **kwargs)` | ИИ генерирует код, человек ревью |
| 1.2 | Сохранить данные в базе (PostgreSQL + TimescaleDB) | Таблица `candles (ts TIMESTAMP, open, high, low, close, volume)` | Человек/DevOps |
| 1.3 **[AI]** | Сбор данных по funding‑rate, open‑interest, order‑book | WebSocket `public/linear/kline`, `public/linear/funding_rate` | ИИ генерирует парсеры |
| 1.4 | Валидация целостности (пробелы, дубликаты) | Pandas `df.isnull().sum()` | Человек |

### Этап 2 – BACKTEST‑ФРЕЙМВОРК
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 2.1 **[AI]** | Сгенерировать skeleton проекта `backtesting.py` с `vectorbt`/`backtrader` | `class MyStrategy(bt.Strategy):` | ИИ + ревью |
| 2.2 | Реализовать расчёт индикаторов (EMA, RSI, имбаланс) | `ta`‑библиотека | Человек |
| 2.3 | Добавить модуль `RiskEngine` (позиционный размер, SL/TP) | `max_risk=0.02` от баланса | Человек |
| 2.4 | Запустить тесты на исторических данных, собрать метрики (Sharpe, MaxDD, WinRate) | `bt.run()` | Человек |
| 2.5 | После стандартных бэктестов выполнить стресс‑тесты на исторических обвалах (март 2020, ноябрь 2022, события 2025) с помощью `stress_tests.py`. Скрипт оценивает P&L, максимальную просадку и ордер‑лог, позволяя понять устойчивость стратегии в экстремальных условиях | `python stress_tests.py BTC-USD 2020-03-01 2020-04-01` | Человек |
| 2.6 **[AI]** | Автоматический поиск оптимальных параметров (grid‑search, Bayesian) | `optuna` | ИИ генерирует скрипт, человек запускает |

### Этап 3 – LIVE‑DATA ENGINE
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 3.1 **[AI]** | Сгенерировать асинхронный клиент WebSocket (пинг/понт, реконнекты) | `websockets.connect(url, ping_interval=20)` | ИИ + ревью |
| 3.2 | На лету рассчитывать индикаторы (rolling windows) | `pandas.rolling` | Человек |
| 3.3 | Встроить фильтрацию шумов (EWMA, Kalman) | `statsmodels.tsa.filters.kalman` | Человек |
| 3.4 | Вывести сигналы в очередь (Redis, RabbitMQ) | `publish(signal_msg)` | DevOps |

### Этап 4 – RISK & MONEY MANAGEMENT
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 4.1 | **Position Sizing** – рассчитывать размер ордера: `size = balance * risk_per_trade / (stop_distance)` | `Decimal` для точности | Человек |
| 4.2 | **Leverage Control** – ограничить до `max_leverage = 20x` (по‑умолчанию) | `client.change_leverage(symbol, max_leverage)` | Человек |
| 4.3 | **Stop‑Loss / Take‑Profit** – динамические уровни, трейлинг‑stop | `client.place_order(..., stopLoss=price, takeProfit=price)` | Человек |
| 4.4 **[AI]** | Генерация сценариев стресс‑тестов (мок‑цен) | `pytest`‑тесты, имитация падения цены 30% | ИИ генерирует тест‑сценарии |
| 4.5 | **Параметры лимитов** – max open positions = 5, max drawdown = 15% | Локальная БД `limits` | Человек |

### Этап 5 – ORDER EXECUTION ENGINE
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 5.1 **[AI]** | Сгенерировать класс `OrderExecutor` с поддержкой: `limit`, `market`, `postOnly`, `ioc` | `client.place_active_order(...)` | ИИ + ревью |
| 5.2 | Реализовать **order throttling** (не более 10 запросов/сек) | `asyncio.Semaphore(10)` | Человек |
| 5.3 | Обрабатывать **order updates** через WS (`order` channel) – подтверждать исполнение, проверять статус | `on_message` → `order_status` | Человек |
| 5.4 | В случае **rejection** → логировать, переоценить сигнал | `logger.error(...)` | Человек |
| 5.5 | **Кросс‑марж**: проверять, достаточно ли маржи перед открытием позиции | `client.get_wallet_balance()` | Человек |

### Этап 6 – MONITORING & ALERTING
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 6.1 | Export метрик в **Prometheus** (latency, success‑rate, equity) | `client.export_metric(...)` | DevOps |
| 6.2 | Дашборд **Grafana**: P&L, открытые позиции, текущий leverage | Grafana panels | DevOps |
| 6.3 **[AI]** | Сгенерировать Telegram‑бота‑алёртера (смс о SL, падении equity) | `python-telegram-bot` | ИИ + ревью |
| 6.4 | **Audit‑log**: сохранять все запросы/ответы в S3/MinIO (immutable) | `boto3` | DevOps |

### Этап 7 – DEPLOYMENT & SCALING
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 7.1 | Docker‑образ с `python:3.11-slim` + зависимости | `Dockerfile` | Человек |
| 7.2 | Подготовить **K8s‑деплоймент** (ReplicaSet = 1, авто‑restart) | `deployment.yaml` | DevOps |
| 7.3 | CI/CD: тесты → билд → push → деплой | GitHub Actions | DevOps |
| 7.4 | **Blue‑Green** переключение при обновлении стратегии | `service`‑selector | DevOps |
| 7.5 | Плановое **backup** БД и конфигураций (daily) | `cronjob` | DevOps |

### Этап 8 – CONTINUOUS IMPROVEMENT
| Шаг | Описание | Тех. детали | Ответственный |
|-----|----------|--------------|----------------|
| 8.1 | Сбор обратной связи: P&L, slippage, частота отмен | `pandas`‑отчёты | Аналитик |
| 8.2 **[AI]** | Тренировать модель‑прогноз **price direction** на новых данных (LSTM, XGBoost) | `sklearn`, `tensorflow` | ИИ генерирует код, человек обучает |
| 8.3 | Применять **walk‑forward validation** перед выкладкой новой модели | `mlfinlab` | Аналитик |
| 8.4 | Регулярно проверять **регуляторные изменения** (AML/KYC) | RSS‑ленты, email‑подписки | Compliance‑officer |
| 8.5 | Планировать **масштабирование** (добавление новых символов, мульти‑бит) | горизонтальное масштабирование pods | DevOps |

---  

## 7. Практические рекомендации  

| Тема | Совет |
|------|-------|
| **Комиссии** | Bybit берёт maker‑0.025 %/taker‑0.075 % (USDT‑perp). При high‑frequency стратегиях учитывайте *fee‑adjusted* P&L. |
| **Slippage** | При market‑orders > 0.1 BTC в BTC‑USDT может быть > 0.1 %. Предпочтительно использовать лимит‑ордера с `postOnly`. |
| **Тестовая среда** | Bybit предлагает **Testnet** – используйте её на этапе разработки, чтобы не рисковать реальными средствами. |
| **Логи** | Храните логи минимум 30 дней, они потребуются при спорных тикетах с биржей. |
| **Управление ключами** | Не храните API‑ключи в репозитории. Используйте переменные окружения, Secrets Manager. |
| **Ограничения по Leverage** | Устанавливайте **max_leverage** ниже официального лимита (например 10×) – снижает риск мгновенного ликвидирования. |
| **Регуляции** | Если бот будет использоваться клиентами, проверьте необходимость лицензии на управление активами (MiFID, FINRA и т.д.). |
| **Fail‑over** | При падении WebSocket автоматически переключайтесь на резервный endpoint (`wss://stream.bybit.com/realtime` → `wss://stream-realtime.bybit.com/realtime`) . |
| **Тестирование на экстремальных сценариях** | Симулируйте «flash‑crash», резкое изменение funding‑rate, отключение API. |

---  

## 8. Пример кода (скелет)  

```python
# file: bot/core.py
import asyncio
import logging
from bybit import bybit   # pip install pybybit
import pandas as pd
import talib as ta

log = logging.getLogger('bybit_bot')
client = bybit(test=False, api_key='YOUR_KEY', api_secret='YOUR_SECRET')

# --------------------------------------------------------------------
# 1. Data ingestion (WebSocket)
# --------------------------------------------------------------------
class WSListener:
    URL = "wss://stream.bybit.com/realtime"

    async def connect(self):
        async with websockets.connect(self.URL) as ws:
            await ws.send(json.dumps({"op":"subscribe","args":["trade.BTCUSDT"]}))
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                # put raw trade into asyncio.Queue for downstream processing
                await trade_queue.put(data)

# --------------------------------------------------------------------
# 2. Signal Engine
# --------------------------------------------------------------------
class SignalEngine:
    def __init__(self):
        self.buffer = pd.DataFrame(columns=['price','ts'])

    async def run(self):
        while True:
            trade = await trade_queue.get()
            self.buffer = self.buffer.append(
                {'price': float(trade['data'][0]['price']),
                 'ts': pd.to_datetime(trade['data'][0]['timestamp'], unit='ms')},
                ignore_index=True)
            # keep only last 500 rows
            self.buffer = self.buffer.tail(500)

            if len(self.buffer) >= 30:
                signal = self.generate_signal()
                if signal:
                    await signal_queue.put(signal)

    def generate_signal(self):
        close = self.buffer['price']
        ema20 = ta.EMA(close, timeperiod=20)
        ema50 = ta.EMA(close, timeperiod=50)
        if ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-2] <= ema50.iloc[-2]:
            return {"side":"Buy","price":close.iloc[-1]}
        if ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-2] >= ema50.iloc[-2]:
            return {"side":"Sell","price":close.iloc[-1]}
        return None

# --------------------------------------------------------------------
# 3. Order Executor
# --------------------------------------------------------------------
class OrderExecutor:
    async def run(self):
        while True:
            sig = await signal_queue.get()
            qty = self.calculate_qty(sig)
            order = client.Order.Order_new(
                side=sig['side'],
                symbol="BTCUSDT",
                order_type="Limit",
                qty=qty,
                price=sig['price'],
                time_in_force="GoodTillCancel",
                reduce_only=False,
                close_on_trigger=False).result()
            log.info(f"Placed {sig['side']} {qty}@{sig['price']} => {order}")

    def calculate_qty(self, sig):
        # simple 2% of equity risk, stop‑loss 1%
        equity = float(client.Wallet.Wallet_getBalance(coin="USDT").result()[0]['result'][0]['wallet_balance'])
        risk = equity * 0.02
        stop = 0.01 * sig['price']
        qty = risk / stop
        return round(qty, 3)

# --------------------------------------------------------------------
# 4. Main loop
# --------------------------------------------------------------------
trade_queue = asyncio.Queue()
signal_queue = asyncio.Queue()

async def main():
    ws = WSListener()
    signal = SignalEngine()
    executor = OrderExecutor()

    await asyncio.gather(
        ws.connect(),
        signal.run(),
        executor.run()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

> **[AI]** – Этот кусок кода можно доработать: добавить обработку ошибок, использовать `aiohttp`‑клиент, внедрить динамический `max_leverage`, подключить `Prometheus`‑метрики.  

---  

## 9. Итоги  

* **Стратегии**: от простых EMA‑пересечений до ML‑моделей – All‑in‑One платформа.  
* **Сигналы**: технические индикаторы, order‑book imbalance, funding‑rate, открытый интерес, новостной сентимент.  
* **Риски**: API‑лимиты, маржинальные требования, проскальзывание, регуляторные ограничения – всё покрывается Risk‑Engine.  
* **Roadmap**: 8 чётко определённых фаз, где ИИ помогает генерировать код, проводить оптимизацию и писать тесты, а человек контролирует безопасность, комплаенс и финальные проверки.  

Следуя этому плану, вы получите надёжного, масштабируемого и полностью автоматизированного криптобота, способного торговать фьючерсами на Bybit 24/7, при этом сохраняя строгий контроль над рисками и соответствием нормативным требованиям.  

---  

*Удачной разработки! Если понадобится конкретный код‑фрагмент, пример оптимизации стратегии или помощь в настройке CI/CD – дайте знать, и я подготовлю детальные материалы.*
