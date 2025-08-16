from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BybitConfig:
    api_key: str
    api_secret: str
    testnet: bool = False
    demo: bool = True
    ignore_ssl: bool = True
    max_positions: int = 3
    max_risk_per_trade: float = 2.0
    min_trade_interval_minutes: int = 5
    strategy_cooldown_minutes: int = 30
    balance_update_interval_minutes: int = 1
    min_trade_amount: float = 10
    max_trade_amount: float = 1000
    min_leverage: int = 1
    max_leverage: int = 20

    @classmethod
    def from_env(cls) -> 'BybitConfig':
        return cls(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=os.getenv("BYBIT_TESTNET", "False").lower() == "true",
            demo=os.getenv("BYBIT_DEMO", "True").lower() == "true",
            ignore_ssl=os.getenv("BYBIT_IGNORE_SSL", "True").lower() == "true",
            max_positions=int(os.getenv("BYBIT_MAX_POSITIONS", "3")),
            max_risk_per_trade=float(os.getenv("BYBIT_MAX_RISK_PER_TRADE", "2.0")),
            min_trade_interval_minutes=int(os.getenv("BYBIT_MIN_TRADE_INTERVAL_MINUTES", "5")),
            strategy_cooldown_minutes=int(os.getenv("BYBIT_STRATEGY_COOLDOWN_MINUTES", "30")),
            balance_update_interval_minutes=int(os.getenv("BYBIT_BALANCE_UPDATE_INTERVAL_MINUTES", "1")),
            min_trade_amount=float(os.getenv("BYBIT_MIN_TRADE_AMOUNT", "10")),
            max_trade_amount=float(os.getenv("BYBIT_MAX_TRADE_AMOUNT", "1000")),
            min_leverage=int(os.getenv("BYBIT_MIN_LEVERAGE", "1")),
            max_leverage=int(os.getenv("BYBIT_MAX_LEVERAGE", "20")),
        )
