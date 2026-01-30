from dataclasses import dataclass
from datetime import datetime

@dataclass
class PositionInfo:
    symbol: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    position_type: str
    quantity: float
    
@dataclass
class RiskMetrics:
    max_drawdown: float
    total_pnl: float
    trade_count: int
    win_rate: float
