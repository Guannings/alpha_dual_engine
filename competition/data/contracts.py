"""
TAIFEX contract specifications, expiry calendars, and rollover logic.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional
import calendar


@dataclass
class ContractSpec:
    """Specification for a single TAIFEX contract type."""
    symbol: str
    name: str
    category: str             # Maps to competition product category
    multiplier: float         # NT$ per index point (or per unit)
    tick_size: float          # Minimum price movement
    tick_value: float         # NT$ value per tick
    margin_initial: float     # Initial margin requirement (NT$)
    margin_maintenance: float # Maintenance margin requirement (NT$)
    trading_hours: str        # e.g. "08:45-13:45"
    settlement: str           # "cash" or "physical"
    months_listed: int        # Number of contract months available


# Core contract specifications (approximate values — verify with TAIFEX before trading)
CONTRACT_SPECS: Dict[str, ContractSpec] = {
    'TX': ContractSpec(
        symbol='TX', name='TAIEX Futures', category='index_futures',
        multiplier=200.0, tick_size=1.0, tick_value=200.0,
        margin_initial=184_000, margin_maintenance=141_000,
        trading_hours='08:45-13:45', settlement='cash', months_listed=5,
    ),
    'MTX': ContractSpec(
        symbol='MTX', name='Mini-TAIEX Futures', category='index_futures',
        multiplier=50.0, tick_size=1.0, tick_value=50.0,
        margin_initial=46_000, margin_maintenance=35_250,
        trading_hours='08:45-13:45', settlement='cash', months_listed=5,
    ),
    'TE': ContractSpec(
        symbol='TE', name='Electronics Sector Futures', category='index_futures',
        multiplier=4000.0, tick_size=0.05, tick_value=200.0,
        margin_initial=150_000, margin_maintenance=115_000,
        trading_hours='08:45-13:45', settlement='cash', months_listed=3,
    ),
    'TF': ContractSpec(
        symbol='TF', name='Finance Sector Futures', category='index_futures',
        multiplier=1000.0, tick_size=0.2, tick_value=200.0,
        margin_initial=67_000, margin_maintenance=51_500,
        trading_hours='08:45-13:45', settlement='cash', months_listed=3,
    ),
    'TXO': ContractSpec(
        symbol='TXO', name='TAIEX Options', category='index_options',
        multiplier=50.0, tick_size=0.1, tick_value=5.0,
        margin_initial=0, margin_maintenance=0,  # Buyer: premium only; seller: margin
        trading_hours='08:45-13:45', settlement='cash', months_listed=6,
    ),
    'GDF': ContractSpec(
        symbol='GDF', name='Gold Futures', category='gold_futures',
        multiplier=10.0, tick_size=0.1, tick_value=1.0,  # 10 grams per contract
        margin_initial=55_000, margin_maintenance=42_000,
        trading_hours='08:45-13:45', settlement='cash', months_listed=6,
    ),
    'UDF': ContractSpec(
        symbol='UDF', name='USD/TWD Futures', category='fx_futures',
        multiplier=50_000.0, tick_size=0.001, tick_value=50.0,
        margin_initial=35_000, margin_maintenance=27_000,
        trading_hours='08:45-13:45', settlement='cash', months_listed=6,
    ),
    'XIF': ContractSpec(
        symbol='XIF', name='Mid-Cap 100 Futures', category='midcap_futures',
        multiplier=100.0, tick_size=1.0, tick_value=100.0,
        margin_initial=60_000, margin_maintenance=46_000,
        trading_hours='08:45-13:45', settlement='cash', months_listed=3,
    ),
}


def get_third_wednesday(year: int, month: int) -> date:
    """Get the third Wednesday of a given month (TAIFEX standard expiry)."""
    cal = calendar.Calendar()
    wednesdays = [
        d for d in cal.itermonthdays2(year, month)
        if d[0] != 0 and d[1] == 2  # weekday 2 = Wednesday
    ]
    third_wed = wednesdays[2][0]
    return date(year, month, third_wed)


def get_expiry_dates(year: int) -> List[date]:
    """Get all monthly expiry dates for a given year."""
    return [get_third_wednesday(year, m) for m in range(1, 13)]


def get_front_month_expiry(as_of: date) -> date:
    """Get the nearest expiry date that hasn't passed yet."""
    year = as_of.year
    for month in range(as_of.month, 13):
        exp = get_third_wednesday(year, month)
        if exp > as_of:
            return exp
    return get_third_wednesday(year + 1, 1)


def get_back_month_expiry(as_of: date) -> date:
    """Get the second-nearest expiry date."""
    front = get_front_month_expiry(as_of)
    next_month = front.month + 1
    next_year = front.year
    if next_month > 12:
        next_month = 1
        next_year += 1
    return get_third_wednesday(next_year, next_month)


def should_roll(as_of: date, days_before_expiry: int = 3) -> bool:
    """Check if we should roll to the next contract month."""
    front_expiry = get_front_month_expiry(as_of)
    days_to_expiry = (front_expiry - as_of).days
    return days_to_expiry <= days_before_expiry


def get_contract_month_code(d: date) -> str:
    """Convert a date to TAIFEX contract month code (e.g., '202604')."""
    return d.strftime('%Y%m')


def get_margin_requirement(symbol: str, num_contracts: int) -> float:
    """Calculate total initial margin for a position."""
    spec = CONTRACT_SPECS.get(symbol)
    if spec is None:
        raise ValueError(f"Unknown contract: {symbol}")
    return spec.margin_initial * abs(num_contracts)


def get_notional_value(symbol: str, price: float, num_contracts: int) -> float:
    """Calculate notional value of a position."""
    spec = CONTRACT_SPECS.get(symbol)
    if spec is None:
        raise ValueError(f"Unknown contract: {symbol}")
    return spec.multiplier * price * abs(num_contracts)
