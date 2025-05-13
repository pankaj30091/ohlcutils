from ohlcutils.data import load_symbol
from ohlcutils.enums import Periodicity

md = load_symbol(
    "INFY_STK___",
    days=100,
    src=Periodicity.DAILY,
    dest_bar_size="1W",
    label="left",
    adjust_for_holidays=True,
    adjustment="fbd",
)
print(md)
