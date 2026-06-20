"""Combine UNSW day1 + day2 NFStream parquets into a single corpus."""
import logging, pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

day1 = Path(r"X:\ADNS\outputs\corpus\unsw_day1_nfstream.parquet")
day2 = Path(r"X:\ADNS\outputs\corpus\unsw_day2_nfstream.parquet")
out  = Path(r"X:\ADNS\outputs\corpus\unsw_flows_nfstream.parquet")

if not day1.exists():
    raise FileNotFoundError(f"Missing: {day1}")
if not day2.exists():
    raise FileNotFoundError(f"Missing: {day2}")

df1 = pd.read_parquet(day1)
df2 = pd.read_parquet(day2)
log.info("day1: %d rows (attack=%d)", len(df1), int((df1['label']==1).sum()))
log.info("day2: %d rows (attack=%d)", len(df2), int((df2['label']==1).sum()))

df = (
    pd.concat([df1, df2])
    .sort_values(["ts", "src_ip", "src_port", "dst_ip", "dst_port"])
    .reset_index(drop=True)
)
att = int((df['label']==1).sum())
log.info("combined: %d rows (attack=%d / %.2f%%)", len(df), att, 100*att/max(len(df),1))
df.to_parquet(out, index=False)
log.info("Wrote %s", out)
