
import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path(__file__).resolve().parent / "data" / "instamart.db"
conn = sqlite3.connect(db_path)

query = """
SELECT
    DATE(order_ts) AS order_date,
    ROUND(SUM(order_amount), 2) AS daily_revenue,
    ROUND(
        SUM(SUM(order_amount)) OVER (
            ORDER BY DATE(order_ts)
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS running_total
FROM orders
WHERE status = 'completed'
GROUP BY DATE(order_ts)
ORDER BY order_date;
"""

result = pd.read_sql_query(query, conn)
print(result.head(10))
print("Total rows:", len(result))

conn.close()
