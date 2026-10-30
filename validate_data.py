
import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent / "data" / "instamart.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

checks = [
    ("Customers table exists",
 "SELECT * FROM customers LIMIT 0"),

("Products table exists",
 "SELECT * FROM products LIMIT 0"),

("Orders table exists",
 "SELECT * FROM orders LIMIT 0"),

    ("No duplicate customer IDs",
     "SELECT customer_id FROM customers GROUP BY customer_id HAVING COUNT(*) > 1"),

    ("No duplicate product IDs",
     "SELECT product_id FROM products GROUP BY product_id HAVING COUNT(*) > 1"),

    ("No duplicate order IDs",
     "SELECT order_id FROM orders GROUP BY order_id HAVING COUNT(*) > 1"),

    ("No missing order amounts",
     "SELECT order_id FROM orders WHERE order_amount IS NULL"),

    ("No negative order amounts",
     "SELECT order_id FROM orders WHERE order_amount < 0"),

    ("Valid order statuses",
     "SELECT order_id FROM orders WHERE status NOT IN ('completed', 'cancelled')")
]

all_passed = True

for name, query in checks:
    cursor.execute(query)
    result = cursor.fetchall()

    if len(result) == 0:
        print("PASS:", name)
    else:
        print("FAIL:", name, "| Issues:", len(result))
        all_passed = False

conn.close()

from pathlib import Path

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

result_text = "ALL CHECKPOINTS PASSED" if all_passed else "SOME CHECKPOINTS FAILED"

with open(output_dir / "validation.txt", "w") as file:
    file.write(result_text)

print("\n" + result_text)
print("Validation result saved in outputs/validation.txt")