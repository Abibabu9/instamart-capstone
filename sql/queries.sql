-- Query 1: Customers with high completed-order spending

SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Query 2: Revenue by city and product category

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, total_revenue DESC;

-- Query 3: Products priced above average

SELECT product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- Query 4: Customers with no completed orders

SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Query 5: Customers spending above average

WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
)
SELECT *
FROM customer_spend
WHERE total_spend > (SELECT AVG(total_spend) FROM customer_spend)
ORDER BY total_spend DESC;


-- Query 6: Customer spending rank

SELECT
    customer_id,
    SUM(order_amount) AS total_spend,
    RANK() OVER (ORDER BY SUM(order_amount) DESC) AS spend_rank,
    DENSE_RANK() OVER (ORDER BY SUM(order_amount) DESC) AS dense_rank,
    ROW_NUMBER() OVER (ORDER BY SUM(order_amount) DESC) AS row_num
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
ORDER BY total_spend DESC;

-- Query 7: Daily revenue and running total

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
