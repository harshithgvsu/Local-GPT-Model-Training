Top N by sum:
SELECT customer_id, SUM(amount) AS total
FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 10;
