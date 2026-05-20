from query_understanding import understand_query
# test_queries = [
#     "List all customers in the European region",
#     "Display the transaction history for the fraud department",
#     "Total fraud amount by transaction type last quarter",
#     "What is the sum of all international sales this year",
#     "Show total revenue per customer segment",
#     "Total refunds processed by the system last week",
#     "Count the number of new users registered in the last 30 days",
#     "How many active accounts do we currently have",    
#     "What are the top 5 transaction types by volume this year",
#     "Find the 3 regions with the highest fraud rates last month",
#     "Show accounts without any login activity this year",
#     "Users who have not made a purchase in the last quarter",
#     "total sales",               
#     "compare the two things",    
#     "What is money?",            
#     "How to hack a bank account", 
#     "How many transactions last month?",
#     "Show all high-value transactions above 50,000",
#     "Top 10 customers by total transaction amount",
#     "Customers with no transactions in the last 30 days"
# ]

test_queries = [
    ""
]
for i, q in enumerate(test_queries):
    print(f"--- Test {i+1} ---")
    print(f"Query: {q}")
    result = understand_query(q)
    print(f"Result: {result}\n")
