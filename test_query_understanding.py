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

# test_queries = [
    # "Which projects have the highest number of listed assets for lease vs sale?",
    # "What is the distribution of assets per project and asset type?",
    # "How many lease listings and sale listings exist for each project?",
    # "Which project has the highest occupancy rate?",
    # "What is the average carpet area of assets per project?",
    # "Which assets currently have active tenants and which are vacant?",
    # "What is the tenant distribution across different projects?",
    # "What is the average lease duration per asset type?",
    # "Which projects have the highest number of tenants?",
    # "How many lease transactions occurred per project?",
    # "Which amenities are most common across assets?",
    # "What is the distribution of amenities across projects?",
    # "Which projects provide the highest number of amenities?",
    # "What percentage of assets have premium amenities?",
    # "Which assets have the highest number of sale negotiations?",
    # "What is the conversion rate from sale listing → sale transaction?",
    # "What is the average negotiation duration before a sale transaction?",
    # "Which projects generate the highest sale value?",
    # "What is the total payment received per project?",
    # "What is the payment trend for lease transactions over time?",
    # "Which tenants contribute the highest rental revenue?",
    # "What is the payment method distribution for lease payments?",
    # "What percentage of assets have completed verification documents?",
    # "Which document types are most commonly submitted?",
    # "Which projects have the highest number of verified assets?",
    # "What is the conversion rate from listing leads to tenants?",
    # "Which projects generate the most leads?",
    # "What is the average time taken to convert a lead to a tenant?",
    # "Which asset type generates the highest revenue?",
    # "What is the distribution of asset types across projects?",
    # "What is the average lease value per asset type?"
# ]

test_queries = [
    "Hello"
]
for i, q in enumerate(test_queries):
    print(f"--- Test {i+1} ---")
    print(f"Query: {q}")
    result = understand_query(q)
    print(f"Result: {result}\n")
