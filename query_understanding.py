import json
import re
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
# USE THIS MODEL IF THE MEMORY CRASHES
# MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16
)

def understand_query(query: str) -> dict:
    
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = f"""
        You are a SQL query intent parser for a business analytics pipeline. 
        Today's date is {today_str}. Resolve all relative dates (last month, last 30 days, etc.) based on this date.

        <schema>
        Output ONLY a raw JSON object matching this exact schema:
        {{
            "intent": "str", // SELECT | COUNT | AGGREGATE | TOP_N | SELECT_SUBQUERY
            "temporal_filter": {{
                "expression": "str", 
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD"
            }} | null,
            "entities": ["str"], // Business concepts (e.g., ["fraud", "transaction_type"])
            "complexity": "str", // simple | medium | complex
            "needs_clarification": bool,
            "clarification_reason": "str" | null
        }}
        </schema>

        <custom_rules>
        1. DOMAIN CHECK: You ONLY process queries about business data (transactions, customers, sales, fraud). IF the user asks a general knowledge question (e.g., "What is money?"), makes small talk, or asks something outside this domain, you MUST set "needs_clarification": true.
        2. AMBIGUITY: IF the query is incomplete, vague, or fewer than 3 words (except "total sales"), set "needs_clarification": true.
        3. OUTPUT: Output raw JSON only. Do NOT wrap in ```json tags.
        </custom_rules>

        <definitions>
        Complexity:
        - simple: Direct SELECT, COUNT, or SUM with basic filters.
        - medium: Multiple conditions, GROUP BY (e.g., 'by', 'per').
        - complex: NOT EXISTS (e.g., 'no transactions'), window functions.

        Intent:
        - SELECT: Show/list items.
        - COUNT: How many.
        - AGGREGATE: Total/sum (especially when grouped 'by' something).
        - TOP_N: Top/highest/lowest items.
        - SELECT_SUBQUERY: Complex exclusions like 'no transactions'.
        </definitions>

        <few_shot_examples>

        User: "Total fraud amount by transaction type last quarter"
        Output:
        {{
            "intent": "AGGREGATE",
            "temporal_filter": {{
                "expression": "last quarter",
                "start_date": "2024-01-01",
                "end_date": "2024-03-31"
            }},
            "entities": ["fraud", "transaction_type"],
            "complexity": "medium",
            "needs_clarification": false,
            "clarification_reason": null
        }}

        User: "Customers with no transactions in last 30 days"
        Output:
        {{
            "intent": "SELECT_SUBQUERY",
            "temporal_filter": {{
                "expression": "last 30 days",
                "start_date": "2024-04-20",
                "end_date": "2024-05-20"
            }},
            "entities": ["customers", "transactions"],
            "complexity": "complex",
            "needs_clarification": false,
            "clarification_reason": null
        }}

        User: "What is money"
        Output:
        {{
            "intent": "SELECT",
            "temporal_filter": null,
            "entities": [],
            "complexity": "simple",
            "needs_clarification": true,
            "clarification_reason": "This pipeline only processes queries about business metrics and transactions. I cannot answer general knowledge questions."
        }}

        User: "compare the two things"
        Output:
        {{
            "intent": "SELECT",
            "temporal_filter": null,
            "entities": [],
            "complexity": "simple",
            "needs_clarification": true,
            "clarification_reason": "Please provide more information."
        }}

        </few_shot_examples>
        """
        
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs, 
                max_new_tokens=256,
                temperature=0.1, 
                do_sample=False
            )
            
        input_length = model_inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        json_string = response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_string = match.group(0)
            
        result_dict = json.loads(json_string)
        
        keys = ["intent", "temporal_filter", "entities", "complexity", "needs_clarification", "clarification_reason"]
        for key in keys:
            if key not in result_dict:
                raise ValueError(f"Missing key in LLM output: {key}")
                
        return result_dict

    except Exception as e:
        return fallback(f"System processing error: {str(e)}")


def fallback(reason: str) -> dict:
    return {
        "intent": "SELECT",
        "temporal_filter": None,
        "entities": [],
        "complexity": "simple",
        "needs_clarification": True,
        "clarification_reason": reason
    }