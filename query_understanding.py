import json
import re
import time
import threading
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar

class QuerySession:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-3B-Instruct", timeout_seconds=300):
        # USE THIS MODEL IF THE MEMORY CRASHES
        # MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.last_activity = time.time()
        self.is_active = True
        
        print(f"Loading {self.model_name}... this might take a sec.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.float16
        )
        print("Model loaded successfully!")
        
        self.monitor_thread = threading.Thread(target=self._monitor_inactivity, daemon=True)
        self.monitor_thread.start()

    def _monitor_inactivity(self):
        while self.is_active:
            time.sleep(5)  
            time_inactive = time.time() - self.last_activity
            
            if time_inactive > self.timeout_seconds:
                print(f"\n[System] No activity for {self.timeout_seconds} seconds. Auto-killing session to free VRAM...")
                self.kill_session()
                break

    def kill_session(self):
        if not self.is_active:
            return
            
        self.is_active = False
        print("\nShutting down session and freeing up resources...")
        
        del self.model
        del self.tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("VRAM cleared. Session terminated.")

    def calculate_temporal_filter(self, temporal_params: dict) -> dict | None:
        if not temporal_params:
            return None
            
        now = datetime.now()
        calc_type = temporal_params.get("type")
        unit = temporal_params.get("unit")
        value = temporal_params.get("value", 1)
        
        start_date, end_date = now, now

        if calc_type == "rolling":
            kwargs = {unit: value}
            start_date = now - relativedelta(**kwargs)
            end_date = now

        
        elif calc_type in ["calendar_last", "calendar_current"]:
            offset = value if calc_type == "calendar_last" else 0
            target_date = now - relativedelta(**{unit: offset})
            
            calendar_bounds = {
                "years": lambda d: (
                    datetime(d.year, 1, 1), 
                    datetime(d.year, 12, 31)
                ),
                "months": lambda d: (
                    datetime(d.year, d.month, 1), 
                    datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
                ),
                "quarters": lambda d: (
                    datetime(d.year, 3 * ((d.month - 1) // 3) + 1, 1),
                    datetime(d.year, 3 * ((d.month - 1) // 3) + 3, calendar.monthrange(d.year, 3 * ((d.month - 1) // 3) + 3)[1])
                ),
                "weeks": lambda d: (
                    d - timedelta(days=d.weekday()), 
                    d + timedelta(days=6 - d.weekday()) 
                )
            }
            
            if unit in calendar_bounds:
                start_date, end_date = calendar_bounds[unit](target_date)

        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }   

    def understand_query(self, query: str) -> dict:
        if not self.is_active:
            return self._fallback("Session has been terminated. Please start a new session to query.")
            
        self.last_activity = time.time()
        
        try:
            # today_str = datetime.now().strftime("%Y-%m-%d")
            
            sys_prompt = f"""
            You are a SQL query intent parser for a business analytics pipeline. 

            <schema>
            Output ONLY a raw JSON object matching this exact schema:
            {{
                "intent": "str", // SELECT | COUNT | AGGREGATE | TOP_N | SELECT_SUBQUERY
                "temporal_filter": {{
                    "expression": "str", // The exact time phrase extracted from the query (e.g., "last 30 days")
                    "type": "str", // rolling | calendar_last | calendar_current
                    "unit": "str", // days | weeks | months | quarters | years
                    "value": int // The integer value (e.g., 30 for "last 30 days", 1 for "last month")
                }} | null, // Use null if no time period is mentioned
                "entities": ["str"], // Business concepts (e.g., ["fraud", "transaction_type"])
                "complexity": "str", // simple | medium | complex
                "needs_clarification": bool,
                "clarification_reason": "str" | null
            }}
            </schema>

            <custom_rules>
            1. DOMAIN CHECK: You ONLY process queries about business data (transactions, customers, sales, fraud, etc.). IF the user asks a general knowledge question (e.g., "What is money?"), makes small talk, or asks something outside this domain, you MUST set "needs_clarification": true.
            2. SPELL AND GRAMMAR CHECK: If the query has simple spelling and grammar errors. Make REASONABLE correction BUT if the errors are major, set "needs_clarification": true. 
            3. AMBIGUITY: IF the query is incomplete, vague, or fewer than 3 words (except "total sales"), set "needs_clarification": true.
            4. OUTPUT: Output raw JSON only. Do NOT wrap in ```json tags.
            </custom_rules>

            <definitions>
            Complexity:
            - simple: 1 datafield accessed, OR Direct SELECT, COUNT, or SUM with basic filters.
            - medium: 2 datafields accessed, OR Multiple conditions, GROUP BY (e.g., 'by', 'per').
            - complex: 3 or more or all datafields accessed, OR NOT EXISTS (e.g., 'no transactions'), window functions, OR includes a MIX of both the simple and medium complexity.
            (Note: If a query meets a higher complexity rule, default to the higher complexity).

            Intent:
            - SELECT: Show/list items.
            - COUNT: How many, What.
            - AGGREGATE: Total/sum (especially when grouped 'by' something or 'per' field).
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
                    "type": "calendar_last",
                    "unit": "quarters",
                    "value": 1
                }},
                "entities": ["fraud", "transaction_type"],
                "complexity": "medium",
                "needs_clarification": false,
                "clarification_reason": null
            }}

            User: "Totl fraud amunt by transaction type lost quarter"
            Output:
            {{
                "intent": "AGGREGATE",
                "temporal_filter": {{
                    "expression": "last quarter",
                    "type": "calendar_last",
                    "unit": "quarters",
                    "value": 1
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
                    "type": "rolling",
                    "unit": "days",
                    "value": 30
                }},
                "entities": ["customers", "transactions"],
                "complexity": "complex",
                "needs_clarification": false,
                "clarification_reason": null
            }}

            User: "Show all high value transactions above 50000"
            Output:
            {{
                "intent": "SELECT",
                "temporal_filter": null,
                "entities": ["transactions"],
                "complexity": "simple",
                "needs_clarification": false,
                "clarification_reason": null
            }}

            User: "List total revenue, customer count, and average order value by region"
            Output:
            {{
                "intent": "AGGREGATE",
                "temporal_filter": null,
                "entities": ["revenue", "customer count", "average order value", "region"],
                "complexity": "complex",
                "needs_clarification": false,
                "clarification_reason": null
            }}

            User: "How many customer are there in per project by region?"
            Output:
            {{
                "intent": "COUNT",
                "temporal_filter": null,
                "entities": ["customer", "project", "region"],
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

            </few_shot_examples>
            """
            
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs, 
                    max_new_tokens=256,
                    do_sample=False
                )
                
            input_length = model_inputs.input_ids.shape[1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            json_string = response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_string = match.group(0)
                
            result_dict = json.loads(json_string)
            temporal_params = result_dict.get("temporal_filter")

            if temporal_params:
                result_dict["temporal_filter"] = self.calculate_temporal_filter(temporal_params)

            keys = ["intent", "temporal_filter", "entities", "complexity", "needs_clarification", "clarification_reason"]
            for key in keys:
                if key not in result_dict:
                    raise ValueError(f"Missing key in LLM output: {key}")
                    
            return result_dict

        except Exception as e:
            return self.fallback(f"System processing error: {str(e)}")


    def fallback(self, reason: str) -> dict:
        return {
            "intent": "SELECT",
            "temporal_filter": None,
            "entities": [],
            "complexity": "simple",
            "needs_clarification": True,
            "clarification_reason": reason
        }
        
    
if __name__ == "__main__":
    session = QuerySession(timeout_seconds=600)
    
    print("Type 'exit' or 'kill' to manually end the session.")
    print("Session will auto-kill after 10 minutes of inactivity.")
    print("\n--- Welcome to VEDA ---")
    
    try:
        while session.is_active:
            user_input = input("\nHow may I help you?: ")
            
            if user_input.lower() in ['exit', 'kill', 'quit']:
                session.kill_session()
                break
                
            if user_input.strip() and session.is_active:
                result = session.understand_query(user_input)
                print(json.dumps(result, indent=2))
                
    except KeyboardInterrupt:
        session.kill_session()