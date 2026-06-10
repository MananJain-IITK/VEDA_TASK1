import json
import re
import time
import threading
import torch
import gc
import csv
import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import TypedDict, List, Literal

import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

# State Definition
class AgentState(TypedDict):
    user_query: str
    retrieved_schema: str
    raw_response: str
    final_json: dict
    error_message: str
    retry_count: int

# Main Agent Class
class LangGraphAgent:
    def __init__(self, timeout_seconds=300):
        self.timeout_seconds = timeout_seconds
        self.last_activity = time.time()
        self.is_active = True

        # Vector Model
        print("Loading BAAI")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

        # Query SLM Model
        print("Loading Qwen 3B")
        model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto"
        )

        # ChromaDB
        print("Initializing ChromaDB")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="homzhub_schema")

        # VRAM Monitor
        self.monitor_thread = threading.Thread(target=self.monitor_inactivity, daemon=True)
        self.monitor_thread.start()

        # Compile Graph
        print("Compiling VEDA Langgraph")
        self.graph = self._build_graph()
        print("System Ready!\n")


    def monitor_inactivity(self):
        while self.is_active:
            time.sleep(5)
            time_inactive = time.time() - self.last_activity
            if time_inactive > self.timeout_seconds:
                print(f"\nNo activity for {self.timeout_seconds}s. Auto-killing session.")
                self.kill_session()
                break

    def kill_session(self):
        if not self.is_active:
            return
        self.is_active = False
        print("\nShutting down session and freeing up resources")

        del self.llm
        del self.tokenizer
        del self.embedder

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VRAM cleared. Session terminated.")

    #Database/RAG Methods
    def build_vector_database(self, csv_filepath="./columns_homzhub.csv"):
        if self.collection.count() > 0:
            print("Database already contains schema, skipping build.")
            return

        documents = []
        metadata = []
        ids = []

        
        with open(csv_filepath, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if not row or len(row) < 3:
                    continue
                table  = row[0].strip()
                column = row[1].strip()
                dtype  = row[2].strip()
                description = f"Table: {table}. Column: {column}. Dtype: {dtype}."

                documents.append(description)
                metadata.append({"table": table, "column": column})
                ids.append(f"row_{idx}")

        if not documents:
            print("WARNING: No valid rows found in CSV. Database not built.")
            return

        print("Calculating Embeddings and Saving to ChromaDB")
        embeddings = self.embedder.encode(documents).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids,
        )
        print("Database Built Successfully!\n")

    #Temporal Engine
    def calculate_temporal_filter(self, temporal_params: dict) -> dict | None:
        if not temporal_params:
            return None
        
        calc_type = temporal_params.get("type")
        if calc_type == "explicit_range":
            return {
                "start_date" : temporal_params.get("start_date"),
                "end_date" : temporal_params.get("end_date"),
            }
        
        # print(temporal_params)

        now = datetime.now()
        unit = temporal_params.get("unit")
        value = int(temporal_params.get("value", 1))

        

        start_date = now
        end_date = now
        
        if calc_type == "rolling":
            if unit == "quarters":
                kwargs = {"months": value * 3}
            else:
                kwargs = {unit: value}
                
            start_date = now - relativedelta(**kwargs)
            end_date = now

        elif calc_type in ["calendar_last", "calendar_current"]:
            if calc_type == "calendar_last":
                start_offset = value  
                end_offset = 1        
            else: 
                start_offset = 0
                end_offset = 0
                
            
            start_kwargs = {"months": start_offset * 3} if unit == "quarters" else {unit: start_offset}
            end_kwargs   = {"months": end_offset * 3}   if unit == "quarters" else {unit: end_offset}
                
            target_start_date = now - relativedelta(**start_kwargs)
            target_end_date   = now - relativedelta(**end_kwargs)
            
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
                start_date = calendar_bounds[unit](target_start_date)[0]
                end_date   = calendar_bounds[unit](target_end_date)[1]

        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date":   end_date.strftime("%Y-%m-%d"),
        }

    #LangGraph Nodes
    def node_retrieve_schema(self, state: AgentState) -> dict:
        print("Retrieving Schema")
        self.last_activity = time.time()
        query = state["user_query"]

        query_vector = self.embedder.encode(query).tolist()

        
        collection_size = self.collection.count()
        if collection_size == 0:
            raise RuntimeError(
                "ChromaDB collection is empty. Run build_vector_database() first."
            )

        n_results = min(15, collection_size)
        results   = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
        )

       
        documents = results.get("documents", [[]])[0]
        if not documents:
            return {
                "retrieved_schema": "",
                "error_message": "Schema retrieval returned no results for this query.",
                "retry_count": state.get("retry_count", 0),
            }

        schema_context = "\n".join(documents)
        return {
            "retrieved_schema": schema_context,
            "error_message":    "",
            "retry_count":      state.get("retry_count", 0),
        }

    def node_generate_json(self, state: AgentState) -> dict:
        # print(state['retrieved_schema'])
        self.last_activity = time.time()

        error_feedback = ""
        if state.get("error_message"):
            error_feedback = (
                f"\n<error_feedback>\n"
                f"Previous attempt failed with error: {state['error_message']}. "
                f"Please fix the JSON structure and keys.\n"
                f"</error_feedback>"
            )

        sys_prompt = f"""
        You are a SQL query intent parser for a business analytics pipeline.

        <database_schema_context>
        Relevant schema objects identified for this query:
        {state['retrieved_schema']}
        </database_schema_context>

        <schema>
        Output ONLY a raw JSON object matching this exact schema:
        {{
            "intent": "str", // SELECT | COUNT | AGGREGATE | TOP_N | SELECT_SUBQUERY
            "temporal_filter": {{
                "expression": "str",
                "type": "str",   // ONLY CHOOSE FROM THIS: rolling | calendar_last | calendar_current | explicit_range
                "unit": "str",   // days | weeks | months | quarters | years
                "value": int,    // N for rolling (e.g. 30). ALWAYS 1 for calendar types.
                "start_date": "str" | null, // Use YYYY-MM-DD format if explicit_range
                "end_date": "str" | null    // Use YYYY-MM-DD format if explicit_range
            }} | null,
            "entities": ["str"], // Refer to "entities_rules" in this system prompt 
            "complexity": "str", // simple | medium | complex
            "needs_clarification": bool,
            "clarification_reason": "str" | null // Polite and encouraging if true
        }}
        </schema>

        <entities_rules>
        1. The entities should be STRICTLY from the "database_schema_context".
        2. Output should be in the form "table: column".
        3. If no table/column makes semantic sense with respect to the query, set "needs_clarification": true, and give a clarification_reason.
        </entities_rules>

        <custom_rules>
            1. DOMAIN CHECK: You ONLY process queries about business data (transactions, customers, sales, fraud, etc.). IF the user asks a general knowledge question (e.g., "What is money?"), makes small talk, or asks something outside this domain, you MUST set "needs_clarification": true.
            2. AMBIGUITY: IF the query is incomplete, vague, or fewer than 3 words (except "total sales"), set "needs_clarification": true.
            3. OUTPUT: Output raw JSON only. Do NOT wrap in ```json tags.
            4. TEMPORAL STRICTNESS: 
               - NEVER convert units. If the user says "quarter", use "unit": "quarters". Do NOT convert to "months".
               - "type": "calendar_last" represents completed historical periods (e.g., "last month", "last 5 quarters").
               - "type": "rolling" represents an exact timeframe up to today (e.g., "rolling 90 days", "past 3 months").
               - For "days", ALWAYS use "type": "rolling".
               - If the user provides specific dates (e.g., "Jan 1st 2023 to Feb 9th 2024"), use "type": "explicit_range" and format the extracted dates strictly as "YYYY-MM-DD" in the "start_date" and "end_date" fields. Set "unit" and "value" to null.
               - If "type": "explicit_range", CHECK if the "start_date" is temporaly before the "end_date", if NOT then, set "need_clarificatio": true
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
        Query: "Total fraud amount by transaction type last quarter"
        "temporal_filter": {{
            "expression": "last quarter",
            "type": "calendar_last",
            "unit": "quarters",
            "value": 1
        }}


        Query: "Sales for the last 30 days"
        "temporal_filter": {{
            "expression": "last 30 days",
            "type": "rolling",
            "unit": "days",
            "value": 30
        }}

        Query: "Total fraud amount by transaction type last 5 quarters"
        "temporal_filter": {{
            "expression": "last 5 quarters",
            "type": "calendar_last",  
            "unit": "quarters",
            "value": 5
        }}

        Query: "How much carpet did we sell from 01-01-2023 to 09-02-2024"
        "temporal_filter": {{
            "expression": "from 01-01-2023 to 09-02-2024",
            "type": "explicit_range",
            "unit": null,
            "value": null,
            "start_date": "2023-01-01",
            "end_date": "2024-02-09"
        }}
        </few_shot_examples>
        {error_feedback}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": f"Query: {state['user_query']}"},
        ]

        text         = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm.generate(**model_inputs, max_new_tokens=256, do_sample=False)

        input_length = model_inputs.input_ids.shape[1]
        response     = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

        return {"raw_response": response}

    def node_validate_and_format(self, state: AgentState) -> dict:
        self.last_activity = time.time()
        response = state["raw_response"]

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in model output.")

            result_dict = json.loads(match.group(0))


            required_keys = ["intent", "temporal_filter", "entities", "complexity", "needs_clarification", "clarification_reason"]
            
            for key in required_keys:
                if key not in result_dict:
                    raise ValueError(f"Missing required key: {key}")

            temporal_params = result_dict.get("temporal_filter")
            if temporal_params:
                result_dict["temporal_filter"] = self.calculate_temporal_filter(temporal_params)

            return {"final_json": result_dict, "error_message": ""}

        except Exception as e:
            return {
                "error_message": str(e),
                "retry_count":   state["retry_count"] + 1,
            }

    #Edge Routing
    def route_validation(self, state: AgentState) -> Literal["generate", "end"]:
        if state.get("error_message") and state.get("retry_count", 0) < 3:
            print(f"[Cyclical Route] Validation failed: {state['error_message']}. Routing back to Generator.")
            return "generate"
        return "end"

    #Graph Compilation
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.node_retrieve_schema)
        workflow.add_node("generate", self.node_generate_json)
        workflow.add_node("validate", self.node_validate_and_format)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")

        workflow.add_conditional_edges(
            "validate",
            self.route_validation,
            {"generate": "generate", "end": END},
        )

        return workflow.compile()

    #Final Execution
    def process_query(self, query: str) -> dict:
        if not self.is_active:
            return {"error": "Session terminated. Please restart."}

        initial_state = {
            "user_query":       query,
            "retry_count":      0,
            "error_message":    "",
            "final_json":       {},
            "retrieved_schema": "",
            "raw_response":     "",
        }

        try:
            final_state = self.graph.invoke(initial_state)
            if final_state.get("error_message"):
                return {
                    "error": (
                        f"Pipeline failed after {final_state['retry_count']} retries: "
                        f"{final_state['error_message']}"
                    )
                }
            return final_state["final_json"]
        except Exception as e:
            return {"error": f"Critical Graph Error: {str(e)}"}


if __name__ == "__main__":
    pipeline = LangGraphAgent(timeout_seconds=600)
    pipeline.build_vector_database("./columns_homzhub.csv")

    print("\n Welcome to VEDA")
    print("Type 'exit' to manually end the session.")

    try:
        while pipeline.is_active:
            user_input = input("\nEnter Analytics Query: ")

            if user_input.lower() in ["exit", "kill", "quit"]:
                pipeline.kill_session()
                break

            if user_input.strip() and pipeline.is_active:
                result = pipeline.process_query(user_input)
                print(json.dumps(result, indent=2))

    except KeyboardInterrupt:
        pipeline.kill_session()


# if __name__ == "__main__":
#     pipeline = LangGraphAgent(timeout_seconds=600)
#     pipeline.build_vector_database("./columns_homzhub.csv")

#     test_queries = [
#     "How much carpet did we sell from 01-01-2023 to 30-03-2023",
#     "How much carpet did we sell from 01-01-2023 to 30-03-2022",
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
#     "Customers with no transactions in the last 15 quarters",
#     "Which projects have the highest number of listed assets for lease vs sale?",
#     # "What is the distribution of assets per project and asset type?",
#     # "How many lease listings and sale listings exist for each project?",
#     # "Which project has the highest occupancy rate?",
#     # "What is the average carpet area of assets per project?",
#     # "Which assets currently have active tenants and which are vacant?",
#     # "What is the tenant distribution across different projects?",
#     # "What is the average lease duration per asset type?",
#     # "Which projects have the highest number of tenants?",
#     # "How many lease transactions occurred per project?",
#     # "Which amenities are most common across assets?",
#     # "What is the distribution of amenities across projects?",
#     # "Which projects provide the highest number of amenities?",
#     # "What percentage of assets have premium amenities?",
#     # "Which assets have the highest number of sale negotiations?",
#     # "What is the conversion rate from sale listing → sale transaction?",
#     # "What is the average negotiation duration before a sale transaction?",
#     # "Which projects generate the highest sale value?",
#     # "What is the total payment received per project?",
#     # "What is the payment trend for lease transactions over time?",
#     # "Which tenants contribute the highest rental revenue?",
#     # "What is the payment method distribution for lease payments?",
#     # "What percentage of assets have completed verification documents?",
#     # "Which document types are most commonly submitted?",
#     # "Which projects have the highest number of verified assets?",
#     # "What is the conversion rate from listing leads to tenants?",
#     # "Which projects generate the most leads?",
#     # "What is the average time taken to convert a lead to a tenant?",
#     # "Which asset type generates the highest revenue?",
#     # "What is the distribution of asset types across projects?",
#     # "What is the average lease value per asset type?"
#     ]

#     filename = "benchmark.csv"

#     headers = [
#         "Query", 
#         "Execution_Time_Sec", 
#         "Intent", 
#         "Complexity",
#         "start_date",
#         "end_date",
#         "Needs_Clarification",
#         "Clarification_Reason",
#         "Entities_Extracted"
#     ]

#     try:
#         with open(filename, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=headers)
#             writer.writeheader()

#             for query in test_queries:
#                 if not pipeline.is_active:
#                     break

#                 start_time = time.time()
                
#                 result = pipeline.process_query(query)
#                 duration = time.time() - start_time
#                 temporal = result.get("temporal_filter") or {}
#                 entities = result.get("entities", [])
#                 entities_str = ", ".join(entities) if isinstance(entities, list) else str(entities)

#                 writer.writerow({
#                     "Query": query,
#                     "Execution_Time_Sec": round(duration, 3),
#                     "Intent": result.get("intent", ""),
#                     "Complexity": result.get("complexity", ""),
#                     "start_date": temporal.get("start_date", ""),
#                     "end_date": temporal.get("end_date", ""),
#                     "Needs_Clarification": result.get("needs_clarification", ""),
#                     "Clarification_Reason": result.get("clarification_reason", ""),
#                     "Entities_Extracted": entities_str
#                 })

#         print(f"Files Saved to: {filename}")

#     except Exception as e:
#         print(f"\n Error during testing: {e}")
        
#     finally:
#         pipeline.kill_session()