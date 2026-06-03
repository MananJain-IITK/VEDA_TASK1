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
        self.monitor_thread = threading.Thread(target=self._monitor_inactivity, daemon=True)
        self.monitor_thread.start()
        
        # Compile Graph
        print("Compiling LangGraph Workflow")
        self.graph = self._build_graph()
        print("System Ready!\n")

    # Hardware Management
    def _monitor_inactivity(self):
        while self.is_active:
            time.sleep(5)  
            time_inactive = time.time() - self.last_activity
            if time_inactive > self.timeout_seconds:
                print(f"\n[System] No activity for {self.timeout_seconds} seconds. Auto-killing session to free VRAM")
                self.kill_session()
                break

    def kill_session(self):
        if not self.is_active: return
        self.is_active = False
        print("\nShutting down session and freeing up resources")
        
        del self.llm
        del self.tokenizer
        del self.embedder
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VRAM cleared. Session terminated.")

    # Database / RAG Methods
    def build_vector_database(self, csv_filepath="./columns_homzhub.csv"):
        if self.collection.count() > 0:
            print("Database already contains schema, skipping build.")
            return

        documents = []
        metadata = []
        ids = []
        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if not row or len(row) < 3: continue
                table, column, dtype = row[0].strip(), row[1].strip(), row[2].strip()
                description = f"Table: {table}. Column: {column}. Dtype: {dtype}."
                
                documents.append(description)
                metadata.append({"table": table, "column": column})
                ids.append(f"row_{idx}")
            
            print("Calculating Embeddings and Saving to ChromaDB")
            embeddings = self.embedder.encode(documents).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            print("Database Built Successfully!\n")

    # Temporal Engine
    def calculate_temporal_filter(self, temporal_params: dict) -> dict | None:
        if not temporal_params: return None
            
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
                "years": lambda d: (datetime(d.year, 1, 1), datetime(d.year, 12, 31)),
                "months": lambda d: (datetime(d.year, d.month, 1), datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1])),
                "quarters": lambda d: (datetime(d.year, 3 * ((d.month - 1) // 3) + 1, 1), datetime(d.year, 3 * ((d.month - 1) // 3) + 3, calendar.monthrange(d.year, 3 * ((d.month - 1) // 3) + 3)[1])),
                "weeks": lambda d: (d - timedelta(days=d.weekday()), d + timedelta(days=6 - d.weekday()))
            }
            if unit in calendar_bounds:
                start_date, end_date = calendar_bounds[unit](target_date)

        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }   

    # LangGraph Nodes
    def node_retrieve_schema(self, state: AgentState) -> dict:
        self.last_activity = time.time()
        query = state["user_query"]
        
        query_vector = self.embedder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=15)
        
        schema_context = "\n".join(results['documents'][0])
        return {"retrieved_schema": schema_context, "retry_count": state.get("retry_count", 0)}

    def node_generate_json(self, state: AgentState) -> dict:
        self.last_activity = time.time()
        
        error_feedback = ""
        if state.get("error_message"):
            error_feedback = f"\n<error_feedback>\nYour previous attempt failed with error: {state['error_message']}. Please fix the JSON structure and keys.\n</error_feedback>"

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
                "type": "str", // rolling | calendar_last | calendar_current
                "unit": "str", // days | weeks | months | quarters | years
                "value": int 
            }} | null,
            "entities": ["str"], // Business concepts, mapping to tables/columns in context if possible
            "complexity": "str", // simple | medium | complex
            "needs_clarification": bool,
            "clarification_reason": "str" | null
        }}
        </schema>
        
        <custom_rules>
            1. DOMAIN CHECK: You ONLY process queries about business data (transactions, customers, sales, fraud, etc.). IF the user asks a general knowledge question (e.g., "What is money?"), makes small talk, or asks something outside this domain, you MUST set "needs_clarification": true.
            2. SPELL AND GRAMMAR CHECK: If the query has simple spelling and grammar errors. Make REASONABLE correction and move forward. BUT if the errors are major, set "needs_clarification": true. 
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
        {error_feedback}
        """
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Query: {state['user_query']}"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(**model_inputs, max_new_tokens=256, do_sample=False)
            
        input_length = model_inputs.input_ids.shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        return {"raw_response": response}

    def node_validate_and_format(self, state: AgentState) -> dict:
        self.last_activity = time.time()
        response = state["raw_response"]
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in model output.")
                
            result_dict = json.loads(match.group(0))
            
            # Validate keys
            required_keys = ["intent", "temporal_filter", "entities", "complexity", "needs_clarification", "clarification_reason"]
            for key in required_keys:
                if key not in result_dict:
                    raise ValueError(f"Missing required key: {key}")
            
            # Apply temporal calculation
            temporal_params = result_dict.get("temporal_filter")
            if temporal_params:
                result_dict["temporal_filter"] = self.calculate_temporal_filter(temporal_params)
                
            return {"final_json": result_dict, "error_message": ""}
            
        except Exception as e:
            return {"error_message": str(e), "retry_count": state["retry_count"] + 1}

    # Edge Routing
    def route_validation(self, state: AgentState) -> Literal["generate", "end"]:
        if state.get("error_message") and state.get("retry_count", 0) < 3:
            print(f"[Cyclical Route] Validation failed: {state['error_message']}. Routing back to Generator.")
            return "generate"
        return "end"

    # Graph Compilation
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("retrieve", self.node_retrieve_schema)
        workflow.add_node("generate", self.node_generate_json)
        workflow.add_node("validate", self.node_validate_and_format)
        
        # Define Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        
        # Conditional Cyclical Edge
        workflow.add_conditional_edges(
            "validate",
            self.route_validation,
            {
                "generate": "generate",
                "end": END
            }
        )
        
        return workflow.compile()

    # Execution Endpoint
    def process_query(self, query: str) -> dict:
        if not self.is_active:
            return {"error": "Session terminated. Please restart the application."}
            
        initial_state = {
            "user_query": query,
            "retry_count": 0,
            "error_message": "",
            "final_json": {}
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            if final_state.get("error_message"):
                return {"error": f"Pipeline failed after {final_state['retry_count']} retries: {final_state['error_message']}"}
            return final_state["final_json"]
        except Exception as e:
            return {"error": f"Critical Graph Error: {str(e)}"}


if __name__ == "__main__":
    pipeline = LangGraphAgent(timeout_seconds=600)
    pipeline.build_vector_database("./columns_homzhub.csv")
    
    print("\ Homzhub Agentic Pipeline (LangGraph)")
    print("Type 'exit' to manually end the session.")
    
    try:
        while pipeline.is_active:
            user_input = input("\nEnter Analytics Query: ")
            
            if user_input.lower() in ['exit', 'kill', 'quit']:
                pipeline.kill_session()
                break
                
            if user_input.strip() and pipeline.is_active:
                result = pipeline.process_query(user_input)
                print(json.dumps(result, indent=2))
                
    except KeyboardInterrupt:
        pipeline.kill_session()