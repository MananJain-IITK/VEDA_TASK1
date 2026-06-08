import json
import re
import torch
import csv
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import datetime
from datetime import timedelta
import calendar
from dateutil.relativedelta import relativedelta

class QuerySession:
    def __init__(self):
        print("Loading BAAI Embedding Model")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        print("BAAI Loaded!")

        print("Loading Qwen 3B Coder")
        model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,   
            device_map="auto"       
        )
        print("Qwen Loaded Successfully!")

        print("Initializing ChromaDB")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="homzhub_schema")
        print("System Ready!\n")


    def build_vector_database(self, csv_filepath="./columns_homzhub.csv"):
        if self.collection.count() > 0:
            print("Database already contains schema, skipping build.")
            return
            
        print(f"Building {csv_filepath}")
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


    def search_schema(self, user_query, k=5):
        print("Schema is working")
        query_vector = self.embedder.encode(user_query).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=k)
        return "\n".join(results['documents'][0])
    

    def calculate_temporal_filter(self, temporal_params: dict) -> dict | None:
        if not temporal_params:
            return None
            
        now = datetime.datetime.now()
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
                    datetime.datetime(d.year, 1, 1), 
                    datetime.datetime(d.year, 12, 31)
                ),
                "months": lambda d: (
                    datetime.datetime(d.year, d.month, 1), 
                    datetime.datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
                ),
                "quarters": lambda d: (
                    datetime.datetime(d.year, 3 * ((d.month - 1) // 3) + 1, 1),
                    datetime.datetime(d.year, 3 * ((d.month - 1) // 3) + 3, calendar.monthrange(d.year, 3 * ((d.month - 1) // 3) + 3)[1])
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

    def process_query(self, user_query: str) -> dict:
        schema_context = self.search_schema(user_query, k=15)
        sys_prompt = f"""
        You are a SQL query intent parser for a business analytics pipeline. 

        <database_schema_context>
        Relevant schema objects identified for this query:
        {schema_context}
        </database_schema_context>

        <schema>
        Output ONLY a raw JSON object matching this exact schema:
        {{
            "intent": "str",
            "temporal_filter": {{
                "expression": "str", 
                "type": "str",
                "unit": "str",
                "value": int 
            }} | null,
            "entities": ["str"],
            "complexity": "str",
            "needs_clarification": bool,
            "clarification_reason": "str" | null
        }}
        </schema>
        
        <custom_rules>
            1. DOMAIN CHECK: You ONLY process queries about business data. IF the user asks a general knowledge question, makes small talk, or asks something outside this domain, you MUST set "needs_clarification": true.
            2. SPELL AND GRAMMAR CHECK: If the query has simple spelling and grammar errors, make REASONABLE correction and move forward. BUT if the errors are major, set "needs_clarification": true. 
            3. AMBIGUITY: IF the query is incomplete, vague, or fewer than 3 words, set "needs_clarification": true.
            4. OUTPUT: Output raw JSON only. Do NOT wrap in ```json tags.
        </custom_rules>

        <definitions>
            Complexity:
            - simple: 1 datafield accessed, OR Direct SELECT, COUNT, or SUM with basic filters.
            - medium: 2 datafields accessed, OR Multiple conditions, GROUP BY.
            - complex: 3 or more or all datafields accessed, OR NOT EXISTS, window functions, OR includes a MIX of both the simple and medium complexity.

            Intent:
            - SELECT: Show/list items.
            - COUNT: How many, What.
            - AGGREGATE: Total/sum.
            - TOP_N: Top/highest/lowest items.
            - SELECT_SUBQUERY: Complex exclusions.
        </definitions>
        """
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Query: {user_query}"}
        ]
        
        print("Generating intent JSON")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(**model_inputs, max_new_tokens=256, do_sample=False)
            
        input_length = model_inputs.input_ids.shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group(0))
                temporal_params = parsed_json.get("temporal_filter")
                if temporal_params:
                    parsed_json["temporal_filter"] = self.calculate_temporal_filter(temporal_params)
                return parsed_json
            else:
                return {"error": "No JSON object found in model output.", "raw_response": response}
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON.", "raw_response": response}


if __name__ == "__main__":
    QuerySession = QuerySession()
    QuerySession.build_vector_database("./columns_homzhub.csv")
    
    test_query = "What is the conversion rate from sale listing -> sale transaction in the last 30 days?"
    result = QuerySession.process_query(test_query)
    print(json.dumps(result, indent=2))