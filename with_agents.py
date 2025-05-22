import asyncio
import fastapi
import ollama
import pgai
import psycopg
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pgai.vectorizer import Worker
from fastapi.logger import logger as fastapi_logger
from pgai.vectorizer import CreateVectorizer
from pgai.vectorizer.configuration import EmbeddingOllamaConfig, LoadingColumnConfig, DestinationTableConfig
from datasets import load_dataset
import logging
from typing import List, Optional, Tuple, Dict
from psycopg_pool import AsyncConnectionPool
from dataclasses import dataclass, asdict
from psycopg.rows import class_row, dict_row
from pgvector.psycopg import register_vector_async
import numpy as np
import json 


DB_URL = "postgresql://realagentsuser:realagentspass@localhost:5432/embeddings_db"


logger = logging.getLogger('fastapi_realagents_nlsearch')

async def setup_pgvector_psycopg(conn: psycopg.AsyncConnection):
    await register_vector_async(conn)

# Change to AsyncConnectionPool
pool = AsyncConnectionPool(DB_URL, min_size=5, max_size=10, open=False, configure=setup_pgvector_psycopg)

async def fetch_agent(id: int) -> Optional[Dict]:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT a.*, p.* 
                FROM agents a
                LEFT JOIN agent_performance p ON p.agent_id = a.id
                WHERE a.id = %s
            """, (id,))
            result = await cur.fetchone()
            return result

async def create_vectorizer():
    vectorizer_statement = CreateVectorizer(
        source="agents",
        destination= DestinationTableConfig(target_table="agents_embeddings"),
        loading=LoadingColumnConfig(column_name='text'),
        embedding=EmbeddingOllamaConfig(model='all-minilm', dimensions=384, base_url="http://localhost:11434")
    ).to_sql()
    
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(vectorizer_statement)
            await conn.commit()
    except Exception as e:
        if "already exists" in str(e):
            # ignore if the vectorizer already exists
            pass
        else:
            raise e

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting lifespan")
    
    # TODO: Remove this once we have latest extension in docker image
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP EXTENSION IF EXISTS ai")
        await conn.commit()
    
    # install pgai tables and functions into the database
    pgai.install(DB_URL)
   
    # Initialize the pool after the pgai tables and functions are installed
    await pool.open()
    
    # start the Worker in a new task running in the background
    worker = Worker(DB_URL)
    task = asyncio.create_task(worker.run())
        
    await create_vectorizer()
    
    yield
    
    print("Shutting down...")
    
    # Close the pool during shutdown
    print("Closing pool")
    await pool.close()
    
    print("gracefully shutting down worker...")
    await worker.request_graceful_shutdown()
    try:
        result = await asyncio.wait_for(task, timeout=20)
        if result is not None:
            print("Worker shutdown with exception:", result)
        else:
            print("Worker shutdown successfully")
    except asyncio.TimeoutError:
        print("Worker did not shutdown in time, killing it")
    
    print("Shutting down complete")

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@dataclass
class AgentSearchResult:
    id: int
    url: str
    title: str
    text: str
    name: str
    agent_legacy_id: str
    chunk: str
    # Distance from vector search
    distance: float
    # Fields from agent_performance table
    perf_id: Optional[int] = None
    agent_id: Optional[int] = None
    year: Optional[int] = None
    quarter: Optional[int] = None
    rank: Optional[int] = None
    sales_amount: Optional[float] = None
    sales_amount_last_year: Optional[float] = None
    change_percentage: Optional[float] = None
    total_commission: Optional[float] = None
    average_sell_time: Optional[int] = None
    client_satisfaction: Optional[float] = None
    total_deals: Optional[int] = None
    last_year_deals: Optional[int] = None
    deals_in_progress: Optional[int] = None
    avg_days_on_market: Optional[int] = None
    lead_conversion_rate: Optional[float] = None
    renewal_rate: Optional[float] = None
    new_listings_count: Optional[int] = None
    sales_growth_goal: Optional[float] = None
    sales_growth_projection: Optional[float] = None
    performance_score: Optional[float] = None
    performance_category: Optional[str] = None

async def _find_relevant_chunks(client: ollama.AsyncClient, query: str, limit: int = 10) -> List[AgentSearchResult]:
    response = await client.embed(model="all-minilm", input=query)
    embedding = np.array(response.embeddings[0])
    
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=class_row(AgentSearchResult)) as cur:
            await cur.execute("""
                SELECT w.id, w.url, w.title, w.text, w.name, w.agent_legacy_id, w.chunk, 
                       p.id as perf_id, p.agent_id, p.year, p.quarter, p.rank, 
                       p.sales_amount, p.sales_amount_last_year, p.change_percentage,
                       p.total_commission, p.average_sell_time, p.client_satisfaction,
                       p.total_deals, p.last_year_deals, p.deals_in_progress,
                       p.avg_days_on_market, p.lead_conversion_rate, p.renewal_rate,
                       p.new_listings_count, p.sales_growth_goal, p.sales_growth_projection,
                       p.performance_score, p.performance_category,
                       w.embedding <=> %s as distance
                FROM agents_embedding w
                LEFT JOIN agent_performance p ON p.agent_id = w.id
                ORDER BY distance
                LIMIT %s
            """, (embedding, limit))
            
            return await cur.fetchall()

@app.get("/search")
async def search(query: str):
    client = ollama.AsyncClient(host="http://localhost:11434")
    results = await _find_relevant_chunks(client, query)
    return [asdict(result) for result in results]  

@app.get("/vectorizer_status")
async def vectorizer_status():
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM ai.vectorizer_status")
            return await cur.fetchall()

# @app.post("/insert_pgai_article")
# async def insert_pgai_article():
#     async with pool.connection() as conn:
#         async with conn.cursor() as cur:
#             await cur.execute("""
#                 INSERT INTO wiki (url, title, text)
#                 VALUES (%s, %s, %s)
#             """, (
#                 "https://en.wikipedia.org/wiki/Pgai",
#                 "pgai - Power your AI applications with PostgreSQL",
#                 "pgai is a tool to make developing RAG and other AI applications easier..."
#             ))
#             await conn.commit()
#     return {"message": "Article inserted successfully"}

# am okay with the results for now, we need to change the prompt to be more specific in the answer as well as the function they search db to be simple for llm to process.

@app.get("/rag")
async def rag(query: str) -> Optional[Dict]:
    """
    Generate a RAG response using pgai, Ollama embeddings, and database content.
    
    Args:
        query_text: The question or query to answer
    
    Returns:
        str: The generated response from the LLM
    """
    # Initialize Ollama client
    client = ollama.AsyncClient(host="http://localhost:11434")
    chunks = await _find_relevant_chunks(client, query, limit=2)
    context = "\n\n".join(
        f"{chunk.title}:\n{chunk.text}" 
        for chunk in chunks
    )
    
    logger.debug(f"Context: {context}")
    
    # Construct prompt with context
#     prompt = f"""Question: {query}
# Please use the following context to provide an accurate short answer: {context}

# Answer:"""

    prompt = f"""
    You are an AI assistant with access to detailed performance data of real estate agents, including metrics such as sales volume, response time, client satisfaction, deal closure rates, and regional activity.

    Question from user: {query}

    Use the following data context to provide a concise, accurate, and helpful answer:  
    {context}

    Answer (limit to a few informative sentences): 
    """
            
    # Generate response using Ollama SDK
    response = await client.generate(
        model='llama3.2',
        prompt=prompt,
        stream=False
    )
    
    # Create a dictionary to track seen IDs and store unique chunks
    seen_ids = set()
    unique_chunks = []
    
    for chunk in chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            # Create performance data object if available
            performance_data = None
            if chunk.perf_id is not None:
                performance_data = {
                    "id": chunk.perf_id,
                    "year": chunk.year,
                    "quarter": chunk.quarter,
                    "rank": chunk.rank,
                    "sales_amount": chunk.sales_amount,
                    "sales_amount_last_year": chunk.sales_amount_last_year,
                    "change_percentage": chunk.change_percentage,
                    "total_commission": chunk.total_commission,
                    "average_sell_time": chunk.average_sell_time,
                    "client_satisfaction": chunk.client_satisfaction,
                    "total_deals": chunk.total_deals,
                    "last_year_deals": chunk.last_year_deals,
                    "deals_in_progress": chunk.deals_in_progress,
                    "avg_days_on_market": chunk.avg_days_on_market,
                    "lead_conversion_rate": chunk.lead_conversion_rate,
                    "renewal_rate": chunk.renewal_rate,
                    "new_listings_count": chunk.new_listings_count,
                    "sales_growth_goal": chunk.sales_growth_goal,
                    "sales_growth_projection": chunk.sales_growth_projection,
                    "performance_score": chunk.performance_score,
                    "performance_category": chunk.performance_category
                }
            
            # Add to unique chunks
            unique_chunks.append({
                "id": chunk.id, 
                "name": chunk.name, 
                "agent_legacy_id": chunk.agent_legacy_id, 
                "distance": chunk.distance,
                "performance": performance_data
            })
    
    response = {
        "query": query,
        "response": response['response'],
        "chunks": unique_chunks
    }
    
    return response