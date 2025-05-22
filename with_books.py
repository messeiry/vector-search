import asyncio
import fastapi
import ollama
import pgai
import psycopg
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
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
from pydantic import BaseModel
from datetime import datetime, timedelta


DB_URL = "postgresql://realagentsuser:realagentspass@localhost:5432/embeddings_db"


logger = logging.getLogger('fastapi_realagents_nlsearch')

async def setup_pgvector_psycopg(conn: psycopg.AsyncConnection):
    await register_vector_async(conn)

# Change to AsyncConnectionPool
pool = AsyncConnectionPool(DB_URL, min_size=5, max_size=10, open=False, configure=setup_pgvector_psycopg)



async def create_vectorizer():
    vectorizer_statement = CreateVectorizer(
        source="books",
        destination= DestinationTableConfig(target_table="books_embeddings"),
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

# Models for book checkout operations
class CheckoutRequest(BaseModel):
    book_id: int
    days: int = 14  # Default checkout period is 14 days

class CheckInRequest(BaseModel):
    book_id: int

class DeleteRequest(BaseModel):
    book_id: int

class BookUpdateRequest(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[str] = None
    isbn: Optional[str] = None
    genre: Optional[str] = None
    pages: Optional[int] = None
    language: Optional[str] = None
    publisher: Optional[str] = None
    description: Optional[str] = None
    rating: Optional[float] = None

class BookCreateRequest(BaseModel):
    title: str
    author: str
    publish_date: str
    isbn: str
    genre: str
    pages: int
    language: str = "English"
    publisher: str
    description: str
    rating: float = 3.0

class BookOperationResponse(BaseModel):
    success: bool
    message: str
    book_id: int
    title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    checked_out: Optional[bool] = None
    checkout_date: Optional[str] = None
    due_date: Optional[str] = None


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

@dataclass
class BooksSearchResult:
    id: int
    title: str
    author: str
    publish_date: str
    isbn: str
    genre: str
    pages: int
    language: str
    publisher: str
    description: str
    rating: float
    text: str
    checked_out: bool
    checkout_date: Optional[str]
    due_date: Optional[str]
    distance: float
    
async def _find_relevant_chunks(client: ollama.AsyncClient, query: str, limit: int = 10) -> List[BooksSearchResult]:
    response = await client.embed(model="all-minilm", input=query)
    embedding = np.array(response.embeddings[0])
    
    async with pool.connection() as conn:
        # Use BooksSearchResult for the main query with no fallback
        async with conn.cursor(row_factory=class_row(BooksSearchResult)) as cur:
            # Direct vector similarity search - no fallback
            await cur.execute("""
                SELECT b.id, b.title, b.author, b.publish_date, b.isbn, b.genre, 
                       b.pages, b.language, b.publisher, b.description, b.rating, 
                       b.text, b.checked_out, b.checkout_date, b.due_date,
                       e.embedding <=> %s as distance
                FROM books b
                JOIN books_embeddings e ON b.id = e.id
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

@app.post("/books/checkout", response_model=BookOperationResponse)
async def checkout_book(request: CheckoutRequest):
    """
    Check out a book from the library
    
    Args:
        request: Contains book_id and optional checkout duration in days
        
    Returns:
        BookOperationResponse: Success status and details about the checkout
    """
    book_id = request.book_id
    checkout_days = request.days
    
    # Get current date and calculate due date
    today = datetime.now().date()
    due_date = today + timedelta(days=checkout_days)
    
    async with pool.connection() as conn:
        # First check if the book exists and is available
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title, author, genre, isbn, publisher, checked_out
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            if book['checked_out']:
                raise HTTPException(status_code=400, detail="Book is already checked out")
            
            # Update the book status to checked out
            await cur.execute("""
                UPDATE books
                SET checked_out = TRUE,
                    checkout_date = %s,
                    due_date = %s
                WHERE id = %s
                RETURNING title, author, genre, isbn, publisher
            """, (today.isoformat(), due_date.isoformat(), book_id))
            
            result = await cur.fetchone()
            
        await conn.commit()
        
    return BookOperationResponse(
        success=True,
        message=f"Book '{result['title']}' has been checked out successfully.",
        book_id=book_id,
        title=result['title'],
        author=result['author'],
        genre=result['genre'],
        isbn=result['isbn'],
        publisher=result['publisher'],
        checked_out=True,
        checkout_date=today.isoformat(),
        due_date=due_date.isoformat()
    )

@app.post("/books/checkin", response_model=BookOperationResponse)
async def checkin_book(request: CheckInRequest):
    """
    Check in a previously checked out book
    
    Args:
        request: Contains book_id to check in
        
    Returns:
        BookOperationResponse: Success status and details about the check-in
    """
    book_id = request.book_id
    
    async with pool.connection() as conn:
        # First check if the book exists and is checked out
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title, author, genre, isbn, publisher, checked_out
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            if not book['checked_out']:
                raise HTTPException(status_code=400, detail="Book is not currently checked out")
            
            # Update the book status to checked in
            await cur.execute("""
                UPDATE books
                SET checked_out = FALSE,
                    checkout_date = NULL,
                    due_date = NULL
                WHERE id = %s
                RETURNING title, author, genre, isbn, publisher
            """, (book_id,))
            
            result = await cur.fetchone()
            
        await conn.commit()
        
    return BookOperationResponse(
        success=True,
        message=f"Book '{result['title']}' has been checked in successfully.",
        book_id=book_id,
        title=result['title'],
        author=result['author'],
        genre=result['genre'],
        isbn=result['isbn'],
        publisher=result['publisher'],
        checked_out=False
    )

@app.post("/books/delete", response_model=BookOperationResponse)
async def delete_book(request: DeleteRequest):
    """
    Delete a book from the library
    
    Args:
        request: Contains book_id to delete the book
        
    Returns:
        BookOperationResponse: Success status and details about the deletion
    """
    book_id = request.book_id
    
    async with pool.connection() as conn:
        # Check if the book exists
        async with conn.cursor(row_factory=dict_row) as cur:
            # Get full book details before deletion for response
            await cur.execute("""
                SELECT id, title, author, genre, isbn, publisher, checked_out, checkout_date, due_date
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            # Delete the book
            await cur.execute("""
                DELETE FROM books
                WHERE id = %s
            """, (book_id,))
            
            # Also try to delete from embeddings table if it exists
            try:
                await cur.execute("""
                    DELETE FROM books_embeddings
                    WHERE id = %s
                """, (book_id,))
            except Exception as e:
                # If embeddings table doesn't exist or other error, just log it
                logger.warning(f"Could not delete from books_embeddings: {str(e)}")
            
        await conn.commit()
        
    return BookOperationResponse(
        success=True,
        message=f"Book '{book['title']}' has been deleted successfully.",
        book_id=book_id,
        title=book['title'],
        author=book['author'],
        genre=book['genre'],
        isbn=book['isbn'],
        publisher=book['publisher'],
        checked_out=book['checked_out'],
        checkout_date=book['checkout_date'],
        due_date=book['due_date']
    )

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
    You are an AI assistant with access to a database of books with detailed information about titles, authors, genres, and descriptions.

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
            
            # Add book information to unique chunks
            unique_chunks.append({
                "id": chunk.id,
                "title": chunk.title,
                "author": chunk.author,
                "genre": chunk.genre,
                "publish_date": chunk.publish_date,
                "isbn": chunk.isbn,
                "pages": chunk.pages,
                "language": chunk.language,
                "publisher": chunk.publisher,
                "rating": chunk.rating,
                "checked_out": chunk.checked_out,
                "checkout_date": chunk.checkout_date,
                "due_date": chunk.due_date,
                "distance": chunk.distance
            })
    
    response = {
        "query": query,
        "response": response['response'],
        "chunks": unique_chunks
    }
    
    return response

@app.get("/books")
async def list_books(
    limit: int = 20, 
    offset: int = 0,
    checked_out: Optional[bool] = None
):
    """
    List books with optional filtering by checkout status
    
    Args:
        limit: Maximum number of books to return (default 20)
        offset: Number of books to skip (default 0)
        checked_out: If provided, filter books by checked_out status
        
    Returns:
        List of books with their details
    """
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            if checked_out is not None:
                # Filter by checkout status
                query = """
                    SELECT id, title, author, genre, publish_date, checked_out, 
                           checkout_date, due_date
                    FROM books
                    WHERE checked_out = %s
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                await cur.execute(query, (checked_out, limit, offset))
            else:
                # Return all books
                query = """
                    SELECT id, title, author, genre, publish_date, checked_out, 
                           checkout_date, due_date
                    FROM books
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                await cur.execute(query, (limit, offset))
            
            books = await cur.fetchall()
            
            # Also get total count for pagination
            if checked_out is not None:
                await cur.execute("SELECT COUNT(*) as total FROM books WHERE checked_out = %s", (checked_out,))
            else:
                await cur.execute("SELECT COUNT(*) as total FROM books")
                
            total = await cur.fetchone()
            
            return {
                "books": books,
                "total": total["total"],
                "limit": limit,
                "offset": offset
            }

@app.get("/books/{book_id}")
async def get_book(book_id: int):
    """
    Get details for a specific book by ID
    
    Args:
        book_id: The ID of the book to retrieve
        
    Returns:
        Book details
    """
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title, author, publish_date, isbn, genre, pages, language,
                       publisher, description, rating, checked_out, checkout_date, due_date
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            return book

@app.delete("/books/{book_id}", response_model=BookOperationResponse)
async def delete_book(book_id: int):
    """
    Delete a book from the database
    
    Args:
        book_id: The ID of the book to delete
        
    Returns:
        BookOperationResponse: Success status and details about the deletion
    """
    async with pool.connection() as conn:
        # First check if the book exists
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            # Store title for response
            book_title = book['title']
            
            # Delete the book
            await cur.execute("""
                DELETE FROM books
                WHERE id = %s
            """, (book_id,))
            
            # Also try to delete from embeddings table if it exists
            try:
                await cur.execute("""
                    DELETE FROM books_embeddings
                    WHERE id = %s
                """, (book_id,))
            except Exception as e:
                # If embeddings table doesn't exist or other error, just log it
                logger.warning(f"Could not delete from books_embeddings: {str(e)}")
            
        await conn.commit()
        
    return BookOperationResponse(
        success=True,
        message=f"Book '{book_title}' has been deleted successfully.",
        book_id=book_id,
        title=book_title
    )

@app.post("/books/delete", response_model=BookOperationResponse)
async def delete_book_by_request(request: DeleteRequest):
    """
    Delete a book from the database using request body
    
    Args:
        request: Contains book_id to delete the book
        
    Returns:
        BookOperationResponse: Success status and details about the deletion
    """
    book_id = request.book_id
    
    async with pool.connection() as conn:
        # First check if the book exists
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title, checked_out
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            # Store title for response
            book_title = book['title']
            
            # Delete the book
            await cur.execute("""
                DELETE FROM books
                WHERE id = %s
            """, (book_id,))
            
            # Also try to delete from embeddings table if it exists
            try:
                await cur.execute("""
                    DELETE FROM books_embeddings
                    WHERE id = %s
                """, (book_id,))
            except Exception as e:
                # If embeddings table doesn't exist or other error, just log it
                logger.warning(f"Could not delete from books_embeddings: {str(e)}")
            
        await conn.commit()
        
    return BookOperationResponse(
        success=True,
        message=f"Book '{book_title}' has been deleted successfully.",
        book_id=book_id,
        title=book_title
    )

@app.put("/books/{book_id}", response_model=BookOperationResponse)
async def update_book(book_id: int, book_data: BookUpdateRequest):
    """
    Update book information
    
    Args:
        book_id: The ID of the book to update
        book_data: The updated book data
        
    Returns:
        BookOperationResponse: Success status and details about the update
    """
    async with pool.connection() as conn:
        # First check if the book exists
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            # Build dynamic update query based on provided fields
            update_fields = []
            params = []
            
            for field, value in book_data.dict(exclude_unset=True).items():
                if value is not None:
                    update_fields.append(f"{field} = %s")
                    params.append(value)
            
            if not update_fields:
                raise HTTPException(status_code=400, detail="No fields provided for update")
            
            # Build and execute the update query
            query = f"""
                UPDATE books
                SET {", ".join(update_fields)}
                WHERE id = %s
                RETURNING title
            """
            
            # Add book_id as the last parameter
            params.append(book_id)
            
            await cur.execute(query, tuple(params))
            result = await cur.fetchone()
            
            # Also update the text field to reflect the changes
            # First get all current book data
            await cur.execute("""
                SELECT title, author, publish_date, isbn, genre, pages, language,
                       publisher, description, rating, checked_out, checkout_date, due_date
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            updated_book = await cur.fetchone()
            
            # Generate new text field based on updated data
            checkout_status = f"This book is currently checked out since {updated_book['checkout_date']} and due back on {updated_book['due_date']}." if updated_book['checked_out'] else "This book is available for checkout."
            
            new_text = f"""
                '{updated_book['title']}' is a {updated_book['genre']} book written by {updated_book['author']} and published on {updated_book['publish_date']} 
                by {updated_book['publisher']}. The book is {updated_book['pages']} pages long and written in {updated_book['language']}. 
                It has an ISBN of {updated_book['isbn']} and an average rating of {updated_book['rating']}/5. 
                Book description: {updated_book['description']}
                {checkout_status}
            """.strip()
            
            # Update the text field
            await cur.execute("""
                UPDATE books
                SET text = %s
                WHERE id = %s
            """, (new_text, book_id))
            
        await conn.commit()
    
    return BookOperationResponse(
        success=True,
        message=f"Book '{updated_book['title']}' has been updated successfully.",
        book_id=book_id,
        title=updated_book['title'],
        author=updated_book['author'],
        genre=updated_book['genre'],
        isbn=updated_book['isbn'],
        publisher=updated_book['publisher'],
        checked_out=updated_book['checked_out'],
        checkout_date=updated_book['checkout_date'],
        due_date=updated_book['due_date']
    )

class BookUpdateRequestWithId(BookUpdateRequest):
    book_id: int

@app.post("/books/update", response_model=BookOperationResponse)
async def update_book_by_request(request: BookUpdateRequestWithId):
    """
    Update book information using request body
    
    Args:
        request: Contains book_id and fields to update
        
    Returns:
        BookOperationResponse: Success status and details about the update
    """
    book_id = request.book_id
    # Create a new BookUpdateRequest without the book_id field
    update_data = BookUpdateRequest(**{k: v for k, v in request.dict().items() if k != 'book_id'})
    
    async with pool.connection() as conn:
        # First check if the book exists
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            book = await cur.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")
            
            # Build dynamic update query based on provided fields
            update_fields = []
            params = []
            
            for field, value in update_data.dict(exclude_unset=True).items():
                if value is not None:
                    update_fields.append(f"{field} = %s")
                    params.append(value)
            
            if not update_fields:
                raise HTTPException(status_code=400, detail="No fields provided for update")
            
            # Build and execute the update query
            query = f"""
                UPDATE books
                SET {", ".join(update_fields)}
                WHERE id = %s
                RETURNING title
            """
            
            # Add book_id as the last parameter
            params.append(book_id)
            
            await cur.execute(query, tuple(params))
            result = await cur.fetchone()
            
            # Also update the text field to reflect the changes
            # First get all current book data
            await cur.execute("""
                SELECT title, author, publish_date, isbn, genre, pages, language,
                       publisher, description, rating, checked_out, checkout_date, due_date
                FROM books
                WHERE id = %s
            """, (book_id,))
            
            updated_book = await cur.fetchone()
            
            # Generate new text field based on updated data
            checkout_status = f"This book is currently checked out since {updated_book['checkout_date']} and due back on {updated_book['due_date']}." if updated_book['checked_out'] else "This book is available for checkout."
            
            new_text = f"""
                '{updated_book['title']}' is a {updated_book['genre']} book written by {updated_book['author']} and published on {updated_book['publish_date']} 
                by {updated_book['publisher']}. The book is {updated_book['pages']} pages long and written in {updated_book['language']}. 
                It has an ISBN of {updated_book['isbn']} and an average rating of {updated_book['rating']}/5. 
                Book description: {updated_book['description']}
                {checkout_status}
            """.strip()
            
            # Update the text field
            await cur.execute("""
                UPDATE books
                SET text = %s
                WHERE id = %s
            """, (new_text, book_id))
            
        await conn.commit()
    
    return BookOperationResponse(
        success=True,
        message=f"Book '{updated_book['title']}' has been updated successfully.",
        book_id=book_id,
        title=updated_book['title'],
        author=updated_book['author'],
        genre=updated_book['genre'],
        isbn=updated_book['isbn'],
        publisher=updated_book['publisher'],
        checked_out=updated_book['checked_out'],
        checkout_date=updated_book['checkout_date'],
        due_date=updated_book['due_date']
    )

@app.post("/books/create", response_model=BookOperationResponse)
async def create_book(book_data: BookCreateRequest):
    """
    Create a new book in the library
    
    Args:
        book_data: The book data to create
        
    Returns:
        BookOperationResponse: Success status and details about the new book
    """
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Check if a book with the same ISBN already exists
            await cur.execute("""
                SELECT id FROM books WHERE isbn = %s
            """, (book_data.isbn,))
            
            existing_book = await cur.fetchone()
            
            if existing_book:
                raise HTTPException(status_code=400, detail=f"Book with ISBN {book_data.isbn} already exists")
            
            # Generate text field for natural language search
            checkout_status = "This book is available for checkout."
            
            text = f"""
                '{book_data.title}' is a {book_data.genre} book written by {book_data.author} and published on {book_data.publish_date} 
                by {book_data.publisher}. The book is {book_data.pages} pages long and written in {book_data.language}. 
                It has an ISBN of {book_data.isbn} and an average rating of {book_data.rating}/5. 
                Book description: {book_data.description}
                {checkout_status}
            """.strip()
            
            # Insert new book
            await cur.execute("""
                INSERT INTO books 
                (title, author, publish_date, isbn, genre, pages, language, publisher, description, rating, checked_out, text)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, title, author, genre, isbn, publisher
            """, (
                book_data.title, 
                book_data.author, 
                book_data.publish_date, 
                book_data.isbn, 
                book_data.genre, 
                book_data.pages, 
                book_data.language, 
                book_data.publisher, 
                book_data.description, 
                book_data.rating, 
                False,  # initially not checked out
                text
            ))
            
            new_book = await cur.fetchone()
            
        await conn.commit()
    
    return BookOperationResponse(
        success=True,
        message=f"Book '{new_book['title']}' has been created successfully.",
        book_id=new_book['id'],
        title=new_book['title'],
        author=new_book['author'],
        genre=new_book['genre'],
        isbn=new_book['isbn'],
        publisher=new_book['publisher'],
        checked_out=False
    )