#!/usr/bin/env python3

import asyncio
import psycopg
from psycopg.rows import dict_row

# Database connection URL
DB_URL = "postgresql://realagentsuser:realagentspass@localhost:5432/embeddings_db"

async def check_books_schema():
    """Check the schema of the books table"""
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Get table schema
            await cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'books'
                ORDER BY ordinal_position
            """)
            columns = await cur.fetchall()
            
            print("=== Books Table Schema ===")
            for col in columns:
                print(f"{col['column_name']} - {col['data_type']}")
            
            # Count total rows
            await cur.execute("SELECT COUNT(*) as count FROM books")
            count = await cur.fetchone()
            print(f"\nTotal books: {count['count']}")
            
            # Get sample data
            await cur.execute("SELECT id, title, author, genre, text FROM books LIMIT 3")
            books = await cur.fetchall()
            
            print("\n=== Sample Book Records ===")
            for book in books:
                print(f"\nID: {book['id']}")
                print(f"Title: {book['title']}")
                print(f"Author: {book['author']}")
                print(f"Genre: {book['genre']}")
                print(f"Text Preview: {book['text'][:150]}...")

if __name__ == "__main__":
    asyncio.run(check_books_schema())
