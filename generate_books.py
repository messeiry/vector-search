#!/usr/bin/env python3

import psycopg
import asyncio
import datetime
from psycopg.rows import dict_row
from faker import Faker
import random
from typing import List, Dict, Any
import json
from pgvector.psycopg import register_vector_async

# Database connection URL
DB_URL = "postgresql://realagentsuser:realagentspass@localhost:5432/embeddings_db"

# Initialize Faker for generating realistic book data
fake = Faker()

# List of common book genres
GENRES = [
    "Fiction", "Non-Fiction", "Mystery", "Science Fiction", "Fantasy", 
    "Romance", "Thriller", "Horror", "Biography", "History",
    "Self-Help", "Business", "Travel", "Cooking", "Poetry"
]

# Book schema structure
class Book:
    def __init__(
        self,
        title: str,
        author: str,
        publish_date: str,
        isbn: str,
        genre: str,
        pages: int,
        language: str,
        publisher: str,
        description: str,
        rating: float,
        checked_out: bool = False,
        checkout_date: str = None,
        due_date: str = None
    ):
        self.title = title
        self.author = author
        self.publish_date = publish_date
        self.isbn = isbn
        self.genre = genre
        self.pages = pages
        self.language = language
        self.publisher = publisher
        self.description = description
        self.rating = rating
        self.checked_out = checked_out
        self.checkout_date = checkout_date
        self.due_date = due_date
        
        # Generate a natural language text field that combines all book information
        checkout_status = f"This book is currently checked out since {checkout_date} and due back on {due_date}." if checked_out else "This book is available for checkout."
        
        self.text = f"""
            '{title}' is a {genre} book written by {author} and published on {publish_date} 
            by {publisher}. The book is {pages} pages long and written in {language}. 
            It has an ISBN of {isbn} and an average rating of {rating}/5. 
            Book description: {description}
            {checkout_status}
        """.strip()

def generate_random_book() -> Book:
    """Generate a random book with realistic metadata"""
    title = fake.catch_phrase()
    author = fake.name()
    publish_date = fake.date_between(start_date='-50y', end_date='today').strftime('%Y-%m-%d')
    isbn = f"978-{random.randint(0, 9)}-{random.randint(10000, 99999)}-{random.randint(100, 999)}-{random.randint(0, 9)}"
    genre = random.choice(GENRES)
    pages = random.randint(80, 1200)
    language = "English"  # Could be expanded to include more languages
    publisher = fake.company()
    description = fake.paragraph(nb_sentences=5)
    rating = round(random.uniform(2.0, 5.0), 1)
    
    # Randomly determine if the book is checked out (20% chance)
    checked_out = random.random() < 0.2
    checkout_date = None
    due_date = None
    
    if checked_out:
        # Book was checked out within the last 30 days
        checkout_date_obj = fake.date_between(start_date='-30d', end_date='today')
        checkout_date = checkout_date_obj.strftime('%Y-%m-%d')
        # Due date is between 1 and 30 days from checkout date
        due_date_obj = checkout_date_obj + datetime.timedelta(days=random.randint(7, 30))
        due_date = due_date_obj.strftime('%Y-%m-%d')
    
    return Book(
        title=title,
        author=author,
        publish_date=publish_date,
        isbn=isbn,
        genre=genre,
        pages=pages,
        language=language,
        publisher=publisher,
        description=description,
        rating=rating,
        checked_out=checked_out,
        checkout_date=checkout_date,
        due_date=due_date
    )

async def create_books_table():
    """Create the books table in the database if it doesn't exist"""
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await register_vector_async(conn)
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    author VARCHAR(255) NOT NULL,
                    publish_date DATE,
                    isbn VARCHAR(20) UNIQUE,
                    genre VARCHAR(50),
                    pages INTEGER,
                    language VARCHAR(50),
                    publisher VARCHAR(255),
                    description TEXT,
                    rating FLOAT,
                    checked_out BOOLEAN DEFAULT FALSE,
                    checkout_date DATE,
                    due_date DATE,
                    text TEXT NOT NULL
                )
            """)
            
            # Create index on title and author for faster searches
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_books_title ON books (title)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_books_author ON books (author)")
            
        await conn.commit()
        print("Books table created successfully")

async def insert_sample_books(num_books: int = 100):
    """Insert sample books into the database"""
    books = [generate_random_book() for _ in range(num_books)]
    
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        async with conn.cursor() as cur:
            for book in books:
                await cur.execute("""
                    INSERT INTO books 
                    (title, author, publish_date, isbn, genre, pages, language, publisher, description, rating, checked_out, checkout_date, due_date, text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    book.title, book.author, book.publish_date, book.isbn, 
                    book.genre, book.pages, book.language, book.publisher, 
                    book.description, book.rating, book.checked_out, book.checkout_date, book.due_date, book.text
                ))
        await conn.commit()
    
    print(f"Inserted {num_books} sample books into the database")

async def create_books_vectorizer():
    """Create a pgai vectorizer for the books table"""
    try:
        async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
            async with conn.cursor() as cur:
                # Create vectorizer for books table
                await cur.execute("""
                    SELECT ai.create_vectorizer(
                        source => 'books',
                        destination => json_build_object('target_table', 'books_embeddings'),
                        loading => json_build_object('column_name', 'text'),
                        embedding => json_build_object(
                            'model', 'all-minilm',
                            'dimensions', 384,
                            'base_url', 'http://localhost:11434'
                        )
                    )
                """)
            await conn.commit()
        print("Books vectorizer created successfully")
    except Exception as e:
        if "already exists" in str(e):
            print("Vectorizer already exists, skipping creation")
        else:
            print(f"Error creating vectorizer: {e}")

async def check_books_table():
    """Check if the books table has data"""
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT COUNT(*) as count FROM books")
            result = await cur.fetchone()
            return result['count']

async def main():
    # Create books table
    await create_books_table()
    
    # Check if table already has data
    book_count = await check_books_table()
    
    if book_count == 0:
        # Insert sample books if the table is empty
        await insert_sample_books(100)  # Generate 100 sample books
    else:
        print(f"Books table already contains {book_count} records, skipping data generation")
    
    # Create vectorizer for natural language search
    await create_books_vectorizer()
    
    print("Books setup completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())