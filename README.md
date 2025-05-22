# Library Management System with Vector Search

A modern library management system that leverages vector embeddings in PostgreSQL for natural language search capabilities across book collections.

## Overview

This application provides a complete API for managing books in a library, including advanced search functionality using vector embeddings. The system allows users to find books using natural language queries, check books in and out, and manage the library inventory.

## Features

- **Vector-Based Natural Language Search**: Find books using semantic search queries
- **Complete Book Management**: Add, update, delete, and retrieve book details
- **Checkout System**: Track book availability, checkout dates, and due dates
- **RAG Integration**: Generate natural language responses about books using retrieval-augmented generation
- **API-First Design**: RESTful endpoints for all functionality

## Technical Architecture

### Components

1. **FastAPI Application** (`with_books.py`): Provides the API endpoints and core functionality
2. **Data Generation** (`generate_books.py`): Creates and populates the database with sample books
3. **PostgreSQL Database**: Stores book data and vector embeddings
4. **pgAI & pgvector**: PostgreSQL extensions for vector operations and AI features
5. **Ollama**: Provides embedding models for vector generation

### Data Model

The system uses a `books` table with the following structure:

- `id`: Primary key
- `title`: Book title
- `author`: Book author
- `publish_date`: Publication date
- `isbn`: International Standard Book Number (unique)
- `genre`: Book genre
- `pages`: Number of pages
- `language`: Book language
- `publisher`: Publisher name
- `description`: Book description
- `rating`: Average rating (0-5)
- `checked_out`: Boolean indicating checkout status
- `checkout_date`: Date when book was checked out (if applicable)
- `due_date`: Date when book is due to be returned (if applicable)
- `text`: Concatenated text field containing all book metadata for embedding

## Vector Search Implementation

This system implements vector embeddings for natural language search using the following components:

1. **pgvector Extension**: PostgreSQL extension that adds vector data types and similarity search
2. **pgAI Vectorizer**: Creates and manages vector embeddings for book data
3. **Text Embedding**: All book metadata is concatenated into a `text` field that is embedded
4. **Similarity Search**: Queries are embedded and compared against book embeddings using cosine similarity
5. **Async Processing**: Background workers process vectorization tasks

### How Vector Search Works

1. A user submits a natural language query like "science fiction books about time travel"
2. The query is embedded into a vector using the same model as the books
3. A vector similarity search finds books with the closest matching embeddings
4. Results are returned sorted by relevance (shortest vector distance)

## API Endpoints

### Book Management

- `GET /books`: List all books with pagination and filtering
- `GET /books/{book_id}`: Get details for a specific book
- `POST /books/create`: Add a new book to the library
- `PUT /books/{book_id}`: Update an existing book's information
- `POST /books/update`: Alternative endpoint to update books using request body
- `DELETE /books/{book_id}`: Remove a book from the database
- `POST /books/delete`: Alternative endpoint to delete books using request body

### Checkout Operations

- `POST /books/checkout`: Check out a book for a specified duration
- `POST /books/checkin`: Return a previously checked out book

### Search Capabilities

- `GET /search`: Vector similarity search for books
- `GET /rag`: Generate natural language responses about books using RAG

### System Operations

- `GET /vectorizer_status`: Check the status of the vectorization processes

## Setup and Usage

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- Ollama for embedding models

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install fastapi pgai psycopg pgvector ollama datasets numpy
   ```
3. Configure the database connection in `.env` or directly in the scripts
4. Run the data generation script:
   ```
   python generate_books.py
   ```
5. Start the API server:
   ```
   ./start_server.sh
   ```

### Quick Start

To start the server with automatic dependency checks:

```bash
chmod +x start_server.sh
./start_server.sh
```

## Implementation Details

### `generate_books.py`

This script is responsible for:

1. Creating the books table with proper schema
2. Generating sample book data with realistic metadata
3. Setting up vectorization for the books table
4. Handling both initial setup and incremental updates

Key functions:
- `create_books_table()`: Sets up the database schema
- `insert_sample_books()`: Populates the database with sample books
- `create_books_vectorizer()`: Configures the vectorizer for books
- `generate_random_book()`: Creates realistic book metadata

### `with_books.py`

The main application that provides:

1. FastAPI endpoints for all book operations
2. Connection pooling for database operations
3. Vector search implementation
4. RAG integration for natural language responses

Key components:
- `_find_relevant_chunks()`: Performs vector similarity search
- `BooksSearchResult`: Data class matching the book schema
- Request/response models for all operations
- Error handling and validation

## Example Usage

### Vector Search for Books

```bash
curl "http://localhost:3066/search?query=science%20fiction%20about%20space%20exploration"
```

### Check Out a Book

```bash
curl -X POST "http://localhost:3066/books/checkout" \
  -H "Content-Type: application/json" \
  -d '{"book_id": 42, "days": 14}'
```

### Add a New Book

```bash
curl -X POST "http://localhost:3066/books/create" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "The Vector Database Guide",
    "author": "Jane Smith",
    "publish_date": "2025-01-15",
    "isbn": "978-1-234567-89-0",
    "genre": "Technical",
    "pages": 320,
    "publisher": "Data Press",
    "description": "A comprehensive guide to vector databases and their applications."
  }'
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
