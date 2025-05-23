#!/usr/bin/env python3

import asyncio
import psycopg
from psycopg.rows import dict_row
import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection URL
DB_URL = "postgresql://realagentsuser:realagentspass@localhost:5432/embeddings_db"

# Helper function to generate edition suffixes
def edition_suffix(num):
    """Convert a number to its ordinal representation (1st, 2nd, 3rd, etc.)"""
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
    return f"{num}{suffix} Edition"

# List of real books with titles, authors, and descriptions
REAL_BOOKS = [
    {
        "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "genre": "Fiction",
        "description": "Set in the American South during the 1930s, the story follows young Scout Finch and her father, Atticus, a lawyer who defends a Black man accused of a terrible crime. A classic exploration of racial injustice and moral growth through a child's awakening to good and evil.",
    },
    {
        "title": "1984",
        "author": "George Orwell",
        "genre": "Dystopian Fiction",
        "description": "A chilling portrayal of a totalitarian future where critical thought is suppressed under a surveillance state. The novel introduces concepts like Big Brother, doublethink, and thoughtcrime that have become part of our cultural lexicon.",
    },
    {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "genre": "Romance",
        "description": "A masterful comedy of manners that follows Elizabeth Bennet as she deals with issues of upbringing, marriage, moral rightness, and education in the landed gentry of early 19th-century England.",
    },
    {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "genre": "Fiction",
        "description": "Set in the Jazz Age on Long Island, the novel depicts narrator Nick Carraway's interactions with mysterious millionaire Jay Gatsby and Gatsby's obsession to reunite with his former love, Daisy Buchanan. A classic exploration of decadence, idealism, and the American Dream.",
    },
    {
        "title": "One Hundred Years of Solitude",
        "author": "Gabriel García Márquez",
        "genre": "Magical Realism",
        "description": "The multi-generational story of the Buendía family, whose patriarch, José Arcadio Buendía, founded the fictional town of Macondo. A landmark of magical realism that blends fantasy with reality in the history of Latin America.",
    },
    {
        "title": "Brave New World",
        "author": "Aldous Huxley",
        "genre": "Dystopian Fiction",
        "description": "Set in a futuristic World State, citizens are environmentally engineered into an intelligence-based social hierarchy. The novel anticipates developments in reproductive technology, sleep-learning, psychological manipulation, and classical conditioning.",
    },
    {
        "title": "The Catcher in the Rye",
        "author": "J.D. Salinger",
        "genre": "Fiction",
        "description": "The story of Holden Caulfield, a teenage boy dealing with alienation and loss after being expelled from his prep school. A classic coming-of-age novel that explores themes of identity, belonging, loss, and connection.",
    },
    {
        "title": "The Lord of the Rings",
        "author": "J.R.R. Tolkien",
        "genre": "Fantasy",
        "description": "An epic high-fantasy trilogy that follows hobbit Frodo Baggins as he and the Fellowship embark on a quest to destroy the One Ring, to ensure the destruction of its maker, the Dark Lord Sauron. A foundational text of modern fantasy literature.",
    },
    {
        "title": "Crime and Punishment",
        "author": "Fyodor Dostoevsky",
        "genre": "Psychological Fiction",
        "description": "A novel that explores the mental anguish and moral dilemmas of Rodion Raskolnikov, an impoverished ex-student in Saint Petersburg who formulates a plan to kill an unscrupulous pawnbroker for her money.",
    },
    {
        "title": "The Hobbit",
        "author": "J.R.R. Tolkien",
        "genre": "Fantasy",
        "description": "The precursor to The Lord of the Rings, this beloved children's classic follows the adventures of Bilbo Baggins, a hobbit who is reluctantly swept into an epic quest to reclaim the Lonely Mountain from the dragon Smaug.",
    },
    {
        "title": "Moby-Dick",
        "author": "Herman Melville",
        "genre": "Adventure",
        "description": "The saga of Captain Ahab, who is driven by monomaniacal obsession to hunt and kill the white whale Moby Dick who had bitten off his leg. A profound inquiry into character, faith, and the nature of perception.",
    },
    {
        "title": "War and Peace",
        "author": "Leo Tolstoy",
        "genre": "Historical Fiction",
        "description": "An epic tale of Russian society during the Napoleonic Wars, following the interconnected lives of five aristocratic families. Regarded as one of the most important works of world literature.",
    },
    {
        "title": "The Odyssey",
        "author": "Homer",
        "genre": "Epic Poetry",
        "description": "One of the oldest works of Western literature, this ancient Greek epic poem follows Odysseus, king of Ithaca, on his ten-year journey home after the fall of Troy, while his wife Penelope and son Telemachus fend off suitors.",
    },
    {
        "title": "The Divine Comedy",
        "author": "Dante Alighieri",
        "genre": "Epic Poetry",
        "description": "A long Italian narrative poem that allegorically describes Dante's journey through Hell (Inferno), Purgatory (Purgatorio), and Paradise (Paradiso). Considered the pre-eminent work in Italian literature.",
    },
    {
        "title": "Don Quixote",
        "author": "Miguel de Cervantes",
        "genre": "Novel",
        "description": "Often described as the first modern novel, it follows the adventures of an aging nobleman who, his head bemused by reading chivalric romances, sets out with his squire Sancho Panza to revive chivalry under the name Don Quixote.",
    },
    {
        "title": "Jane Eyre",
        "author": "Charlotte Brontë",
        "genre": "Gothic Fiction",
        "description": "The story of a young woman's journey from a harsh childhood to independence as a governess at Thornfield Hall, where she falls in love with the mysterious Edward Rochester. A pioneering work in feminist literature.",
    },
    {
        "title": "Wuthering Heights",
        "author": "Emily Brontë",
        "genre": "Gothic Fiction",
        "description": "A passionate and dark tale of love and revenge set on the Yorkshire moors, following the turbulent relationship between Catherine Earnshaw and Heathcliff from childhood to their haunting afterlives.",
    },
    {
        "title": "The Count of Monte Cristo",
        "author": "Alexandre Dumas",
        "genre": "Adventure",
        "description": "A tale of betrayal, imprisonment, escape, and elaborate revenge as Edmond Dantès, wrongfully imprisoned, escapes and transforms himself into the wealthy Count of Monte Cristo to take revenge on those responsible for his incarceration.",
    },
    {
        "title": "The Brothers Karamazov",
        "author": "Fyodor Dostoevsky",
        "genre": "Philosophical Fiction",
        "description": "The final novel by Dostoevsky explores ethical debates of God, free will, and morality through the three Karamazov brothers and their troubled relationship with their father, Fyodor Pavlovich.",
    },
    {
        "title": "Frankenstein",
        "author": "Mary Shelley",
        "genre": "Gothic Fiction",
        "description": "Often considered the first true science fiction story, it tells of Victor Frankenstein, a young scientist who creates a sapient creature in an unorthodox scientific experiment, with devastating consequences.",
    },
    {
        "title": "The Picture of Dorian Gray",
        "author": "Oscar Wilde",
        "genre": "Gothic Fiction",
        "description": "The story of a handsome young man whose portrait ages while he remains young and beautiful. As he descends into a life of debauchery, the portrait serves as a reminder of the effect each act has upon his soul.",
    },
    {
        "title": "The Alchemist",
        "author": "Paulo Coelho",
        "genre": "Fiction",
        "description": "A philosophical novel about a young Andalusian shepherd named Santiago who dreams of finding a worldly treasure and embarks on a journey to find it, discovering the importance of listening to one's heart and following one's dreams.",
    },
    {
        "title": "Sapiens: A Brief History of Humankind",
        "author": "Yuval Noah Harari",
        "genre": "Non-fiction",
        "description": "A survey of the history of humankind from the evolution of archaic human species in the Stone Age to the political and technological revolutions of the 21st century, exploring how biology and history have defined what it means to be 'human'.",
    },
    {
        "title": "The Handmaid's Tale",
        "author": "Margaret Atwood",
        "genre": "Dystopian Fiction",
        "description": "Set in a near-future New England, in a strongly patriarchal, totalitarian theonomic state known as the Republic of Gilead, the novel explores themes of women in subjugation and the various means by which they gain agency.",
    },
    {
        "title": "The Hunger Games",
        "author": "Suzanne Collins",
        "genre": "Dystopian Fiction",
        "description": "In a dystopian future, the nation of Panem forces each of its twelve districts to send two teenagers to compete in the Hunger Games: a televised fight to the death. When Katniss Everdeen volunteers to take her sister's place, she is thrust into a battle for survival.",
    },
    {
        "title": "Harry Potter and the Philosopher's Stone",
        "author": "J.K. Rowling",
        "genre": "Fantasy",
        "description": "The first novel in the Harry Potter series follows a young boy who discovers he is a wizard on his 11th birthday and is invited to attend Hogwarts School of Witchcraft and Wizardry, beginning his adventure into a magical world.",
    },
    {
        "title": "The Da Vinci Code",
        "author": "Dan Brown",
        "genre": "Mystery Thriller",
        "description": "A mystery thriller novel that follows symbologist Robert Langdon and cryptologist Sophie Neveu as they investigate a murder in Paris's Louvre Museum and discover a battle between the Priory of Sion and Opus Dei over the possibility of Jesus Christ having been married to Mary Magdalene.",
    },
    {
        "title": "A Brief History of Time",
        "author": "Stephen Hawking",
        "genre": "Non-fiction",
        "description": "A landmark volume in science writing that explains complex concepts in cosmology, from the Big Bang to black holes, in language accessible to the general reader. It explores questions about the origin and fate of the universe.",
    },
    {
        "title": "The Shining",
        "author": "Stephen King",
        "genre": "Horror",
        "description": "A psychological horror novel that follows Jack Torrance, his wife Wendy, and their five-year-old son Danny as they become caretakers of the isolated Overlook Hotel for the winter, where supernatural forces influence Jack's sanity.",
    },
    {
        "title": "The Grapes of Wrath",
        "author": "John Steinbeck",
        "genre": "Fiction",
        "description": "Set during the Great Depression, the novel focuses on the Joads, a poor family of tenant farmers driven from their Oklahoma home by drought, economic hardship, agricultural industry changes, and bank foreclosures. A powerful portrayal of the American Dust Bowl and its impact.",
    },
    {
        "title": "To the Lighthouse",
        "author": "Virginia Woolf",
        "genre": "Modernist Fiction",
        "description": "Using a stream of consciousness narrative, the novel centers on the Ramsay family and their visits to the Isle of Skye in Scotland between 1910 and 1920. It is considered a landmark of modernist literature for its innovative narrative style.",
    },
    {
        "title": "Invisible Man",
        "author": "Ralph Ellison",
        "genre": "Fiction",
        "description": "The novel addresses many of the social and intellectual issues faced by African Americans in the early twentieth century, including black nationalism, racial policies, and identity through the journey of its unnamed narrator.",
    },
    {
        "title": "Ulysses",
        "author": "James Joyce",
        "genre": "Modernist Fiction",
        "description": "Set in Dublin on a single day, June 16, 1904, the novel follows the life and thoughts of Leopold Bloom and other characters through an odyssey of literary styles that parallels Homer's Odyssey. It is considered one of the most important works of modernist literature.",
    },
    {
        "title": "Dune",
        "author": "Frank Herbert",
        "genre": "Science Fiction",
        "description": "Set in the distant future amidst a feudal interstellar society, the novel tells the story of young Paul Atreides, whose family accepts stewardship of the planet Arrakis, the only source of the 'spice' melange, the most valuable substance in the universe.",
    },
    {
        "title": "Foundation",
        "author": "Isaac Asimov",
        "genre": "Science Fiction",
        "description": "The first novel in the Foundation series, it chronicles the fall of the Galactic Empire and mathematician Hari Seldon's attempt to preserve knowledge and shorten the dark age through the establishment of the Foundation.",
    },
    {
        "title": "The Road",
        "author": "Cormac McCarthy",
        "genre": "Post-apocalyptic Fiction",
        "description": "A father and his young son journey across post-apocalyptic America years after an extinction event. The novel explores themes of survival, death, and the human capacity for both good and evil.",
    },
    {
        "title": "Beloved",
        "author": "Toni Morrison",
        "genre": "Historical Fiction",
        "description": "Set after the American Civil War, the novel tells the story of a family of former slaves whose Cincinnati home is haunted by a malevolent spirit. It explores the trauma and psychological legacy of slavery through a powerful blend of realism and the supernatural.",
    },
    {
        "title": "The Name of the Rose",
        "author": "Umberto Eco",
        "genre": "Historical Mystery",
        "description": "Set in an Italian monastery in the year 1327, the novel follows Franciscan friar William of Baskerville and his apprentice Adso of Melk as they investigate a series of suspicious deaths while navigating the complex world of medieval religious politics, philosophy, and intrigue.",
    },
    {
        "title": "Anna Karenina",
        "author": "Leo Tolstoy",
        "genre": "Realist Fiction",
        "description": "The tragic story of married aristocrat Anna Karenina and her affair with the affluent Count Vronsky, set against the backdrop of Russian high society. The novel explores themes of hypocrisy, jealousy, faith, fidelity, family, marriage, and societal norms.",
    },
    {
        "title": "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe",
        "author": "C.S. Lewis",
        "genre": "Fantasy",
        "description": "Four siblings discover a magical wardrobe that transports them to the land of Narnia, where they join forces with the lion Aslan to defeat the White Witch and free Narnia from eternal winter. A beloved children's classic with deep allegorical meanings.",
    },
    {
        "title": "The Little Prince",
        "author": "Antoine de Saint-Exupéry",
        "genre": "Children's Literature",
        "description": "A poetic tale about a young prince who visits Earth from a tiny asteroid after falling in love with a rose. Through his travels, he learns about the narrow-mindedness of adults and the importance of seeing with one's heart rather than just with one's eyes.",
    },
    {
        "title": "Things Fall Apart",
        "author": "Chinua Achebe",
        "genre": "Historical Fiction",
        "description": "The story of Okonkwo, a leader and wrestling champion in a fictional Nigerian village who struggles with the changes brought by British colonial rule and Christian missionaries. A pioneering work in postcolonial literature.",
    },
    {
        "title": "Catch-22",
        "author": "Joseph Heller",
        "genre": "Satirical War Novel",
        "description": "Set during World War II, it follows U.S. Air Force Captain John Yossarian and other airmen based on the island of Pianosa as they attempt to maintain their sanity while fulfilling service requirements so they may return home. Known for its distinctive non-chronological structure and sardonic humor.",
    },
]

async def update_books_with_realistic_data():
    """
    Update existing books in the database with realistic titles, authors, and descriptions
    ensuring each book gets a unique title and no duplicates are created
    """
    logger.info("Starting book database update with realistic data")
    
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Get count of existing books to update
            await cur.execute("SELECT COUNT(*) as count FROM books")
            book_count = (await cur.fetchone())['count']
            logger.info(f"Found {book_count} books to update")
            
            if book_count == 0:
                logger.warning("No books found in database. Please run generate_books.py first.")
                return
            
            # Get all book IDs
            await cur.execute("SELECT id FROM books ORDER BY id")
            book_ids = [row['id'] for row in await cur.fetchall()]
            
            # Create a list of book data that avoids duplicates
            # If we have more books than REAL_BOOKS entries, we'll need to generate unique variants
            all_books_data = []
            
            # First, use all original books from REAL_BOOKS
            original_count = min(len(REAL_BOOKS), book_count)
            all_books_data.extend(REAL_BOOKS[:original_count])
            
            # If we need more books than are in REAL_BOOKS, create variants with edition numbers
            if book_count > len(REAL_BOOKS):
                books_needed = book_count - len(REAL_BOOKS)
                base_idx = 0
                edition = 2  # Start with 2nd edition
                
                while len(all_books_data) < book_count:
                    base_book = REAL_BOOKS[base_idx % len(REAL_BOOKS)]
                    variant = {
                        "title": f"{base_book['title']} ({edition_suffix(edition)})",
                        "author": base_book['author'],
                        "genre": base_book['genre'],
                        "description": f"{base_book['description']} This special edition includes additional commentary and analysis."
                    }
                    all_books_data.append(variant)
                    
                    base_idx += 1
                    if base_idx % len(REAL_BOOKS) == 0:
                        edition += 1
            
            # Update each book with realistic data
            books_updated = 0
            for i, book_id in enumerate(book_ids):
                book_data = all_books_data[i]
                
                # Get current book data
                await cur.execute("""
                    SELECT checked_out, checkout_date, due_date, pages, language, rating, isbn
                    FROM books WHERE id = %s
                """, (book_id,))
                current_book = await cur.fetchone()
                
                if not current_book:
                    logger.warning(f"Book with ID {book_id} not found. Skipping.")
                    continue
                
                # Keep existing pages, language, rating, ISBN, and checkout status
                pages = current_book['pages']
                language = current_book['language']
                rating = current_book['rating']
                isbn = current_book['isbn']
                checked_out = current_book['checked_out']
                checkout_date = current_book['checkout_date']
                due_date = current_book['due_date']
                
                # Generate new text field based on updated data
                checkout_status = f"This book is currently checked out since {checkout_date} and due back on {due_date}." if checked_out else "This book is available for checkout."
                
                new_text = f"""
                    '{book_data['title']}' is a {book_data['genre']} book written by {book_data['author']} and published on 2023-01-01 
                    by Penguin Random House. The book is {pages} pages long and written in {language}. 
                    It has an ISBN of {isbn} and an average rating of {rating}/5. 
                    Book description: {book_data['description']}
                    {checkout_status}
                """.strip()
                
                # Update the book with new title, author, genre, and description
                await cur.execute("""
                    UPDATE books
                    SET title = %s, author = %s, genre = %s, description = %s, text = %s
                    WHERE id = %s
                """, (
                    book_data['title'],
                    book_data['author'],
                    book_data['genre'],
                    book_data['description'],
                    new_text,
                    book_id
                ))
                
                books_updated += 1
                
                # Log progress every 10 books
                if books_updated % 10 == 0:
                    logger.info(f"Updated {books_updated}/{book_count} books")
            
            await conn.commit()
            
            logger.info(f"Successfully updated {books_updated} books with realistic data")

async def main():
    try:
        await update_books_with_realistic_data()
    except Exception as e:
        logger.error(f"Error updating books: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())