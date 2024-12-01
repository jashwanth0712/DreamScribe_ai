from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import random
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO
import logging
from typing import List, Tuple, Optional
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
import time
from functools import wraps
import pinecone

import numpy as np
import os
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import random
from functools import wraps



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(title="AI Content Processing API",
             description="API for content moderation, similarity search, and image comparison")

# Load environment variables
load_dotenv()


# Validate and set environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "object-phrases"


if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize Pinecone client
try:
    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    raise

# Retry decorator for external service calls
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise
                    sleep_time = (backoff_in_seconds * 2 ** x + 
                                random.uniform(0, 1))
                    time.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

# Initialize vector store
def initialize_vector_store():
    try:
        # Check if index exists
        existing_indexes = pinecone_client.list_indexes()
        
        # Only create if it doesn't exist
        if INDEX_NAME not in existing_indexes.names():
            logger.info(f"Creating new index: {INDEX_NAME}")
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            time.sleep(10)  # Wait for index to be ready
        
        logger.info("Initializing embeddings...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            timeout=30,
            max_retries=3
        )
        
        logger.info("Creating vector store...")
        vector_store = Pinecone.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text"
        )
        
        logger.info("Vector store initialized successfully")
        return vector_store
    
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

vector_store = initialize_vector_store()

# Request/Response Models
class PhraseIndexRequest(BaseModel):
    phrases: List[str]

class SimilaritySearchRequest(BaseModel):
    query: str
    top_k: int = 2
    related_k: int = 2

class NSFWCheckRequest(BaseModel):
    input: str

class NSFWCheckResponse(BaseModel):
    is_nsfw: bool
    explanation: str

class ImageSimilarityRequest(BaseModel):
    image1: str
    image2: str

class ImageSimilarityResponse(BaseModel):
    similarity: float

# Vector store operations
@retry_with_backoff()
async def index_phrases(phrases: List[str]):
    """Index phrases into Pinecone with metadata."""
    if not phrases:
        raise ValueError("No phrases provided for indexing")
    
    try:
        # Add phrases with metadata
        vector_store.add_texts(
            texts=phrases,
            metadatas=[{"text": phrase, "timestamp": time.time()} for phrase in phrases]
        )
        return {"message": f"Successfully indexed {len(phrases)} phrases"}
    except Exception as e:
        logger.error(f"Failed to index phrases: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to index phrases")

@retry_with_backoff()
async def search_similar_phrases(query: str, top_k: int = 2, related_k: int = 2) -> dict:
    """Perform similarity search for the given query."""
    try:
        results = vector_store.similarity_search(
            query,
            k=top_k + related_k
        )
        
        if not results:
            return {
                "primary_results": [],
                "related_results": [],
                "query": query
            }
        
        primary_results = [result.metadata.get('text', '') for result in results[:top_k]]
        related_results = [result.metadata.get('text', '') for result in results[top_k:]]
        
        return {
            "primary_results": primary_results,
            "related_results": related_results,
            "query": query
        }
    except Exception as e:
        logger.error(f"Failed to perform similarity search: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to perform similarity search")

# NSFW content check
@retry_with_backoff()
async def check_nsfw_content(input: str) -> Tuple[bool, str]:
    """Check if content is NSFW using OpenAI."""
    try:
        llm = OpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            max_retries=3,
            timeout=30
        )
        
        prompt = (
            "You are an expert content moderation AI. Analyze the given input "
            "and determine if it contains NSFW content. NSFW content includes "
            "offensive language, slurs, or explicit content of racial, gender, "
            "sexual, ethnic, caste, or religious nature.\n\n"
            f"Input: {input}\n\n"
            "Respond with:\n"
            "True -> [explanation] if NSFW\n"
            "False -> None if safe"
        )
        
        response = llm.predict(prompt).strip()
        
        if response.startswith("True"):
            explanation = response[len("True -> "):].strip()
            return True, explanation
        elif response.startswith("False"):
            return False, "None"
        else:
            raise ValueError(f"Unexpected LLM response format: {response}")
            
    except Exception as e:
        logger.error(f"Failed to check NSFW content: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check content")

# Image processing
def process_image(base64_str: str) -> np.ndarray:
    """Process base64 encoded image to numpy array."""
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        img = img.convert("L")  # Convert to grayscale
        return np.array(img)
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        raise ValueError("Invalid image data")

def calculate_image_similarity(img1_base64: str, img2_base64: str) -> float:
    """Calculate similarity between two base64 encoded images."""
    try:
        img1_array = process_image(img1_base64)
        img2_array = process_image(img2_base64)
        
        if img1_array.shape != img2_array.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to binary (threshold at 128)
        img1_bin = img1_array >= 128
        img2_bin = img2_array >= 128
        
        # Calculate similarity
        matching_pixels = np.sum(img1_bin == img2_bin)
        total_pixels = img1_bin.size
        
        return float(matching_pixels / total_pixels * 100)
    except Exception as e:
        logger.error(f"Failed to calculate image similarity: {str(e)}")
        raise

# API Endpoints
# Then update the indexing function to use INDEX_NAME
@app.post("/index_phrases")
async def api_index_phrases(request: PhraseIndexRequest):
    """Endpoint to index phrases."""
    try:
        if not request.phrases:
            raise HTTPException(status_code=400, detail="No phrases provided")
        
        logger.info(f"Starting indexing of {len(request.phrases)} phrases")
        
        # Generate embeddings directly first to verify OpenAI connection
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                timeout=30,
                max_retries=3
            )
            
            # Test embedding generation
            test_embedding = await embeddings.aembed_query(request.phrases[0])
            logger.info(f"Successfully generated test embedding of size: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}"
            )
        
        # Prepare metadata
        texts_with_metadata = []
        for i, phrase in enumerate(request.phrases):
            metadata = {
                "text": phrase,
                "id": f"phrase_{i}",
                "timestamp": str(time.time())
            }
            texts_with_metadata.append((phrase, metadata))
        
        logger.info("Attempting to add texts to vector store...")
        
        try:
            ids = [f"id_{i}" for i in range(len(request.phrases))]
            embeddings_list = await embeddings.aembed_documents(request.phrases)
            
            # Get the Pinecone index directly
            index = pinecone_client.Index(INDEX_NAME)  # Use the constant here
            
            # Upsert directly to Pinecone
            vectors_to_upsert = []
            for i, (text, emb) in enumerate(zip(request.phrases, embeddings_list)):
                vector_data = {
                    "id": ids[i],
                    "values": emb,
                    "metadata": {"text": text}
                }
                vectors_to_upsert.append(vector_data)
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                index.upsert(vectors=batch)
                logger.info(f"Upserted batch of {len(batch)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to add texts to vector store: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store vectors: {str(e)}"
            )
        
        # Verify storage
        try:
            # Query the first phrase to verify storage
            query_response = index.query(
                vector=embeddings_list[0],
                top_k=1,
                include_metadata=True
            )
            
            if query_response.matches:
                logger.info("Successfully verified data storage")
                verification_result = query_response.matches[0].metadata.get('text')
            else:
                logger.warning("No matches found in verification query")
                verification_result = None
                
        except Exception as e:
            logger.error(f"Verification query failed: {str(e)}")
            verification_result = None
        
        return {
            "message": f"Processed {len(request.phrases)} phrases",
            "verification": {
                "first_phrase_found": bool(verification_result),
                "sample_result": verification_result,
                "phrases_processed": len(request.phrases),
                "embeddings_generated": len(embeddings_list)
            }
        }
        
    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index phrases: {str(e)}"
        )

# Update the vector store contents endpoint too
@app.get("/vector_store_contents")
async def check_vector_store():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        index = pinecone_client.Index(INDEX_NAME)  # Use the constant here
        
        # Get index stats
        stats = index.describe_index_stats()
        
        # Get a sample of vectors
        sample_query = await embeddings.aembed_query("test query")
        sample_results = index.query(
            vector=sample_query,
            top_k=5,
            include_metadata=True
        )
        
        return {
            "index_stats": stats,
            "sample_results": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in sample_results.matches
            ] if sample_results.matches else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/search_phrases")
async def api_search_phrases(request: SimilaritySearchRequest):
    try:
        logger.info(f"Searching for query: {request.query}")
        
        # Get embeddings for the query
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        query_embedding = await embeddings.aembed_query(request.query)
        
        # Get the index
        index = pinecone_client.Index(INDEX_NAME)
        
        # Search directly with Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=request.top_k + request.related_k,
            include_metadata=True
        )
        
        # Process results
        primary_results = []
        related_results = []
        
        for i, match in enumerate(search_results.matches):
            result = {
                "text": match.metadata.get("text", "No text found"),
                "score": float(match.score)
            }
            
            if i < request.top_k:
                primary_results.append(result)
            else:
                related_results.append(result)
        
        return {
            "primary_results": primary_results,
            "related_results": related_results,
            "query": request.query,
            "total_matches": len(search_results.matches)
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/check_nsfw", response_model=NSFWCheckResponse)
async def api_check_nsfw(request: NSFWCheckRequest):
    """Endpoint to check for NSFW content."""
    is_nsfw, explanation = await check_nsfw_content(request.input)
    return NSFWCheckResponse(is_nsfw=is_nsfw, explanation=explanation)

@app.get("/index_status")
async def check_index_status():
    try:
        # Get index statistics
        index = pinecone_client.Index(index_name)
        stats = index.describe_index_stats()
        
        # Try to get a few samples if any exist
        sample_results = vector_store.similarity_search(
            "test",
            k=5
        )
        
        return {
            "index_stats": stats,
            "sample_entries": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in sample_results
            ] if sample_results else []
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed to get index status"
        }

@app.post("/image_similarity", response_model=ImageSimilarityResponse)
async def api_image_similarity(request: ImageSimilarityRequest):
    """Endpoint to calculate image similarity."""
    try:
        similarity = calculate_image_similarity(request.image1, request.image2)
        return ImageSimilarityResponse(similarity=similarity)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process images")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/check_vectors")
async def check_vectors():
    try:
        # Get embeddings for a test query
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        test_embedding = await embeddings.aembed_query("test")
        
        # Get the index
        index = pinecone_client.Index(INDEX_NAME)
        
        # Get index stats
        stats = index.describe_index_stats()
        
        # Get a sample of vectors
        results = index.query(
            vector=test_embedding,
            top_k=5,
            include_metadata=True
        )
        
        return {
            "index_stats": stats,
            "vector_count": stats.get("total_vector_count", 0),
            "sample_results": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                } for match in results.matches
            ] if results.matches else []
        }
    except Exception as e:
        return {"error": str(e)}

# Error handling for the entire application
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {str(exc)}")
    return {"error": "An unexpected error occurred", "detail": str(exc)}