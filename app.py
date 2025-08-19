import os
import logging
import uuid
import tempfile
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.graph import MixRAGGraph
from src.document_processor import DocumentProcessor
from src.document_splitter import DocumentSplitter
from src.document_retriever import DocumentRetriever
from src.pydantic_models import RAGRequest, FileUploadResponse
from src.config import ConfigManager

# Load configuration with fallback
try:
    config = ConfigManager()._config
except Exception as e:
    print(f"Failed to load config: {e}, using fallback configuration")
    config = {
        'api': {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'host': os.getenv('HOST', '0.0.0.0'),
            'port': int(os.getenv('PORT', '5001')),
            'debug': os.getenv('DEBUG', 'true').lower() == 'true',
            'cors_origins': ["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"]
        },
        'model_config': {
            'response_model': 'gpt-3.5-turbo',
            'grader_model': 'gpt-3.5-turbo',
            'temperature': 0.7
        },
        'document': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_chunks': 100
        },
        'prompts': {
            'GRADE_PROMPT': 'You are a document relevance grader. Your task is to determine if the retrieved documents are relevant to the user\'s question.\n\nQuestion: {question}\nRetrieved Context: {context}\n\nGrade the relevance using a binary score:\n- "yes" if the documents are relevant and can help answer the question\n- "no" if the documents are not relevant or insufficient\n\nBinary Score:',
            'REWRITE_PROMPT': 'You are a question rewriter. The user\'s question was not answered well by the retrieved documents.\n\nOriginal Question: {question}\n\nPlease rewrite the question to be more specific, clear, or focused.\n\nRewritten Question:',
            'GENERATE_PROMPT': 'You are an AI assistant that answers questions based on retrieved document content.\n\nQuestion: {question}\nRetrieved Context: {context}\n\nPlease provide a comprehensive answer based on the context provided. If the context doesn\'t contain enough information to answer the question completely, acknowledge what you can answer and what information is missing.\n\nAnswer:'
        }
    }

# Set USER_AGENT to avoid warnings
os.environ['USER_AGENT'] = 'AgenticRAG/1.0'

# Configure logging
logging.basicConfig(
    level=getattr(logging, config['api']['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG API",
    description="An intelligent RAG API with agentic capabilities using LangGraph",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.get("/")
def home():
    return {
        "message": "Welcome to the Agentic RAG API",
        "version": "2.0.0",
        "environment": config['api']['environment']
    }

# In-memory storage for uploaded files (in production, use database/cloud storage)
uploaded_files = {}

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and get a file ID for use in RAG requests."""
    try:
        # Validate file type
        allowed_types = ["pdf", "docx", "doc", "txt"]
        file_extension = file.filename.split(".")[-1].lower() if "." in file.filename else ""
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Store file content (in production, save to cloud storage)
        uploaded_files[file_id] = {
            "name": file.filename,
            "content": file_content,
            "type": file_extension,
            "size": len(file_content)
        }
        
        logger.info(f"File uploaded successfully: {file.filename} (ID: {file_id})")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            message="File uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """Get information about an uploaded file."""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    return {
        "file_id": file_id,
        "filename": file_info["name"],
        "type": file_info["type"],
        "size": file_info["size"]
    }

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    deleted_file = uploaded_files.pop(file_id)
    logger.info(f"File deleted: {deleted_file['name']} (ID: {file_id})")
    
    return {"message": "File deleted successfully", "filename": deleted_file["name"]}

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    try:
        return {"status": "healthy", "message": "Agentic RAG API is running"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Error: {str(e)}"}

@app.post("/rag")
async def rag(request: RAGRequest):
    """Simple RAG endpoint that processes documents and returns answers."""
    try:
        logger.info(f"Processing RAG request for query: {request.query[:100]}...")
        
        # Process documents
        document_processor = DocumentProcessor()
        
        # Convert file IDs to file content objects
        processed_sources = []
        for source in request.file_paths_urls:
            if isinstance(source, str):
                # Check if it's a file ID (UUID format)
                if len(source) == 36 and source.count('-') == 4:  # Simple UUID check
                    if source in uploaded_files:
                        file_info = uploaded_files[source]
                        processed_sources.append({
                            "name": file_info["name"],
                            "content": file_info["content"],
                            "type": "file"
                        })
                        logger.info(f"Converted file ID {source} to file content")
                    else:
                        logger.warning(f"File ID {source} not found, treating as URL")
                        processed_sources.append(source)
                else:
                    # Treat as URL
                    processed_sources.append(source)
            else:
                # Already a file content object
                processed_sources.append(source)
        
        docs_list = document_processor.load_documents(processed_sources)
        logger.info(f"Loaded {len(docs_list)} documents")
        
        if not docs_list:
            raise HTTPException(status_code=400, detail="No documents could be loaded from the provided sources")
        
        # Split documents
        document_splitter = DocumentSplitter()
        doc_splits = document_splitter.split_documents(docs_list)
        logger.info(f"Split documents into {len(doc_splits)} chunks")
        
        # Create retriever
        document_retriever = DocumentRetriever(doc_splits)
        logger.info("Retriever created successfully")
        
        # Retrieve relevant documents
        retrieved_docs = document_retriever.retrieve_documents(request.query, k=5)
        logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        if not retrieved_docs:
            raise HTTPException(status_code=400, detail="No relevant documents found for the query")
        
        # Initialize RAG graph (with fallback)
        final_answer = ""
        try:
            rag_graph = MixRAGGraph(document_retriever.retriever_tool)
            logger.info("Executing LangGraph workflow")
            
            for event in rag_graph.workflow.stream({
                "messages": [{"role": "user", "content": request.query}]
            }):
                for node, update in event.items():
                    logger.debug(f"Processing node: {node}")
                    if "messages" in update and update["messages"]:
                        for msg in update["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                logger.debug(f"Message from {node}: {msg.content[:100]}...")
                                if node == "generate_answer" or (not final_answer and msg.content.strip()):
                                    final_answer += msg.content
                                    logger.info(f"Found answer from node '{node}': {msg.content[:100]}...")
        except Exception as workflow_error:
            logger.error(f"Workflow error: {workflow_error}")
            final_answer = ""
        
        # Fallback: Generate simple answer if workflow fails
        if not final_answer and retrieved_docs:
            logger.warning("No answer from workflow, creating simple answer")
            context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])
            simple_prompt = f"Based on the following context, answer this question: {request.query}\n\nContext:\n{context}"
            
            try:
                from langchain.chat_models import init_chat_model
                model = init_chat_model(
                    config['model_config']['response_model'], 
                    temperature=config['model_config']['temperature']
                )
                
                simple_response = model.invoke([{"role": "user", "content": simple_prompt}])
                final_answer = simple_response.content
                logger.info(f"Generated simple answer: {final_answer[:100]}...")
            except Exception as e:
                logger.error(f"Error generating simple answer: {e}")
                final_answer = "Based on the retrieved documents, I found relevant information but couldn't generate a complete response due to a technical issue."
        
        # Prepare response data
        response_data = {
            "response": final_answer if final_answer else "No response generated",
            "top_3_retrieved_docs": [doc.page_content for doc in retrieved_docs[:3]],
            "metadata": [doc.metadata for doc in retrieved_docs[:3]]
        }
        
        logger.info("RAG request completed successfully")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing RAG request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    try:
        logger.info("Starting Agentic RAG API server")
        uvicorn.run(
            "app:app", 
            host=config['api']['host'], 
            port=config['api']['port'], 
            reload=config['api']['debug'],
            log_level=config['api']['log_level'].lower()
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        # Fallback to basic uvicorn run
        uvicorn.run(app, host="0.0.0.0", port=5001)
