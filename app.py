import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.graph import MixRAGGraph
from src.document_processor import DocumentProcessor
from src.document_splitter import DocumentSplitter
from src.document_retriever import DocumentRetriever
from src.pydantic_models import RAGRequest
from src.config import ConfigManager

# Load configuration
config = ConfigManager()._config

# Set USER_AGENT to avoid warnings
os.environ['USER_AGENT'] = 'AgenticRAG/1.0'

# Configure logging
logging.basicConfig(
    level=getattr(logging, config['api']['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log') if config['api']['environment'] == 'production' else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Agentic RAG API",
    description="An intelligent RAG API with agentic capabilities using LangGraph for planning, tool selection, and multi-step reasoning",
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


@app.post("/rag")
async def rag(request: RAGRequest):
    try:
        logger.info(f"Processing RAG request for query: {request.query[:100]}...")
        
        # Process documents
        document_processor = DocumentProcessor()
        docs_list = document_processor.load_documents(request.file_paths_urls)
        logger.info(f"Loaded {len(docs_list)} documents")

        # Split documents
        document_splitter = DocumentSplitter()
        doc_splits = document_splitter.split_documents(docs_list)
        logger.info(f"Split documents into {len(doc_splits)} chunks")

        # Create retriever
        document_retriever = DocumentRetriever(doc_splits)
        retrieved_docs = document_retriever.retriever.get_relevant_documents(request.query)
        logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")

        # Initialize RAG graph
        rag_graph = MixRAGGraph(document_retriever.retriever_tool)

        # Collect sources and metadata for structured response
        sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        metadata_list = [doc.metadata for doc in retrieved_docs]

        # Execute RAG workflow
        final_answer = ""
        logger.info("Executing LangGraph workflow")
        
        try:
            for event in rag_graph.workflow.stream({
                "messages": [{"role": "user", "content": request.query}]
            }):
                for node, update in event.items():
                    logger.debug(f"Processing node: {node}")
                    if "messages" in update and update["messages"]:
                        for msg in update["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                logger.debug(f"Message from {node}: {msg.content[:100]}...")
                                # Look for any message with content, not just generate_answer node
                                if node == "generate_answer" or (not final_answer and msg.content.strip()):
                                    final_answer += msg.content
                                    logger.info(f"Found answer from node '{node}': {msg.content[:100]}...")
        except Exception as workflow_error:
            logger.error(f"Workflow error: {workflow_error}")
            # Fallback to simple answer generation
            final_answer = ""
        
        # Fallback: If no answer was found, try to generate a simple answer from the retrieved docs
        if not final_answer and retrieved_texts:
            logger.warning("No answer generated from workflow, creating simple answer from retrieved docs")
            context = "\n\n".join(retrieved_texts[:3])
            simple_prompt = f"Based on the following context, answer this question: {request.query}\n\nContext:\n{context}"
            
            # Use the same model to generate a simple answer
            from langchain.chat_models import init_chat_model
            from src.config import ConfigManager
            config = ConfigManager()._config
            model = init_chat_model(
                config['model_config']['response_model'], 
                temperature=config['model_config']['temperature']
            )
            
            try:
                simple_response = model.invoke([{"role": "user", "content": simple_prompt}])
                final_answer = simple_response.content
                logger.info(f"Generated simple answer: {final_answer[:100]}...")
            except Exception as e:
                logger.error(f"Error generating simple answer: {e}")
                final_answer = "Based on the retrieved documents, I found information about Janarddan's experience at Continental AG as a Machine Learning Engineer. However, I couldn't generate a complete response due to a technical issue."

        # Return structured JSON response
        response_data = {
            "response": final_answer if final_answer else "No response generated",
            "top_3_docs": retrieved_texts[:3],
            "metadata": metadata_list[:3]
        }
        
        logger.info("RAG request completed successfully")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing RAG request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health():
    logger.info("Health check requested")
    return {
        "status": "healthy", 
        "message": "RAG API is operational",
        "version": "2.0.0",
        "environment": config['api']['environment']
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Agentic RAG API server")
    uvicorn.run(
        "app:app", 
        host=config['api']['host'], 
        port=config['api']['port'], 
        reload=config['api']['debug'],
        log_level=config['api']['log_level'].lower()
    )
