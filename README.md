# Agentic RAG API

A production-ready Retrieval-Augmented Generation (RAG) API with agentic capabilities using LangGraph for intelligent planning, tool selection, and multi-step reasoning.

## About This Project

The Agentic RAG API is an intelligent document processing and question-answering system that combines the power of Large Language Models (LLMs) with advanced retrieval techniques. Unlike traditional RAG systems that simply retrieve and generate responses, this API implements an **agentic workflow** that can reason, plan, and make intelligent decisions about how to process user queries.

### What It Does

The system processes user questions against a collection of documents (PDFs, Word documents, text files, or web pages) and provides intelligent, context-aware answers. It uses a sophisticated multi-step workflow that:

1. **Analyzes the query** to determine the best approach
2. **Retrieves relevant documents** using vector similarity search
3. **Evaluates document relevance** using AI-powered grading
4. **Reformulates queries** when needed for better results
5. **Generates comprehensive answers** based on retrieved context

### How It Works

The core innovation lies in the **LangGraph workflow** that orchestrates multiple AI agents working together:

- **Query Analysis Agent**: Determines whether to retrieve documents or respond directly
- **Document Retriever**: Uses in-memory vector storage and OpenAI embeddings to find relevant content
- **Relevance Grader**: AI-powered assessment of whether retrieved documents actually answer the question
- **Query Rewriter**: Reformulates questions when initial retrieval yields poor results
- **Answer Generator**: Creates comprehensive, well-structured responses

This agentic approach ensures higher quality answers by allowing the system to iterate and improve its retrieval strategy, rather than simply returning the first set of relevant documents it finds.

## 🚀 Features

- **Intelligent Document Processing**: Supports PDF, DOCX, DOC, TXT files and web URLs
- **File Upload API**: RESTful endpoints for file management with unique file IDs
- **Advanced RAG Workflow**: Uses LangGraph for multi-step reasoning and planning
- **Vector Search**: In-memory vector storage with OpenAI embeddings
- **Document Grading**: AI-powered relevance assessment
- **Production Ready**: Environment-based configuration, logging, and security features
- **RESTful API**: FastAPI-based with automatic documentation
- **Docker Support**: Production-ready containerization

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 2_agentic_rag_api_deploy
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Set your OpenAI API key (REQUIRED)
   export OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: Set other environment variables
   export ENVIRONMENT=development
   export LOG_LEVEL=INFO
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5001`

## 🏗️ Architecture

### Project Structure

```
2_agentic_rag_api_deploy/
├── app.py                 # FastAPI application entry point
├── Dockerfile            # Production Docker configuration
├── requirements.txt      # Python dependencies
├── README.md            # This documentation
├── logs/                # Application logs directory
├── uploads/             # File upload directory
└── src/
    ├── __init__.py
    ├── config/
    │   └── config.yaml  # Main configuration file
    ├── config.py        # Configuration manager
    ├── document_processor.py    # Document loading and processing
    ├── document_retriever.py    # Vector search and retrieval
    ├── document_splitter.py     # Text chunking
    ├── graph.py          # LangGraph workflow definition
    ├── graph_nodes.py    # Workflow node implementations
    └── pydantic_models.py # Data models
```

### Core Components

#### 1. **Document Processing Pipeline**
- **DocumentProcessor**: Loads documents from various sources (PDF, DOCX, DOC, TXT, URLs)
- **DocumentSplitter**: Splits documents into chunks for vector storage
- **DocumentRetriever**: Manages vector search and retrieval operations

#### 2. **RAG Workflow (LangGraph)**
- **MixRAGGraph**: Main workflow orchestrator
- **Graph Nodes**: Individual processing steps
  - `generate_query_or_respond`: Decides whether to retrieve or respond
  - `grade_documents`: Assesses relevance of retrieved documents
  - `rewrite_question`: Reformulates queries for better retrieval
  - `generate_answer`: Creates final responses

#### 3. **Vector Storage**
- **In-Memory Vector Store**: Uses LangChain's InMemoryVectorStore for document embeddings
- **OpenAI Embeddings**: Leverages OpenAI's embedding models for semantic search

#### 4. **Configuration Management**
- **ConfigManager**: Loads and manages configuration from YAML
- **Environment Variables**: Override configuration for different environments
- **Centralized Settings**: All settings in `src/config/config.yaml`

## 📚 API Documentation

### Interactive Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:5001/docs
- **ReDoc documentation**: http://localhost:5001/redoc

### Endpoints

#### `GET /`
Welcome endpoint with API information.

**Response:**
```json
{
  "message": "Welcome to the Agentic RAG API",
  "version": "2.0.0",
  "environment": "development"
}
```

#### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "message": "Agentic RAG API is running"
}
```

#### `POST /upload`
Upload a file and get a file ID for use in RAG requests.

**Request:** Multipart form data with file
**Response:**
```json
{
  "file_id": "uuid-string",
  "filename": "document.pdf",
  "message": "File uploaded successfully"
}
```

#### `GET /files/{file_id}`
Get information about an uploaded file.

**Response:**
```json
{
  "file_id": "uuid-string",
  "filename": "document.pdf",
  "type": "pdf",
  "size": 1024
}
```

#### `DELETE /files/{file_id}`
Delete an uploaded file.

**Response:**
```json
{
  "message": "File deleted successfully",
  "filename": "document.pdf"
}
```

#### `POST /rag`
Main RAG processing endpoint.

**Request Body:**
```json
{
  "query": "What is the main topic of the document?",
  "file_paths_urls": [
    "file-uuid-here",
    "https://example.com/document.txt"
  ]
}
```

**Response:**
```json
{
  "response": "The document discusses...",
  "top_3_retrieved_docs": [
    "Retrieved document content 1",
    "Retrieved document content 2",
    "Retrieved document content 3"
  ],
  "metadata": [
    {
      "source": "document.pdf",
      "page": 1
    }
  ]
}
```

### Example Usage

#### File Upload and RAG Processing
```python
import requests

# 1. Upload a file
with open("document.pdf", "rb") as f:
    files = {"file": f}
    upload_response = requests.post("http://localhost:5001/upload", files=files)
    file_id = upload_response.json()["file_id"]

# 2. Process RAG request
rag_response = requests.post("http://localhost:5001/rag", json={
    "query": "What is the main topic of the document?",
    "file_paths_urls": [file_id]
})

result = rag_response.json()
print(f"Answer: {result['response']}")
print(f"Sources: {len(result['top_3_retrieved_docs'])} documents retrieved")
```

#### Direct URL Processing
```python
# Process documents from URLs directly
rag_response = requests.post("http://localhost:5001/rag", json={
    "query": "What is the main topic?",
    "file_paths_urls": ["https://example.com/document.txt"]
})
```

## ⚙️ Configuration

### Configuration File (`src/config/config.yaml`)

The main configuration file contains all settings:

```yaml
# OpenAI Configuration
openai:
  api_key: ""  # Will be overridden by environment variable
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000

# Model Configuration
model_config:
  response_model: "gpt-3.5-turbo"
  grader_model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000

# API Configuration
api:
  environment: "development"
  host: "0.0.0.0"
  port: 5001
  debug: true
  log_level: "INFO"
  cors_origins: ["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"]

# Document Processing
document:
  chunk_size: 1000
  chunk_overlap: 200
  max_chunks: 100

# RAG Configuration
rag:
  top_k: 5
  similarity_threshold: 0.7
  max_context_length: 4000
```

### Environment Variables

Environment variables can override configuration values:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | **Required** |
| `ENVIRONMENT` | Environment mode | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HOST` | API host | `0.0.0.0` |
| `PORT` | API port | `5001` |
| `DEBUG` | Debug mode | `true` |

### Environment Setup

Set your environment variables directly:

```bash
# Required: Set your OpenAI API key
export OPENAI_API_KEY=your_openai_api_key_here

# Optional: Set other environment variables
export ENVIRONMENT=development
export LOG_LEVEL=INFO
export HOST=0.0.0.0
export PORT=5001
export DEBUG=true
```

## 🛠️ Development

### Running in Development Mode

```bash
# Set development environment
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with auto-reload
python app.py
```

### Code Structure

#### Document Processing
```python
# Load documents
processor = DocumentProcessor()
docs = processor.load_documents(["file.pdf", "https://example.com/doc.txt"])

# Split into chunks
splitter = DocumentSplitter()
chunks = splitter.split_documents(docs)

# Create retriever
retriever = DocumentRetriever(chunks)
results = retriever.retriever.get_relevant_documents("query")
```

#### RAG Workflow
```python
# Initialize workflow
rag_graph = MixRAGGraph(retriever.retriever_tool)

# Execute workflow
for event in rag_graph.workflow.stream({
    "messages": [{"role": "user", "content": "query"}]
}):
    # Process workflow events
    pass
```

### Adding New Features

1. **New Document Types**: Add to `document_processor.py`
2. **New Embedding Models**: Update `document_retriever.py`
3. **New Workflow Nodes**: Add to `graph_nodes.py` and update `graph.py`
4. **New API Endpoints**: Add to `app.py`

## 🚀 Production Deployment

### Environment Setup

```bash
# Production environment variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export DEBUG=false
```

### Using Production Script

```bash
# Run with production script
python app.py
```

### Docker Deployment

#### Using Docker directly

```bash
# Build the image
docker build -t agentic-rag-api .

# Run the container
docker run -p 5001:5001 --env-file .env agentic-rag-api

# Run with custom environment variables
docker run -p 5001:5001 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  -e ENVIRONMENT=production \
  -e DEBUG=false \
  agentic-rag-api
```

#### Dockerfile Features:
- Python 3.11 slim base image
- Non-root user for security
- Health checks
- Optimized layer caching
- Minimal image size

### Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### Systemd Service

Create `/etc/systemd/system/agentic-rag-api.service`:

```ini
[Unit]
Description=Agentic RAG API
After=network.target

[Service]
Type=exec
User=rag-api
WorkingDirectory=/opt/agentic-rag-api
Environment=PATH=/opt/agentic-rag-api/venv/bin
ExecStart=/opt/agentic-rag-api/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Monitoring

#### Health Checks
```bash
# Check API health
curl http://localhost:5001/health

# Monitor logs
tail -f logs/app.log
```

#### Performance Monitoring
- Monitor response times
- Check memory usage
- Monitor vector search performance
- Track API error rates

## 🔧 Troubleshooting

### Common Issues

#### 1. OpenAI API Errors
```
Error: Invalid API key
```
**Solution**: Verify your `OPENAI_API_KEY` environment variable is set correctly

```
Error: OPENAI_API_KEY environment variable is required
```
**Solution**: Set the `OPENAI_API_KEY` environment variable before running the application

#### 2. Document Loading Failures
```
Error: File not found
```
**Solution**: Check file paths and permissions

#### 3. Memory Issues
```
Error: Out of memory
```
**Solution**: Reduce chunk size in config or increase system memory

#### 4. CORS Errors
```
Error: CORS policy violation
```
**Solution**: Update `cors_origins` in configuration

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
python app.py
```

### Log Files

- **Application logs**: `logs/app.log`
- **Error logs**: Check console output
- **Access logs**: Available in debug mode

### Performance Optimization

1. **Reduce chunk size** for large documents
2. **Use faster embedding models** for better performance
3. **Implement caching** for repeated queries
4. **Optimize memory usage** for in-memory vector storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Use meaningful commit messages

## 📄 License

[Add your license here]

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the configuration options
- Check the troubleshooting section

## 🔄 Changelog

### Version 2.0.0
- Production-ready configuration
- Environment variable support
- Enhanced logging and monitoring
- Improved error handling
- Comprehensive documentation
- File upload API endpoints
- Docker containerization

### Version 1.0.0
- Initial RAG implementation
- Basic API endpoints
- Document processing pipeline
- LangGraph workflow integration
