from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import api_router
from app.core.context_manager import ContextManager
from app.core.bedrock_client import BedrockClient
from app.utils.vector_store import VectorStore
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Backend Thynkr Service",
    description="LLM-powered intelligence service for AECAD prediction enhancement and context-aware analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
context_manager = None
bedrock_client = None
vector_store = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global context_manager, bedrock_client, vector_store
    
    logging.info("🚀 Starting Backend Thynkr Service...")
    
    try:
        # Initialize Bedrock client
        bedrock_client = BedrockClient()
        await bedrock_client.initialize()
        
        # Initialize vector store
        vector_store = VectorStore()
        await vector_store.initialize()
        
        # Initialize context manager
        context_manager = ContextManager(vector_store=vector_store)
        await context_manager.initialize()
        
        # Make instances available globally
        app.state.bedrock_client = bedrock_client
        app.state.context_manager = context_manager
        app.state.vector_store = vector_store
        
        logging.info("✅ All services initialized successfully")
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logging.info("🛑 Shutting down Backend Thynkr Service...")
    
    try:
        if context_manager:
            await context_manager.cleanup()
        if vector_store:
            await vector_store.cleanup()
        if bedrock_client:
            await bedrock_client.cleanup()
            
        logging.info("✅ Cleanup completed successfully")
    except Exception as e:
        logging.error(f"❌ Error during cleanup: {e}")

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Backend Thynkr Service",
        "status": "healthy",
        "version": "1.0.0",
        "description": "LLM-powered intelligence for AECAD"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check service health
        bedrock_healthy = hasattr(app.state, 'bedrock_client') and app.state.bedrock_client is not None
        context_healthy = hasattr(app.state, 'context_manager') and app.state.context_manager is not None
        vector_healthy = hasattr(app.state, 'vector_store') and app.state.vector_store is not None
        
        return {
            "status": "healthy" if all([bedrock_healthy, context_healthy, vector_healthy]) else "degraded",
            "services": {
                "bedrock": "healthy" if bedrock_healthy else "unhealthy",
                "context_manager": "healthy" if context_healthy else "unhealthy", 
                "vector_store": "healthy" if vector_healthy else "unhealthy"
            },
            "timestamp": "2025-01-14T10:00:00Z"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-01-14T10:00:00Z"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)