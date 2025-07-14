import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Service Configuration
    SERVICE_NAME: str = "backend-thynkr-service"
    SERVICE_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # AWS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # Bedrock Configuration
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", "us-east-1")
    BEDROCK_MODEL_CLAUDE: str = os.getenv("BEDROCK_MODEL_CLAUDE", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    BEDROCK_MODEL_TITAN: str = os.getenv("BEDROCK_MODEL_TITAN", "amazon.titan-embed-text-v1")
    
    # LLM Configuration
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    
    # ChromaDB Configuration
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "aecad_context")
    
    # Redis Configuration (Updated for aioredis)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
    REDIS_RETRY_ON_TIMEOUT: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    
    # Context Management
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "50000"))  # characters
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "10"))  # number of interactions
    
    # Service URLs
    FILE_MGMT_SERVICE_URL: str = os.getenv("FILE_MGMT_SERVICE_URL", "http://backend-filemgmt-service.prod.svc.cluster.local:80")
    CORE_SERVICE_URL: str = os.getenv("CORE_SERVICE_URL", "http://backend-core-service.prod.svc.cluster.local:80")
    
    # Confidence Thresholds
    LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.7"))
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.9"))
    CONSENSUS_THRESHOLD: float = float(os.getenv("CONSENSUS_THRESHOLD", "0.6"))
    
    # Layer Classification Labels
    LAYER_LABELS: List[str] = [
        "2D-Drainage Easement", "2D-Easement", "2D-EC Ditch", "2D-Limits of Disturbance",
        "2D-Lot Lines", "2D-PARKING LINES", "2D-PIPE", "2D-Right of Way",
        "2D-Setback Lines", "2D-Sewer", "2D-Sewer Easement", "2D-Sidewalk",
        "2D-Silt Fence", "2D-Storm", "2D-Tree Fence", "2D-WALL", "2D-Water",
        "2D-WORKLINE", "3D-BACK OF CURB", "3D-BREAK LINE", "3D-BREAKLINE",
        "3D-BUILDING PAD", "3D-CONCRETE", "3D-Contours", "3D-Crown",
        "3D-EC Basin Limits", "3D-EC Contours", "3D-Edge of Pavement",
        "3D-Existing Crown", "3D-Face of Curb", "3D-Model Limit",
        "3D-Retaining Wall", "3D-Shoulder", "3D-Sidewalk", "Other"
    ]
    
    # Industry Context Categories
    INDUSTRY_CONTEXTS: List[str] = [
        "Residential Development", "Commercial Development", "Infrastructure",
        "Transportation", "Utilities", "Industrial", "Municipal", "Mixed-Use"
    ]
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()