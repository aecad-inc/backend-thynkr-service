from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum

class ModelType(str, Enum):
    CNN = "cnn"
    GNN = "gnn"
    BERT = "bert"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class IndustryContext(str, Enum):
    RESIDENTIAL = "Residential Development"
    COMMERCIAL = "Commercial Development"
    INFRASTRUCTURE = "Infrastructure"
    TRANSPORTATION = "Transportation"
    UTILITIES = "Utilities"
    INDUSTRIAL = "Industrial"
    MUNICIPAL = "Municipal"
    MIXED_USE = "Mixed-Use"

# === Model Prediction Structures ===
class ModelPrediction(BaseModel):
    model_type: ModelType
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class LayerPredictions(BaseModel):
    layer_name: str
    predictions: List[ModelPrediction]
    geometric_context: Optional[Dict[str, Any]] = None
    
    @validator('predictions')
    def validate_predictions(cls, v):
        if not v:
            raise ValueError('At least one prediction must be provided')
        return v

# === Context Information ===
class ProjectContext(BaseModel):
    project_id: str
    project_type: Optional[IndustryContext] = None
    project_description: Optional[str] = None
    standards: Optional[Dict[str, Any]] = None
    historical_patterns: Optional[Dict[str, Any]] = None

class DocumentContext(BaseModel):
    document_type: str = Field(..., description="Type of document (pdf, dwg, spec, etc.)")
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embeddings: Optional[List[float]] = None

class FileContext(BaseModel):
    file_id: str
    file_name: str
    file_type: str
    project_context: Optional[ProjectContext] = None
    document_contexts: List[DocumentContext] = []
    processing_history: Optional[List[Dict[str, Any]]] = None

# === Enhancement Requests ===
class ConfidenceEnhancementRequest(BaseModel):
    """Request to enhance low-confidence predictions"""
    file_version_id: str
    user_id: str
    layer_predictions: List[LayerPredictions]
    file_context: Optional[FileContext] = None
    enhancement_type: str = Field(default="standard", description="Type of enhancement requested")
    
    class Config:
        schema_extra = {
            "example": {
                "file_version_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user_12345",
                "layer_predictions": [
                    {
                        "layer_name": "CONC_WALK_6IN",
                        "predictions": [
                            {
                                "model_type": "cnn",
                                "predicted_class": "3D-Sidewalk",
                                "confidence": 0.65
                            },
                            {
                                "model_type": "gnn", 
                                "predicted_class": "3D-CONCRETE",
                                "confidence": 0.70
                            }
                        ]
                    }
                ],
                "enhancement_type": "standard"
            }
        }

class ConsensusAnalysisRequest(BaseModel):
    """Request for multi-model consensus analysis"""
    file_version_id: str
    user_id: str
    conflicting_predictions: List[LayerPredictions]
    context: Optional[FileContext] = None
    require_explanation: bool = True

class ContextAnalysisRequest(BaseModel):
    """Request for context-based analysis"""
    document_content: str
    document_type: str
    project_context: Optional[ProjectContext] = None
    analysis_type: str = Field(default="layer_extraction", description="Type of analysis to perform")

# === Interactive Requests ===
class NaturalLanguageQueryRequest(BaseModel):
    """Natural language query about predictions"""
    query: str
    session_id: Optional[str] = None
    file_context: Optional[FileContext] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return v.strip()

class FeedbackRequest(BaseModel):
    """User feedback on predictions"""
    file_version_id: str
    user_id: str
    layer_name: str
    original_prediction: str
    user_correction: str
    confidence_rating: Optional[float] = Field(None, ge=0.0, le=1.0)
    comments: Optional[str] = None

# === Batch Processing ===
class BatchEnhancementRequest(BaseModel):
    """Batch processing request for multiple files"""
    file_version_ids: List[str]
    user_id: str
    processing_options: Dict[str, Any] = {}
    priority: str = Field(default="normal", regex="^(low|normal|high)$")
    
    @validator('file_version_ids')
    def validate_file_ids(cls, v):
        if not v:
            raise ValueError('At least one file ID must be provided')
        if len(v) > 50:  # Reasonable batch limit
            raise ValueError('Maximum 50 files per batch request')
        return v

# === Context Management ===
class ContextUpdateRequest(BaseModel):
    """Update context information"""
    session_id: str
    context_type: str = Field(..., regex="^(session|file|project|global)$")
    context_data: Dict[str, Any]
    merge_strategy: str = Field(default="update", regex="^(replace|update|append)$")

class ContextRetrievalRequest(BaseModel):
    """Retrieve context information"""
    session_id: Optional[str] = None
    file_id: Optional[str] = None
    project_id: Optional[str] = None
    context_types: List[str] = Field(default=["session", "file"])
    include_history: bool = False

# === Advanced Analysis ===
class PatternAnalysisRequest(BaseModel):
    """Request for pattern analysis across projects"""
    user_id: str
    project_ids: Optional[List[str]] = None
    analysis_scope: str = Field(default="user", regex="^(user|project|global)$")
    pattern_types: List[str] = Field(default=["naming", "classification", "geometric"])
    time_range: Optional[Dict[str, str]] = None

class QualityAssessmentRequest(BaseModel):
    """Request for quality assessment"""
    file_version_id: str
    predictions: List[LayerPredictions]
    project_standards: Optional[Dict[str, Any]] = None
    assessment_criteria: List[str] = Field(default=["consistency", "accuracy", "compliance"])