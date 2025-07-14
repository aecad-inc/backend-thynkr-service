from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

# === Base Response Models ===
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = True
    timestamp: str = Field(default="2025-01-14T10:00:00Z")
    processing_time_ms: Optional[int] = None

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None

# === Enhancement Response Models ===
class EnhancementResult(BaseModel):
    """Individual layer enhancement result"""
    layer_name: str
    enhanced: bool
    improved: Optional[bool] = None
    original_max_confidence: float
    final_confidence: float
    recommended_classification: Optional[str] = None
    reasoning: List[str] = []
    key_factors: List[str] = []
    alternative_classifications: List[Dict[str, Any]] = []
    improvement_explanation: Optional[str] = None
    uncertainty_factors: List[str] = []
    original_predictions: List[Dict[str, Any]] = []
    enhancement_method: str

class EnhancementSummary(BaseModel):
    """Summary of enhancement processing"""
    total_layers: int
    enhanced_count: int
    skipped_count: int
    improvements: List[Dict[str, Any]] = []
    average_confidence_improvement: Optional[float] = None
    processing_duration_seconds: Optional[float] = None

class EnhancementResponse(BaseResponse):
    """Response for prediction enhancement requests"""
    file_version_id: str
    user_id: str
    enhancement_results: List[EnhancementResult]
    summary: EnhancementSummary
    processing_metadata: Dict[str, Any] = {}

# === Consensus Analysis Response Models ===
class ConsensusResult(BaseModel):
    """Individual consensus analysis result"""
    layer_name: str
    consensus_achieved: bool
    consensus_classification: Optional[str] = None
    consensus_confidence: Optional[float] = None
    agreeing_models: List[str] = []
    resolution_method: str
    resolution_reasoning: List[str] = []
    model_analysis: Dict[str, Any] = {}
    disagreement_factors: List[str] = []

class ConsensusSummary(BaseModel):
    """Summary of consensus analysis"""
    total_analyzed: int
    consensus_achieved: int
    llm_resolutions: int
    model_agreement_rate: Optional[float] = None
    common_disagreement_patterns: List[str] = []

class ConsensusResponse(BaseResponse):
    """Response for consensus analysis requests"""
    file_version_id: str
    user_id: str
    consensus_results: List[ConsensusResult]
    summary: ConsensusSummary
    explanations_provided: bool

# === Natural Language Query Response Models ===
class QueryResponse(BaseResponse):
    """Response for natural language queries"""
    query: str
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: List[str] = []
    related_suggestions: List[str] = []
    session_id: Optional[str] = None
    conversation_context: Optional[Dict[str, Any]] = None

# === Context Analysis Response Models ===
class ContextAnalysisResponse(BaseResponse):
    """Response for document/context analysis"""
    analysis_type: str
    extracted_insights: Dict[str, Any]
    layer_recommendations: List[Dict[str, Any]] = []
    confidence_boosts: List[Dict[str, Any]] = []
    document_summary: Optional[str] = None

# === Batch Processing Response Models ===
class BatchResponse(BaseResponse):
    """Response for batch processing operations"""
    batch_id: str
    status: str = Field(..., regex="^(queued|processing|completed|completed_with_errors|failed)$")
    total_files: int
    processed_files: int = 0
    failed_files: int = 0
    estimated_completion_time: Optional[str] = None
    results_available: bool = False
    results_url: Optional[str] = None
    message: str = ""

# === Learning and Feedback Response Models ===
class FeedbackResponse(BaseResponse):
    """Response for user feedback submission"""
    feedback_id: str
    learning_impact: str = Field(..., regex="^(low|medium|high)$")
    pattern_updated: bool = False
    similar_cases_found: int = 0
    recommendation_adjustments: List[str] = []

# === Pattern Analysis Response Models ===
class PatternInsight(BaseModel):
    """Individual pattern insight"""
    pattern_type: str
    pattern_description: str
    frequency: int
    confidence: float
    examples: List[Dict[str, Any]] = []
    recommendations: List[str] = []

class PatternAnalysisResponse(BaseResponse):
    """Response for pattern analysis requests"""
    analysis_scope: str
    time_range: Optional[Dict[str, str]] = None
    patterns_found: List[PatternInsight]
    trend_analysis: Dict[str, Any] = {}
    actionable_insights: List[str] = []

# === Explanation Response Models ===
class PredictionExplanation(BaseModel):
    """Detailed explanation of a prediction"""
    layer_name: str
    prediction: str
    confidence: float
    explanation: str
    key_factors: List[str]
    confidence_breakdown: Dict[str, float]
    alternative_considerations: List[Dict[str, Any]]
    visual_cues: List[str] = []
    industry_context: Optional[str] = None

class ExplanationResponse(BaseResponse):
    """Response for prediction explanation requests"""
    explanations: List[PredictionExplanation]
    summary_insights: Dict[str, Any] = {}

# === Quality Assessment Response Models ===
class QualityMetric(BaseModel):
    """Individual quality metric"""
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    description: str
    recommendations: List[str] = []

class QualityAssessmentResponse(BaseResponse):
    """Response for quality assessment requests"""
    file_version_id: str
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    quality_metrics: List[QualityMetric]
    compliance_issues: List[Dict[str, Any]] = []
    improvement_suggestions: List[str] = []

# === Advanced Analysis Response Models ===
class SimilarityMatch(BaseModel):
    """Similar pattern match"""
    layer_name: str
    classification: str
    similarity_score: float
    context: Dict[str, Any]
    source_project: Optional[str] = None

class SimilarityAnalysisResponse(BaseResponse):
    """Response for similarity analysis"""
    query_layer: str
    matches_found: List[SimilarityMatch]
    pattern_insights: List[str] = []
    recommendations: List[str] = []

# === Health and Status Response Models ===
class ServiceHealthResponse(BaseResponse):
    """Response for service health checks"""
    service_name: str
    service_version: str
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    components: Dict[str, str] = {}
    performance_metrics: Dict[str, Any] = {}
    uptime_seconds: Optional[int] = None

class CapabilityResponse(BaseResponse):
    """Response describing service capabilities"""
    available_models: List[str]
    supported_file_types: List[str]
    layer_classifications: List[str]
    feature_flags: Dict[str, bool] = {}
    rate_limits: Dict[str, int] = {}

# === Integration Response Models ===
class WebhookResponse(BaseResponse):
    """Response for webhook notifications"""
    webhook_id: str
    event_type: str
    payload: Dict[str, Any]
    delivery_attempts: int = 1
    next_retry: Optional[str] = None

class IntegrationStatusResponse(BaseResponse):
    """Response for integration status"""
    integration_name: str
    status: str
    last_sync: Optional[str] = None
    sync_status: Dict[str, Any] = {}
    error_details: Optional[str] = None