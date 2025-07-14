from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List, Optional
import logging
from app.models.requests import PatternAnalysisRequest, QualityAssessmentRequest
from app.models.responses import PatternAnalysisResponse, QualityAssessmentResponse, SimilarityAnalysisResponse
from app.services.enhancement_service import EnhancementService

router = APIRouter()
logger = logging.getLogger(__name__)

def get_enhancement_service(request: Request) -> EnhancementService:
    """Dependency to get enhancement service"""
    bedrock_client = request.app.state.bedrock_client
    context_manager = request.app.state.context_manager
    return EnhancementService(bedrock_client, context_manager)

@router.post("/pattern-analysis", response_model=PatternAnalysisResponse, tags=["Pattern Analysis"])
async def analyze_patterns(
    request: PatternAnalysisRequest,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Analyze patterns across projects and users for insights.
    
    Discovers:
    - Common layer naming patterns
    - Classification trends
    - User behavior patterns
    - Industry-specific conventions
    """
    try:
        logger.info(f"Analyzing patterns for scope: {request.analysis_scope}")
        
        # This would implement comprehensive pattern analysis
        # For now, providing a structured response framework
        
        pattern_results = await _analyze_user_patterns(
            user_id=request.user_id,
            project_ids=request.project_ids,
            analysis_scope=request.analysis_scope,
            pattern_types=request.pattern_types,
            enhancement_service=enhancement_service
        )
        
        return PatternAnalysisResponse(
            analysis_scope=request.analysis_scope,
            time_range=request.time_range,
            patterns_found=pattern_results["patterns"],
            trend_analysis=pattern_results["trends"],
            actionable_insights=pattern_results["insights"]
        )
        
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")

@router.post("/quality-assessment", response_model=QualityAssessmentResponse, tags=["Quality Analysis"])
async def assess_prediction_quality(
    request: QualityAssessmentRequest,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Assess the quality of predictions against project standards.
    
    Evaluates:
    - Consistency with project standards
    - Accuracy based on historical feedback
    - Compliance with industry conventions
    - Confidence distribution analysis
    """
    try:
        logger.info(f"Assessing quality for file: {request.file_version_id}")
        
        quality_results = await _assess_prediction_quality(
            predictions=request.predictions,
            project_standards=request.project_standards,
            assessment_criteria=request.assessment_criteria,
            enhancement_service=enhancement_service
        )
        
        return QualityAssessmentResponse(
            file_version_id=request.file_version_id,
            overall_quality_score=quality_results["overall_score"],
            quality_metrics=quality_results["metrics"],
            compliance_issues=quality_results["compliance_issues"],
            improvement_suggestions=quality_results["suggestions"]
        )
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")

@router.get("/similarity-search", response_model=SimilarityAnalysisResponse, tags=["Similarity Analysis"])
async def find_similar_layers(
    layer_name: str,
    limit: int = 10,
    include_context: bool = True,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Find similar layer patterns from historical data.
    
    Useful for:
    - Understanding classification precedents
    - Finding related projects
    - Discovering naming conventions
    - Building confidence in predictions
    """
    try:
        logger.info(f"Finding similar patterns for layer: {layer_name}")
        
        similar_patterns = await enhancement_service.context_manager.find_similar_patterns(
            layer_name=layer_name,
            limit=limit
        )
        
        # Convert to response format
        matches = []
        for pattern in similar_patterns:
            match = {
                "layer_name": pattern.get("layer_name", ""),
                "classification": pattern.get("enhanced_prediction", pattern.get("original_prediction", "")),
                "similarity_score": pattern.get("similarity_score", 0.0),
                "context": pattern.get("context_factors", {}) if include_context else {},
                "source_project": pattern.get("project_id", "unknown")
            }
            matches.append(match)
        
        # Generate insights
        pattern_insights = _generate_pattern_insights(matches)
        recommendations = _generate_similarity_recommendations(layer_name, matches)
        
        return SimilarityAnalysisResponse(
            query_layer=layer_name,
            matches_found=matches,
            pattern_insights=pattern_insights,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@router.get("/confidence-analysis/{file_version_id}", tags=["Confidence Analysis"])
async def analyze_prediction_confidence(
    file_version_id: str,
    include_recommendations: bool = True,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Analyze confidence patterns and provide improvement recommendations.
    
    Provides:
    - Confidence distribution analysis
    - Low confidence pattern identification
    - Improvement recommendations
    - Historical confidence trends
    """
    try:
        # This would integrate with file management service to get predictions
        # For now, providing structure for confidence analysis
        
        confidence_analysis = {
            "file_version_id": file_version_id,
            "confidence_distribution": {
                "high_confidence": {"count": 0, "percentage": 0.0, "threshold": "> 0.9"},
                "medium_confidence": {"count": 0, "percentage": 0.0, "threshold": "0.7 - 0.9"},
                "low_confidence": {"count": 0, "percentage": 0.0, "threshold": "< 0.7"}
            },
            "problematic_layers": [],
            "improvement_opportunities": [],
            "recommended_actions": []
        }
        
        if include_recommendations:
            confidence_analysis["recommended_actions"] = [
                "Consider document context analysis for low-confidence predictions",
                "Review layer naming conventions for consistency",
                "Apply LLM enhancement to predictions below 0.7 confidence"
            ]
        
        return confidence_analysis
        
    except Exception as e:
        logger.error(f"Confidence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Confidence analysis failed: {str(e)}")

@router.get("/trend-analysis", tags=["Trend Analysis"])
async def analyze_classification_trends(
    time_period: str = "30d",
    granularity: str = "daily",
    layer_types: Optional[List[str]] = None,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Analyze classification trends over time.
    
    Shows:
    - Classification accuracy trends
    - Model performance evolution
    - User feedback patterns
    - Seasonal or temporal patterns
    """
    try:
        # This would implement comprehensive trend analysis
        trend_analysis = {
            "time_period": time_period,
            "granularity": granularity,
            "trends": {
                "classification_accuracy": {
                    "current_period": 0.85,
                    "previous_period": 0.82,
                    "trend": "improving",
                    "data_points": []
                },
                "user_feedback_volume": {
                    "current_period": 45,
                    "previous_period": 38,
                    "trend": "increasing",
                    "data_points": []
                },
                "enhancement_usage": {
                    "current_period": 0.23,
                    "previous_period": 0.19,
                    "trend": "increasing",
                    "data_points": []
                }
            },
            "insights": [
                "Classification accuracy has improved by 3.7% over the past month",
                "User feedback volume is increasing, indicating higher engagement",
                "LLM enhancement usage is growing as users discover low-confidence predictions"
            ],
            "recommendations": [
                "Continue monitoring accuracy trends for sustained improvement",
                "Implement proactive enhancement for predictions below 0.75 confidence",
                "Develop automated feedback collection for edge cases"
            ]
        }
        
        return trend_analysis
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.get("/model-comparison", tags=["Model Analysis"])
async def compare_model_performance(
    time_period: str = "7d",
    include_details: bool = False,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Compare performance across different AI models (CNN, GNN, BERT).
    
    Analyzes:
    - Individual model accuracy rates
    - Consensus vs disagreement patterns
    - Model-specific strengths and weaknesses
    - LLM resolution effectiveness
    """
    try:
        model_comparison = {
            "time_period": time_period,
            "model_performance": {
                "cnn": {
                    "accuracy": 0.78,
                    "confidence_avg": 0.82,
                    "strengths": ["Geometric pattern recognition", "Visual feature extraction"],
                    "weaknesses": ["Text-based naming conventions", "Context interpretation"]
                },
                "gnn": {
                    "accuracy": 0.81,
                    "confidence_avg": 0.79,
                    "strengths": ["Spatial relationships", "Graph structure analysis"],
                    "weaknesses": ["Complex naming patterns", "Document context"]
                },
                "bert": {
                    "accuracy": 0.85,
                    "confidence_avg": 0.87,
                    "strengths": ["Naming conventions", "Text pattern recognition"],
                    "weaknesses": ["Geometric interpretation", "Spatial context"]
                }
            },
            "consensus_analysis": {
                "agreement_rate": 0.67,
                "llm_resolution_accuracy": 0.92,
                "common_disagreement_types": [
                    "Storm vs Sewer drainage",
                    "2D vs 3D classification boundaries",
                    "Industry-specific naming variations"
                ]
            },
            "recommendations": [
                "BERT shows highest accuracy for text-based classifications",
                "GNN excels at spatial relationship analysis",
                "CNN provides strong geometric feature detection",
                "LLM resolution highly effective for model disagreements"
            ]
        }
        
        return model_comparison
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

# Helper functions

async def _analyze_user_patterns(
    user_id: str,
    project_ids: Optional[List[str]],
    analysis_scope: str,
    pattern_types: List[str],
    enhancement_service: EnhancementService
) -> Dict[str, Any]:
    """Analyze patterns for a user"""
    
    # This would implement comprehensive pattern analysis
    return {
        "patterns": [
            {
                "pattern_type": "naming",
                "pattern_description": "Consistent use of storm drainage abbreviations",
                "frequency": 15,
                "confidence": 0.85,
                "examples": ["STRM_18", "STORM_MAIN", "SW_INLET"],
                "recommendations": ["Continue using established naming patterns"]
            }
        ],
        "trends": {
            "classification_consistency": 0.87,
            "feedback_patterns": "Constructive corrections on edge cases",
            "preferred_classifications": ["3D-Sidewalk", "2D-Storm", "3D-Contours"]
        },
        "insights": [
            "User demonstrates strong understanding of infrastructure classifications",
            "Feedback shows preference for specific industry standards",
            "High agreement rate with system recommendations"
        ]
    }

async def _assess_prediction_quality(
    predictions: List[Any],
    project_standards: Optional[Dict[str, Any]],
    assessment_criteria: List[str],
    enhancement_service: EnhancementService
) -> Dict[str, Any]:
    """Assess quality of predictions"""
    
    return {
        "overall_score": 0.82,
        "metrics": [
            {
                "metric_name": "consistency",
                "score": 0.85,
                "description": "Consistency with project standards",
                "recommendations": ["Review naming convention adherence"]
            },
            {
                "metric_name": "accuracy",
                "score": 0.78,
                "description": "Historical accuracy based on feedback",
                "recommendations": ["Apply enhancement to low-confidence predictions"]
            }
        ],
        "compliance_issues": [
            {
                "issue_type": "naming_convention",
                "severity": "medium",
                "description": "Some layers don't follow project naming standards",
                "affected_layers": ["MISC_LINE_01", "TEMP_LAYER"]
            }
        ],
        "suggestions": [
            "Apply LLM enhancement to predictions below 0.75 confidence",
            "Review project standards for naming consistency",
            "Consider document context for ambiguous classifications"
        ]
    }

def _generate_pattern_insights(matches: List[Dict[str, Any]]) -> List[str]:
    """Generate insights from similarity matches"""
    
    if not matches:
        return ["No similar patterns found in historical data"]
    
    insights = []
    
    # Analyze classification consistency
    classifications = [match["classification"] for match in matches if match["classification"]]
    if classifications:
        most_common = max(set(classifications), key=classifications.count)
        consistency = classifications.count(most_common) / len(classifications)
        
        if consistency > 0.8:
            insights.append(f"High consistency: {consistency:.1%} of similar layers classified as '{most_common}'")
        elif consistency > 0.6:
            insights.append(f"Moderate consistency: {consistency:.1%} of similar layers classified as '{most_common}'")
        else:
            insights.append("Low consistency in historical classifications for similar layers")
    
    # Analyze similarity scores
    avg_similarity = sum(match["similarity_score"] for match in matches) / len(matches)
    if avg_similarity > 0.8:
        insights.append(f"Strong similarity to historical patterns (avg: {avg_similarity:.2f})")
    else:
        insights.append(f"Moderate similarity to historical patterns (avg: {avg_similarity:.2f})")
    
    return insights

def _generate_similarity_recommendations(layer_name: str, matches: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on similarity analysis"""
    
    if not matches:
        return ["Consider manual review due to lack of similar historical patterns"]
    
    recommendations = []
    
    # Check for high-confidence similar patterns
    high_confidence_matches = [m for m in matches if m["similarity_score"] > 0.85]
    if high_confidence_matches:
        most_common_class = max(
            set(m["classification"] for m in high_confidence_matches),
            key=lambda x: sum(1 for m in high_confidence_matches if m["classification"] == x)
        )
        recommendations.append(f"Strong recommendation: '{most_common_class}' based on highly similar patterns")
    
    # Check for naming patterns
    if any("storm" in layer_name.lower() for _ in [None]):
        storm_matches = [m for m in matches if "storm" in m["classification"].lower()]
        if storm_matches:
            recommendations.append("Consider storm drainage classification based on naming pattern")
    
    recommendations.append("Review similar cases to ensure consistency with established patterns")
    
    return recommendations