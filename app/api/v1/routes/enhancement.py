from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from typing import Dict, Any, List
import logging
from app.models.requests import (
    ConfidenceEnhancementRequest,
    ConsensusAnalysisRequest,
    NaturalLanguageQueryRequest,
    FeedbackRequest,
    BatchEnhancementRequest
)
from app.models.responses import (
    EnhancementResponse,
    ConsensusResponse,
    QueryResponse,
    BatchResponse
)
from app.services.enhancement_service import EnhancementService
from app.services.context_service import ContextService

router = APIRouter()
logger = logging.getLogger(__name__)

def get_enhancement_service(request: Request) -> EnhancementService:
    """Dependency to get enhancement service"""
    bedrock_client = request.app.state.bedrock_client
    context_manager = request.app.state.context_manager
    return EnhancementService(bedrock_client, context_manager)

def get_context_service(request: Request) -> ContextService:
    """Dependency to get context service"""
    context_manager = request.app.state.context_manager
    return ContextService(context_manager)

@router.post("/enhance-predictions", response_model=EnhancementResponse, tags=["Enhancement"])
async def enhance_low_confidence_predictions(
    request: ConfidenceEnhancementRequest,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Enhance low-confidence predictions using LLM analysis.
    
    This endpoint takes predictions from CNN, GNN, and BERT models and uses
    LLM intelligence to improve classifications that fall below confidence thresholds.
    """
    try:
        logger.info(f"Processing enhancement request for {len(request.layer_predictions)} layers")
        
        # Validate request
        if not request.layer_predictions:
            raise HTTPException(status_code=400, detail="No layer predictions provided")
        
        # Perform enhancement
        result = await enhancement_service.enhance_low_confidence_predictions(
            layer_predictions=request.layer_predictions,
            file_context=request.file_context
        )
        
        return EnhancementResponse(
            file_version_id=request.file_version_id,
            user_id=request.user_id,
            enhancement_results=result["enhanced_predictions"],
            summary=result["summary"],
            processing_metadata={
                "enhancement_type": request.enhancement_type,
                "total_layers_processed": len(request.layer_predictions),
                "timestamp": "2025-01-14T10:00:00Z"
            }
        )
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement processing failed: {str(e)}")

@router.post("/analyze-consensus", response_model=ConsensusResponse, tags=["Analysis"])
async def analyze_model_consensus(
    request: ConsensusAnalysisRequest,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Analyze consensus between different AI models and resolve conflicts.
    
    When CNN, GNN, and BERT models disagree on classifications, this endpoint
    uses LLM analysis to provide definitive resolutions with reasoning.
    """
    try:
        logger.info(f"Processing consensus analysis for {len(request.conflicting_predictions)} conflicted layers")
        
        result = await enhancement_service.analyze_model_consensus(
            layer_predictions=request.conflicting_predictions,
            file_context=request.context
        )
        
        return ConsensusResponse(
            file_version_id=request.file_version_id,
            user_id=request.user_id,
            consensus_results=result["consensus_analysis"],
            summary=result["summary"],
            explanations_provided=request.require_explanation
        )
        
    except Exception as e:
        logger.error(f"Consensus analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consensus analysis failed: {str(e)}")

@router.post("/natural-language-query", response_model=QueryResponse, tags=["Interactive"])
async def process_natural_language_query(
    request: NaturalLanguageQueryRequest,
    enhancement_service: EnhancementService = Depends(get_enhancement_service),
    context_service: ContextService = Depends(get_context_service)
):
    """
    Process natural language queries about predictions and provide explanations.
    
    Users can ask questions like:
    - "Why was this layer classified as storm drainage?"
    - "What are the most likely alternatives for this classification?"
    - "How confident should I be in this prediction?"
    """
    try:
        logger.info(f"Processing natural language query: {request.query[:100]}...")
        
        # Retrieve conversation context if session provided
        conversation_context = None
        if request.session_id:
            conversation_context = await context_service.get_conversation_context(
                session_id=request.session_id
            )
        
        # Process query using LLM
        query_result = await enhancement_service.process_natural_language_query(
            query=request.query,
            file_context=request.file_context,
            conversation_history=request.conversation_history or [],
            session_context=conversation_context
        )
        
        # Update conversation context
        if request.session_id and query_result.get("success"):
            await context_service.update_conversation_context(
                session_id=request.session_id,
                user_query=request.query,
                system_response=query_result.get("response", "")
            )
        
        return QueryResponse(
            query=request.query,
            response=query_result.get("response", ""),
            confidence=query_result.get("confidence", 0.0),
            supporting_evidence=query_result.get("supporting_evidence", []),
            related_suggestions=query_result.get("related_suggestions", []),
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Natural language query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.post("/submit-feedback", tags=["Learning"])
async def submit_user_feedback(
    request: FeedbackRequest,
    context_service: ContextService = Depends(get_context_service)
):
    """
    Submit user feedback on predictions for continuous learning.
    
    User corrections and feedback help improve future predictions by:
    - Building a knowledge base of user preferences
    - Identifying systematic model errors
    - Improving context-aware recommendations
    """
    try:
        logger.info(f"Processing feedback for layer: {request.layer_name}")
        
        # Store feedback for learning
        feedback_result = await context_service.store_user_feedback(
            file_version_id=request.file_version_id,
            user_id=request.user_id,
            layer_name=request.layer_name,
            original_prediction=request.original_prediction,
            user_correction=request.user_correction,
            confidence_rating=request.confidence_rating,
            comments=request.comments
        )
        
        return {
            "message": "Feedback received successfully",
            "feedback_id": feedback_result.get("feedback_id"),
            "learning_impact": feedback_result.get("learning_impact", "low"),
            "timestamp": "2025-01-14T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@router.post("/batch-enhance", response_model=BatchResponse, tags=["Batch Processing"])
async def batch_enhance_predictions(
    request: BatchEnhancementRequest,
    background_tasks: BackgroundTasks,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Process multiple files for enhancement in batch mode.
    
    Useful for processing large numbers of files efficiently.
    Results are processed asynchronously and can be retrieved via status endpoint.
    """
    try:
        logger.info(f"Starting batch enhancement for {len(request.file_version_ids)} files")
        
        # Validate batch size
        if len(request.file_version_ids) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 files per batch")
        
        # Generate batch job ID
        import uuid
        batch_id = str(uuid.uuid4())
        
        # Start background processing
        background_tasks.add_task(
            _process_batch_enhancement,
            batch_id=batch_id,
            file_ids=request.file_version_ids,
            user_id=request.user_id,
            processing_options=request.processing_options,
            enhancement_service=enhancement_service
        )
        
        return BatchResponse(
            batch_id=batch_id,
            status="queued",
            total_files=len(request.file_version_ids),
            processed_files=0,
            estimated_completion_time="2025-01-14T10:30:00Z",
            message="Batch enhancement job queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Batch enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch enhancement failed: {str(e)}")

@router.get("/batch-status/{batch_id}", response_model=BatchResponse, tags=["Batch Processing"])
async def get_batch_status(
    batch_id: str,
    context_service: ContextService = Depends(get_context_service)
):
    """Get status of a batch enhancement job"""
    try:
        status_result = await context_service.get_batch_status(batch_id)
        
        if not status_result:
            raise HTTPException(status_code=404, detail="Batch job not found")
        
        return BatchResponse(
            batch_id=batch_id,
            status=status_result.get("status", "unknown"),
            total_files=status_result.get("total_files", 0),
            processed_files=status_result.get("processed_files", 0),
            failed_files=status_result.get("failed_files", 0),
            estimated_completion_time=status_result.get("estimated_completion"),
            results_available=status_result.get("results_available", False),
            message=status_result.get("message", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/explain-prediction", tags=["Analysis"])
async def explain_prediction(
    file_version_id: str,
    layer_name: str,
    prediction: str,
    confidence: float,
    enhancement_service: EnhancementService = Depends(get_enhancement_service)
):
    """
    Generate detailed explanation for a specific prediction.
    
    Provides user-friendly explanations of why certain classifications were made,
    what factors influenced the decision, and how confident the system is.
    """
    try:
        logger.info(f"Generating explanation for layer: {layer_name}")
        
        explanation_result = await enhancement_service.generate_prediction_explanation(
            file_version_id=file_version_id,
            layer_name=layer_name,
            prediction=prediction,
            confidence=confidence
        )
        
        return {
            "layer_name": layer_name,
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation_result.get("explanation", ""),
            "key_factors": explanation_result.get("key_factors", []),
            "confidence_breakdown": explanation_result.get("confidence_breakdown", {}),
            "alternative_considerations": explanation_result.get("alternatives", []),
            "visual_cues": explanation_result.get("visual_cues", [])
        }
        
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

async def _process_batch_enhancement(
    batch_id: str,
    file_ids: List[str],
    user_id: str,
    processing_options: Dict[str, Any],
    enhancement_service: EnhancementService
):
    """Background task for processing batch enhancement"""
    try:
        logger.info(f"Starting batch processing for batch_id: {batch_id}")
        
        # Update status to processing
        await enhancement_service.context_manager.update_batch_status(
            batch_id=batch_id,
            status="processing",
            total_files=len(file_ids),
            processed_files=0
        )
        
        processed_count = 0
        failed_count = 0
        results = []
        
        for file_id in file_ids:
            try:
                # Process individual file
                # This would integrate with existing core service to get predictions
                # then enhance them using the enhancement service
                
                # Placeholder for actual file processing
                file_result = await _process_single_file_enhancement(
                    file_id, user_id, processing_options, enhancement_service
                )
                
                results.append(file_result)
                processed_count += 1
                
                # Update progress
                await enhancement_service.context_manager.update_batch_status(
                    batch_id=batch_id,
                    status="processing",
                    total_files=len(file_ids),
                    processed_files=processed_count,
                    failed_files=failed_count
                )
                
            except Exception as e:
                logger.error(f"Failed to process file {file_id}: {e}")
                failed_count += 1
                results.append({
                    "file_id": file_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Mark batch as completed
        final_status = "completed" if failed_count == 0 else "completed_with_errors"
        await enhancement_service.context_manager.update_batch_status(
            batch_id=batch_id,
            status=final_status,
            total_files=len(file_ids),
            processed_files=processed_count,
            failed_files=failed_count,
            results=results
        )
        
        logger.info(f"Batch {batch_id} completed: {processed_count} success, {failed_count} failed")
        
    except Exception as e:
        logger.error(f"Batch processing failed for {batch_id}: {e}")
        await enhancement_service.context_manager.update_batch_status(
            batch_id=batch_id,
            status="failed",
            error=str(e)
        )

async def _process_single_file_enhancement(
    file_id: str,
    user_id: str,
    processing_options: Dict[str, Any],
    enhancement_service: EnhancementService
) -> Dict[str, Any]:
    """Process enhancement for a single file"""
    try:
        # This would integrate with the core service to:
        # 1. Get original predictions from CNN/GNN/BERT models
        # 2. Retrieve file context and metadata
        # 3. Apply LLM enhancement
        # 4. Return enhanced results
        
        # Placeholder implementation
        return {
            "file_id": file_id,
            "success": True,
            "enhanced_predictions": [],
            "improvement_summary": {
                "layers_enhanced": 0,
                "confidence_improvements": []
            }
        }
        
    except Exception as e:
        logger.error(f"Single file enhancement failed for {file_id}: {e}")
        raise