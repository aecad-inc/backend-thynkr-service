from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List, Optional
import logging
from app.models.requests import (
    ContextUpdateRequest,
    ContextRetrievalRequest,
    ContextAnalysisRequest
)
from app.services.context_service import ContextService

router = APIRouter()
logger = logging.getLogger(__name__)

def get_context_service(request: Request) -> ContextService:
    """Dependency to get context service"""
    context_manager = request.app.state.context_manager
    return ContextService(context_manager)

@router.post("/store-session", tags=["Session Context"])
async def store_session_context(
    session_id: str,
    context_data: Dict[str, Any],
    ttl: Optional[int] = None,
    context_service: ContextService = Depends(get_context_service)
):
    """
    Store session-level context for temporary user interactions.
    
    Session context includes:
    - Current file being processed
    - User preferences and settings
    - Conversation history
    - Temporary analysis results
    """
    try:
        await context_service.store_session_context(
            session_id=session_id,
            context_data=context_data,
            ttl=ttl
        )
        
        return {
            "message": "Session context stored successfully",
            "session_id": session_id,
            "ttl_seconds": ttl or 3600
        }
        
    except Exception as e:
        logger.error(f"Failed to store session context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store session context: {str(e)}")

@router.get("/retrieve-session/{session_id}", tags=["Session Context"])
async def retrieve_session_context(
    session_id: str,
    context_service: ContextService = Depends(get_context_service)
):
    """Retrieve session context by session ID"""
    try:
        context = await context_service.get_session_context(session_id)
        
        if not context:
            raise HTTPException(status_code=404, detail="Session context not found")
        
        return {
            "session_id": session_id,
            "context": context,
            "retrieved_at": "2025-01-14T10:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session context: {str(e)}")

@router.post("/store-file-context", tags=["File Context"])
async def store_file_context(
    file_id: str,
    context_data: Dict[str, Any],
    context_service: ContextService = Depends(get_context_service)
):
    """
    Store file-specific context for persistent reference.
    
    File context includes:
    - Project information and standards
    - Document metadata and insights
    - Processing history and results
    - User corrections and feedback
    """
    try:
        await context_service.store_file_context(
            file_id=file_id,
            context_data=context_data
        )
        
        return {
            "message": "File context stored successfully",
            "file_id": file_id,
            "stored_at": "2025-01-14T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to store file context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store file context: {str(e)}")

@router.get("/retrieve-file/{file_id}", tags=["File Context"])
async def retrieve_file_context(
    file_id: str,
    context_service: ContextService = Depends(get_context_service)
):
    """Retrieve file context by file ID"""
    try:
        context = await context_service.get_file_context(file_id)
        
        if not context:
            raise HTTPException(status_code=404, detail="File context not found")
        
        return {
            "file_id": file_id,
            "context": context,
            "retrieved_at": "2025-01-14T10:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve file context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file context: {str(e)}")

@router.post("/store-project-context", tags=["Project Context"])
async def store_project_context(
    project_id: str,
    context_data: Dict[str, Any],
    context_service: ContextService = Depends(get_context_service)
):
    """
    Store project-level context for organizational standards.
    
    Project context includes:
    - Industry type and standards
    - Naming conventions
    - Quality requirements
    - Historical patterns and preferences
    """
    try:
        await context_service.store_project_context(
            project_id=project_id,
            context_data=context_data
        )
        
        return {
            "message": "Project context stored successfully",
            "project_id": project_id,
            "stored_at": "2025-01-14T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to store project context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store project context: {str(e)}")

@router.get("/retrieve-project/{project_id}", tags=["Project Context"])
async def retrieve_project_context(
    project_id: str,
    context_service: ContextService = Depends(get_context_service)
):
    """Retrieve project context by project ID"""
    try:
        context = await context_service.get_project_context(project_id)
        
        if not context:
            raise HTTPException(status_code=404, detail="Project context not found")
        
        return {
            "project_id": project_id,
            "context": context,
            "retrieved_at": "2025-01-14T10:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve project context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve project context: {str(e)}")

@router.post("/semantic-search", tags=["Context Search"])
async def semantic_search_context(
    query: str,
    context_types: Optional[List[str]] = None,
    limit: int = 10,
    context_service: ContextService = Depends(get_context_service)
):
    """
    Perform semantic search across all stored contexts.
    
    This enables finding relevant context based on natural language queries,
    useful for discovering similar patterns, related projects, or relevant history.
    """
    try:
        if limit > 50:
            raise HTTPException(status_code=400, detail="Maximum limit is 50 results")
        
        results = await context_service.semantic_search(
            query=query,
            context_types=context_types,
            limit=limit
        )
        
        return {
            "query": query,
            "context_types_searched": context_types or ["all"],
            "results_count": len(results),
            "results": results,
            "searched_at": "2025-01-14T10:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@router.post("/analyze-document", tags=["Document Analysis"])
async def analyze_document_context(
    request: ContextAnalysisRequest,
    context_service: ContextService = Depends(get_context_service)
):
    """
    Analyze document content to extract layer insights and context.
    
    Supports analysis of:
    - PDF specifications and standards
    - Project documentation
    - Technical drawings metadata
    - Industry standards documents
    """
    try:
        analysis_result = await context_service.analyze_document_context(
            document_content=request.document_content,
            document_type=request.document_type,
            project_context=request.project_context,
            analysis_type=request.analysis_type
        )
        
        return {
            "document_type": request.document_type,
            "analysis_type": request.analysis_type,
            "analysis_results": analysis_result,
            "analyzed_at": "2025-01-14T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")

@router.get("/statistics", tags=["Context Analytics"])
async def get_context_statistics(
    context_service: ContextService = Depends(get_context_service)
):
    """
    Get statistics about stored contexts and patterns.
    
    Provides insights into:
    - Total contexts stored by type
    - Pattern frequency and trends
    - User feedback patterns
    - System learning progress
    """
    try:
        stats = await context_service.get_context_statistics()
        
        return {
            "statistics": stats,
            "generated_at": "2025-01-14T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to get context statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.delete("/clear-session/{session_id}", tags=["Session Management"])
async def clear_session_context(
    session_id: str,
    context_service: ContextService = Depends(get_context_service)
):
    """Clear/delete session context"""
    try:
        success = await context_service.clear_session_context(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "message": "Session context cleared successfully",
            "session_id": session_id,
            "cleared_at": "2025-01-14T10:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear session context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

@router.post("/conversation/update", tags=["Conversation Management"])
async def update_conversation_context(
    session_id: str,
    user_query: str,
    system_response: str,
    context_service: ContextService = Depends(get_context_service)
):
    """Update conversation history for a session"""
    try:
        await context_service.update_conversation_context(
            session_id=session_id,
            user_query=user_query,
            system_response=system_response
        )
        
        return {
            "message": "Conversation context updated successfully",
            "session_id": session_id,
            "updated_at": "2025-01-14T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to update conversation context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {str(e)}")

@router.get("/conversation/{session_id}", tags=["Conversation Management"])
async def get_conversation_history(
    session_id: str,
    limit: Optional[int] = 10,
    context_service: ContextService = Depends(get_context_service)
):
    """Get conversation history for a session"""
    try:
        conversation = await context_service.get_conversation_context(session_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Limit conversation history if requested
        history = conversation.get("conversation_history", [])
        if limit and len(history) > limit:
            history = history[-limit:]
        
        return {
            "session_id": session_id,
            "conversation_history": history,
            "total_interactions": len(conversation.get("conversation_history", [])),
            "retrieved_at": "2025-01-14T10:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")