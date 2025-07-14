import logging
from typing import Dict, Any, List, Optional
from app.core.context_manager import ContextManager
from app.models.requests import ProjectContext

logger = logging.getLogger(__name__)

class ContextService:
    """Service layer for context management operations"""
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
    
    # === Session Context Operations ===
    async def store_session_context(
        self,
        session_id: str,
        context_data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Store session context with validation"""
        # Validate context data
        if not isinstance(context_data, dict):
            raise ValueError("Context data must be a dictionary")
        
        await self.context_manager.store_session_context(
            session_id=session_id,
            context_data=context_data,
            ttl=ttl
        )
    
    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session context"""
        return await self.context_manager.get_session_context(session_id)
    
    async def clear_session_context(self, session_id: str) -> bool:
        """Clear session context"""
        try:
            # Since Redis doesn't have a direct clear method, we'll set empty context with TTL=1
            await self.context_manager.store_session_context(
                session_id=session_id,
                context_data={},
                ttl=1
            )
            return True
        except Exception as e:
            logger.error(f"Failed to clear session context: {e}")
            return False
    
    # === File Context Operations ===
    async def store_file_context(
        self,
        file_id: str,
        context_data: Dict[str, Any]
    ):
        """Store file context with validation"""
        # Add file-specific metadata
        context_data["file_id"] = file_id
        context_data["context_type"] = "file"
        
        await self.context_manager.store_file_context(
            file_id=file_id,
            context_data=context_data
        )
    
    async def get_file_context(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file context"""
        return await self.context_manager.get_file_context(file_id)
    
    # === Project Context Operations ===
    async def store_project_context(
        self,
        project_id: str,
        context_data: Dict[str, Any]
    ):
        """Store project context with validation"""
        # Add project-specific metadata
        context_data["project_id"] = project_id
        context_data["context_type"] = "project"
        
        await self.context_manager.store_project_context(
            project_id=project_id,
            context_data=context_data
        )
    
    async def get_project_context(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project context"""
        return await self.context_manager.get_project_context(project_id)
    
    # === Conversation Management ===
    async def update_conversation_context(
        self,
        session_id: str,
        user_query: str,
        system_response: str
    ):
        """Update conversation history"""
        await self.context_manager.update_conversation_context(
            session_id=session_id,
            user_query=user_query,
            system_response=system_response
        )
    
    async def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation context"""
        return await self.context_manager.get_conversation_context(session_id)
    
    # === User Feedback Management ===
    async def store_user_feedback(
        self,
        file_version_id: str,
        user_id: str,
        layer_name: str,
        original_prediction: str,
        user_correction: str,
        confidence_rating: Optional[float] = None,
        comments: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store user feedback for learning"""
        return await self.context_manager.store_user_feedback(
            file_version_id=file_version_id,
            user_id=user_id,
            layer_name=layer_name,
            original_prediction=original_prediction,
            user_correction=user_correction,
            confidence_rating=confidence_rating,
            comments=comments
        )
    
    # === Batch Processing Management ===
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch processing status"""
        return await self.context_manager.get_batch_status(batch_id)
    
    # === Semantic Search ===
    async def semantic_search(
        self,
        query: str,
        context_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform semantic search across contexts"""
        return await self.context_manager.search_context_by_query(
            query=query,
            context_types=context_types,
            limit=limit
        )
    
    # === Document Analysis ===
    async def analyze_document_context(
        self,
        document_content: str,
        document_type: str,
        project_context: Optional[ProjectContext] = None,
        analysis_type: str = "layer_extraction"
    ) -> Dict[str, Any]:
        """Analyze document content for layer insights"""
        try:
            # This is a placeholder for document analysis logic
            # In a full implementation, this would use LLM to analyze documents
            
            analysis_result = {
                "document_type": document_type,
                "analysis_type": analysis_type,
                "content_length": len(document_content),
                "extracted_insights": [],
                "layer_recommendations": [],
                "confidence_boosts": []
            }
            
            # Basic content analysis
            if "storm" in document_content.lower():
                analysis_result["layer_recommendations"].append({
                    "layer_type": "2D-Storm",
                    "confidence_boost": 0.2,
                    "reasoning": "Document mentions storm water management"
                })
            
            if "sewer" in document_content.lower():
                analysis_result["layer_recommendations"].append({
                    "layer_type": "2D-Sewer",
                    "confidence_boost": 0.2,
                    "reasoning": "Document mentions sewer systems"
                })
            
            # Store analysis as context for future reference
            await self.context_manager.store_enhancement_pattern({
                "pattern_type": "document_analysis",
                "document_type": document_type,
                "analysis_results": analysis_result,
                "timestamp": "2025-01-14T10:00:00Z"
            })
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            raise
    
    # === Statistics and Analytics ===
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get context storage statistics"""
        try:
            # Get vector store statistics
            vector_stats = await self.context_manager.vector_store.get_context_statistics()
            
            # Add additional analytics
            stats = {
                "vector_store": vector_stats,
                "session_contexts": {
                    "active_sessions": len(self.context_manager._batch_status),
                    "batch_jobs": len([k for k in self.context_manager._batch_status.keys() if k.startswith("batch:")])
                },
                "system_health": {
                    "redis_connected": self.context_manager.redis_client is not None,
                    "vector_store_connected": self.context_manager.vector_store.client is not None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get context statistics: {e}")
            raise