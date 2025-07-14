import logging
import json
import redis
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from app.config.settings import settings
from app.utils.vector_store import VectorStore

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages multi-level context for LLM operations"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.redis_client = None
        self._batch_status = {}  # In-memory batch status cache
        
    async def initialize(self):
        """Initialize context management components"""
        try:
            # Initialize Redis for session context
            self.redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            
            # Test Redis connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            logger.info("✅ Context Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Context Manager: {e}")
            raise
    
    # === Session Context Management ===
    async def store_session_context(
        self,
        session_id: str,
        context_data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Store session-level context in Redis"""
        try:
            ttl = ttl or settings.SESSION_TIMEOUT
            context_key = f"session:{session_id}"
            
            # Serialize context data
            context_json = json.dumps(context_data, default=str)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(context_key, ttl, context_json)
            )
            
            logger.debug(f"Stored session context for {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store session context: {e}")
            raise
    
    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session context from Redis"""
        try:
            context_key = f"session:{session_id}"
            
            context_json = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, context_key
            )
            
            if context_json:
                return json.loads(context_json)
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve session context: {e}")
            return None
    
    async def update_session_context(
        self,
        session_id: str,
        updates: Dict[str, Any],
        merge: bool = True
    ):
        """Update existing session context"""
        try:
            if merge:
                existing_context = await self.get_session_context(session_id) or {}
                existing_context.update(updates)
                await self.store_session_context(session_id, existing_context)
            else:
                await self.store_session_context(session_id, updates)
                
        except Exception as e:
            logger.error(f"Failed to update session context: {e}")
            raise
    
    # === Conversation Context Management ===
    async def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history and context"""
        try:
            session_context = await self.get_session_context(session_id) or {}
            
            return {
                "conversation_history": session_context.get("conversation_history", []),
                "current_file_context": session_context.get("current_file_context"),
                "user_preferences": session_context.get("user_preferences", {}),
                "active_queries": session_context.get("active_queries", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {}
    
    async def update_conversation_context(
        self,
        session_id: str,
        user_query: str,
        system_response: str
    ):
        """Add new interaction to conversation history"""
        try:
            session_context = await self.get_session_context(session_id) or {}
            
            # Get existing conversation history
            conversation_history = session_context.get("conversation_history", [])
            
            # Add new interaction
            new_interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_query": user_query,
                "system_response": system_response
            }
            
            conversation_history.append(new_interaction)
            
            # Limit conversation history size
            if len(conversation_history) > settings.CONTEXT_WINDOW_SIZE:
                conversation_history = conversation_history[-settings.CONTEXT_WINDOW_SIZE:]
            
            # Update session context
            session_context["conversation_history"] = conversation_history
            await self.store_session_context(session_id, session_context)
            
        except Exception as e:
            logger.error(f"Failed to update conversation context: {e}")
            raise
    
    # === File Context Management ===
    async def store_file_context(
        self,
        file_id: str,
        context_data: Dict[str, Any]
    ):
        """Store file-specific context in vector store"""
        try:
            # Add metadata for better retrieval
            context_data["file_id"] = file_id
            context_data["stored_at"] = datetime.utcnow().isoformat()
            context_data["context_type"] = "file"
            
            # Store in vector database
            await self.vector_store.store_context(
                context_id=f"file:{file_id}",
                context_data=context_data,
                context_type="file"
            )
            
            logger.debug(f"Stored file context for {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to store file context: {e}")
            raise
    
    async def get_file_context(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file context"""
        try:
            context = await self.vector_store.get_context(f"file:{file_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve file context: {e}")
            return None
    
    # === Pattern Storage and Retrieval ===
    async def store_enhancement_pattern(self, pattern_data: Dict[str, Any]):
        """Store enhancement pattern for future learning"""
        try:
            pattern_id = f"enhancement:{pattern_data['layer_name']}:{datetime.utcnow().timestamp()}"
            pattern_data["pattern_type"] = "enhancement"
            pattern_data["stored_at"] = datetime.utcnow().isoformat()
            
            await self.vector_store.store_pattern(
                pattern_id=pattern_id,
                pattern_data=pattern_data
            )
            
            logger.debug(f"Stored enhancement pattern for {pattern_data['layer_name']}")
            
        except Exception as e:
            logger.error(f"Failed to store enhancement pattern: {e}")
            raise
    
    async def find_similar_patterns(
        self,
        layer_name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar layer patterns from historical data"""
        try:
            similar_patterns = await self.vector_store.find_similar_patterns(
                query_text=layer_name,
                pattern_type="enhancement",
                limit=limit
            )
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
            return []
    
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
        try:
            feedback_id = f"feedback:{file_version_id}:{layer_name}:{datetime.utcnow().timestamp()}"
            
            feedback_data = {
                "feedback_id": feedback_id,
                "file_version_id": file_version_id,
                "user_id": user_id,
                "layer_name": layer_name,
                "original_prediction": original_prediction,
                "user_correction": user_correction,
                "confidence_rating": confidence_rating,
                "comments": comments,
                "timestamp": datetime.utcnow().isoformat(),
                "context_type": "feedback"
            }
            
            # Store feedback
            await self.vector_store.store_context(
                context_id=feedback_id,
                context_data=feedback_data,
                context_type="feedback"
            )
            
            # Analyze learning impact
            learning_impact = await self._analyze_feedback_impact(feedback_data)
            
            return {
                "feedback_id": feedback_id,
                "learning_impact": learning_impact,
                "stored_successfully": True
            }
            
        except Exception as e:
            logger.error(f"Failed to store user feedback: {e}")
            raise
    
    async def _analyze_feedback_impact(self, feedback_data: Dict[str, Any]) -> str:
        """Analyze the learning impact of user feedback"""
        try:
            # Find similar corrections
            similar_feedback = await self.vector_store.find_similar_patterns(
                query_text=feedback_data["layer_name"],
                pattern_type="feedback",
                limit=10
            )
            
            # Count similar corrections
            similar_corrections = sum(
                1 for fb in similar_feedback
                if fb.get("user_correction") == feedback_data["user_correction"]
            )
            
            # Determine impact level
            if similar_corrections >= 3:
                return "high"  # Pattern confirmed by multiple users
            elif similar_corrections >= 1:
                return "medium"  # Some agreement
            else:
                return "low"  # New or unique correction
                
        except Exception as e:
            logger.warning(f"Could not analyze feedback impact: {e}")
            return "low"
    
    # === Batch Processing Management ===
    async def update_batch_status(
        self,
        batch_id: str,
        status: str,
        total_files: Optional[int] = None,
        processed_files: Optional[int] = None,
        failed_files: Optional[int] = None,
        results: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None
    ):
        """Update batch processing status"""
        try:
            batch_data = self._batch_status.get(batch_id, {})
            
            # Update fields
            batch_data.update({
                "batch_id": batch_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            })
            
            if total_files is not None:
                batch_data["total_files"] = total_files
            if processed_files is not None:
                batch_data["processed_files"] = processed_files
            if failed_files is not None:
                batch_data["failed_files"] = failed_files
            if results is not None:
                batch_data["results"] = results
            if error is not None:
                batch_data["error"] = error
            
            # Calculate estimated completion if processing
            if status == "processing" and total_files and processed_files:
                progress_rate = processed_files / total_files
                if progress_rate > 0:
                    remaining_time = (1 - progress_rate) * 300  # Estimate 5 minutes per file
                    completion_time = datetime.utcnow() + timedelta(seconds=remaining_time)
                    batch_data["estimated_completion"] = completion_time.isoformat()
            
            # Store in memory cache and optionally persist
            self._batch_status[batch_id] = batch_data
            
            # Also store in Redis for persistence
            await self.store_session_context(
                session_id=f"batch:{batch_id}",
                context_data=batch_data,
                ttl=86400  # 24 hours
            )
            
        except Exception as e:
            logger.error(f"Failed to update batch status: {e}")
            raise
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch processing status"""
        try:
            # Try memory cache first
            if batch_id in self._batch_status:
                return self._batch_status[batch_id]
            
            # Fall back to Redis
            status = await self.get_session_context(f"batch:{batch_id}")
            if status:
                self._batch_status[batch_id] = status  # Cache in memory
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get batch status: {e}")
            return None
    
    # === Project Context Management ===
    async def store_project_context(
        self,
        project_id: str,
        context_data: Dict[str, Any]
    ):
        """Store project-level context"""
        try:
            context_data["project_id"] = project_id
            context_data["context_type"] = "project"
            context_data["stored_at"] = datetime.utcnow().isoformat()
            
            await self.vector_store.store_context(
                context_id=f"project:{project_id}",
                context_data=context_data,
                context_type="project"
            )
            
            logger.debug(f"Stored project context for {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to store project context: {e}")
            raise
    
    async def get_project_context(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project context"""
        try:
            context = await self.vector_store.get_context(f"project:{project_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve project context: {e}")
            return None
    
    # === Context Search and Analysis ===
    async def search_context_by_query(
        self,
        query: str,
        context_types: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search context using semantic similarity"""
        try:
            results = await self.vector_store.semantic_search(
                query=query,
                context_types=context_types,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search context: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.close
                )
            
            if self.vector_store:
                await self.vector_store.cleanup()
            
            self._batch_status.clear()
            
            logger.info("✅ Context Manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during Context Manager cleanup: {e}")