import logging
import json
import aioredis
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import asyncio
from app.config.settings import settings
from app.utils.vector_store import VectorStore

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages multi-level context for LLM operations with proper async Redis handling"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.redis_client: Optional[aioredis.Redis] = None
        self._batch_status: Dict[str, Dict[str, Any]] = {}  # In-memory batch status cache
        self._connection_pool: Optional[aioredis.ConnectionPool] = None
        
    async def initialize(self):
        """Initialize context management components with proper async Redis"""
        try:
            # Create Redis connection pool for better performance
            self._connection_pool = aioredis.ConnectionPool.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Initialize async Redis client
            self.redis_client = aioredis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            logger.info("✅ Context Manager initialized successfully with async Redis")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Context Manager: {e}")
            await self.cleanup()  # Cleanup on failure
            raise
    
    # === Session Context Management ===
    async def store_session_context(
        self,
        session_id: str,
        context_data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Store session-level context in Redis with proper async handling"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
            
        try:
            ttl = ttl or settings.SESSION_TIMEOUT
            context_key = f"session:{session_id}"
            
            # Serialize context data with error handling
            try:
                context_json = json.dumps(context_data, default=str, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize context data: {e}")
                raise ValueError(f"Context data serialization failed: {e}")
            
            # Store with TTL using async setex
            await self.redis_client.setex(context_key, ttl, context_json)
            
            logger.debug(f"Stored session context for {session_id} with TTL {ttl}s")
            
        except Exception as e:
            logger.error(f"Failed to store session context for {session_id}: {e}")
            raise
    
    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session context from Redis with proper error handling"""
        if not self.redis_client:
            logger.warning("Redis client not initialized, returning None")
            return None
            
        try:
            context_key = f"session:{session_id}"
            
            # Get data using async get
            context_json: Optional[str] = await self.redis_client.get(context_key)
            
            if context_json is None:
                logger.debug(f"No session context found for {session_id}")
                return None
            
            # Parse JSON with error handling
            try:
                context_data = json.loads(context_json)
                return context_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse session context JSON for {session_id}: {e}")
                # Delete corrupted data
                await self.redis_client.delete(context_key)
                return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve session context for {session_id}: {e}")
            return None
    
    async def update_session_context(
        self,
        session_id: str,
        updates: Dict[str, Any],
        merge: bool = True
    ):
        """Update existing session context with proper merging"""
        try:
            if merge:
                existing_context = await self.get_session_context(session_id) or {}
                # Deep merge for nested dictionaries
                merged_context = self._deep_merge_dicts(existing_context, updates)
                await self.store_session_context(session_id, merged_context)
            else:
                await self.store_session_context(session_id, updates)
                
        except Exception as e:
            logger.error(f"Failed to update session context for {session_id}: {e}")
            raise
    
    def _deep_merge_dicts(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
                
        return result
    
    # === Conversation Context Management ===
    async def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history and context with safe defaults"""
        try:
            session_context = await self.get_session_context(session_id) or {}
            
            return {
                "conversation_history": session_context.get("conversation_history", []),
                "current_file_context": session_context.get("current_file_context"),
                "user_preferences": session_context.get("user_preferences", {}),
                "active_queries": session_context.get("active_queries", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation context for {session_id}: {e}")
            return {
                "conversation_history": [],
                "current_file_context": None,
                "user_preferences": {},
                "active_queries": []
            }
    
    async def update_conversation_context(
        self,
        session_id: str,
        user_query: str,
        system_response: str
    ):
        """Add new interaction to conversation history with size limits"""
        try:
            session_context = await self.get_session_context(session_id) or {}
            
            # Get existing conversation history
            conversation_history = session_context.get("conversation_history", [])
            
            # Add new interaction with timestamp
            new_interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_query": user_query[:1000],  # Limit query length
                "system_response": system_response[:2000]  # Limit response length
            }
            
            conversation_history.append(new_interaction)
            
            # Limit conversation history size for memory management
            max_history_size = settings.CONTEXT_WINDOW_SIZE
            if len(conversation_history) > max_history_size:
                conversation_history = conversation_history[-max_history_size:]
            
            # Update session context
            session_context["conversation_history"] = conversation_history
            session_context["last_interaction"] = datetime.utcnow().isoformat()
            
            await self.store_session_context(session_id, session_context)
            
        except Exception as e:
            logger.error(f"Failed to update conversation context for {session_id}: {e}")
            raise
    
    # === File Context Management ===
    async def store_file_context(
        self,
        file_id: str,
        context_data: Dict[str, Any]
    ):
        """Store file-specific context in vector store with validation"""
        try:
            # Validate file_id
            if not file_id or not isinstance(file_id, str):
                raise ValueError("Invalid file_id provided")
            
            # Add metadata for better retrieval
            enhanced_context = {
                **context_data,
                "file_id": file_id,
                "stored_at": datetime.utcnow().isoformat(),
                "context_type": "file",
                "version": "1.0"
            }
            
            # Store in vector database
            await self.vector_store.store_context(
                context_id=f"file:{file_id}",
                context_data=enhanced_context,
                context_type="file"
            )
            
            logger.debug(f"Stored file context for {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to store file context for {file_id}: {e}")
            raise
    
    async def get_file_context(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file context with error handling"""
        try:
            if not file_id:
                return None
                
            context = await self.vector_store.get_context(f"file:{file_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve file context for {file_id}: {e}")
            return None
    
    # === Pattern Storage and Retrieval ===
    async def store_enhancement_pattern(self, pattern_data: Dict[str, Any]):
        """Store enhancement pattern for future learning with validation"""
        try:
            # Validate required fields
            if not pattern_data.get("layer_name"):
                raise ValueError("layer_name is required for enhancement patterns")
            
            timestamp = datetime.utcnow().timestamp()
            pattern_id = f"enhancement:{pattern_data['layer_name']}:{timestamp}"
            
            enhanced_pattern = {
                **pattern_data,
                "pattern_id": pattern_id,
                "pattern_type": "enhancement",
                "stored_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
            
            await self.vector_store.store_pattern(
                pattern_id=pattern_id,
                pattern_data=enhanced_pattern
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
        """Find similar layer patterns from historical data with caching"""
        try:
            if not layer_name or limit <= 0:
                return []
            
            # Check cache first
            cache_key = f"patterns:{layer_name}:{limit}"
            cached_patterns = await self._get_cached_patterns(cache_key)
            
            if cached_patterns is not None:
                return cached_patterns
            
            # Query vector store
            similar_patterns = await self.vector_store.find_similar_patterns(
                query_text=layer_name,
                pattern_type="enhancement",
                limit=limit
            )
            
            # Cache results for 5 minutes
            await self._cache_patterns(cache_key, similar_patterns, ttl=300)
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns for {layer_name}: {e}")
            return []
    
    async def _get_cached_patterns(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached patterns from Redis"""
        if not self.redis_client:
            return None
            
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.debug(f"Failed to get cached patterns: {e}")
        
        return None
    
    async def _cache_patterns(self, cache_key: str, patterns: List[Dict[str, Any]], ttl: int):
        """Cache patterns in Redis"""
        if not self.redis_client:
            return
            
        try:
            patterns_json = json.dumps(patterns, default=str)
            await self.redis_client.setex(cache_key, ttl, patterns_json)
        except Exception as e:
            logger.debug(f"Failed to cache patterns: {e}")
    
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
        """Store user feedback for learning with validation"""
        try:
            # Validate inputs
            required_fields = {
                "file_version_id": file_version_id,
                "user_id": user_id,
                "layer_name": layer_name,
                "original_prediction": original_prediction,
                "user_correction": user_correction
            }
            
            for field, value in required_fields.items():
                if not value or not isinstance(value, str):
                    raise ValueError(f"Invalid {field}: {value}")
            
            timestamp = datetime.utcnow().timestamp()
            feedback_id = f"feedback:{file_version_id}:{layer_name}:{timestamp}"
            
            feedback_data = {
                "feedback_id": feedback_id,
                "file_version_id": file_version_id,
                "user_id": user_id,
                "layer_name": layer_name,
                "original_prediction": original_prediction,
                "user_correction": user_correction,
                "confidence_rating": confidence_rating,
                "comments": comments[:500] if comments else None,  # Limit comment length
                "timestamp": datetime.utcnow().isoformat(),
                "context_type": "feedback",
                "version": "1.0"
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
            
            # Determine impact level based on frequency
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
        """Update batch processing status with proper persistence"""
        try:
            # Get existing batch data
            batch_data = self._batch_status.get(batch_id, {})
            
            # Update fields
            batch_data.update({
                "batch_id": batch_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            })
            
            # Update optional fields
            if total_files is not None:
                batch_data["total_files"] = total_files
            if processed_files is not None:
                batch_data["processed_files"] = processed_files
            if failed_files is not None:
                batch_data["failed_files"] = failed_files
            if results is not None:
                batch_data["results"] = results[:100]  # Limit results size
            if error is not None:
                batch_data["error"] = error[:1000]  # Limit error message size
            
            # Calculate estimated completion if processing
            if status == "processing" and total_files and processed_files:
                progress_rate = processed_files / total_files
                if progress_rate > 0 and progress_rate < 1:
                    remaining_files = total_files - processed_files
                    remaining_time = remaining_files * 60  # Estimate 1 minute per file
                    completion_time = datetime.utcnow() + timedelta(seconds=remaining_time)
                    batch_data["estimated_completion"] = completion_time.isoformat()
            
            # Store in memory cache
            self._batch_status[batch_id] = batch_data
            
            # Persist to Redis with 24-hour TTL
            if self.redis_client:
                batch_key = f"batch:{batch_id}"
                batch_json = json.dumps(batch_data, default=str)
                await self.redis_client.setex(batch_key, 86400, batch_json)
            
        except Exception as e:
            logger.error(f"Failed to update batch status for {batch_id}: {e}")
            raise
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch processing status with fallback to Redis"""
        try:
            # Try memory cache first
            if batch_id in self._batch_status:
                return self._batch_status[batch_id]
            
            # Fall back to Redis
            if self.redis_client:
                batch_key = f"batch:{batch_id}"
                batch_json = await self.redis_client.get(batch_key)
                
                if batch_json:
                    status_data = json.loads(batch_json)
                    self._batch_status[batch_id] = status_data  # Cache in memory
                    return status_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get batch status for {batch_id}: {e}")
            return None
    
    # === Project Context Management ===
    async def store_project_context(
        self,
        project_id: str,
        context_data: Dict[str, Any]
    ):
        """Store project-level context with validation"""
        try:
            if not project_id:
                raise ValueError("project_id is required")
            
            enhanced_context = {
                **context_data,
                "project_id": project_id,
                "context_type": "project",
                "stored_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
            
            await self.vector_store.store_context(
                context_id=f"project:{project_id}",
                context_data=enhanced_context,
                context_type="project"
            )
            
            logger.debug(f"Stored project context for {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to store project context for {project_id}: {e}")
            raise
    
    async def get_project_context(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project context with error handling"""
        try:
            if not project_id:
                return None
                
            context = await self.vector_store.get_context(f"project:{project_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve project context for {project_id}: {e}")
            return None
    
    # === Context Search and Analysis ===
    async def search_context_by_query(
        self,
        query: str,
        context_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search context using semantic similarity with validation"""
        try:
            if not query or limit <= 0:
                return []
            
            # Sanitize query
            clean_query = query.strip()[:500]  # Limit query length
            
            # Use safe default for context_types
            safe_context_types = context_types if context_types else []
            
            results = await self.vector_store.semantic_search(
                query=clean_query,
                context_types=safe_context_types,
                limit=min(limit, 50)  # Cap at 50 results
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search context: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources with proper error handling for aioredis"""
        try:
            # Clear memory cache first
            self._batch_status.clear()
            
            # Close Redis client properly
            if self.redis_client:
                try:
                    # For aioredis, we need to close the connection pool
                    await self.redis_client.close()
                    self.redis_client = None
                    logger.debug("Redis client closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing Redis client: {e}")
            
            # Close connection pool if it exists
            if self._connection_pool:
                try:
                    await self._connection_pool.disconnect()
                    self._connection_pool = None
                    logger.debug("Redis connection pool disconnected successfully")
                except Exception as e:
                    logger.warning(f"Error disconnecting Redis connection pool: {e}")
            
            # Cleanup vector store
            if self.vector_store:
                try:
                    await self.vector_store.cleanup()
                    logger.debug("Vector store cleaned up successfully")
                except Exception as e:
                    logger.warning(f"Error cleaning up vector store: {e}")
            
            logger.info("✅ Context Manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during Context Manager cleanup: {e}")
            # Don't re-raise here as cleanup should be robust
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()