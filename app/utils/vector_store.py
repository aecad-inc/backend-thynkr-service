import logging
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Any, Optional
import json
from app.config.settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB-based vector store for context and pattern storage"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        
    async def initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "AECAD context and pattern storage"}
            )
            
            logger.info(f"✅ Vector store initialized with collection: {settings.CHROMA_COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    async def store_context(
        self,
        context_id: str,
        context_data: Dict[str, Any],
        context_type: str
    ):
        """Store context data with embeddings"""
        try:
            # Create text representation for embedding
            text_content = self._create_searchable_text(context_data, context_type)
            
            # Prepare metadata
            metadata = {
                "context_type": context_type,
                "stored_at": context_data.get("stored_at", ""),
                "file_id": context_data.get("file_id", ""),
                "project_id": context_data.get("project_id", ""),
                "user_id": context_data.get("user_id", "")
            }
            
            # Remove None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            # Store in ChromaDB
            self.collection.add(
                documents=[text_content],
                metadatas=[metadata],
                ids=[context_id]
            )
            
            # Also store full context data as JSON in a separate document
            full_context_id = f"{context_id}_full"
            self.collection.add(
                documents=[json.dumps(context_data, default=str)],
                metadatas=[{**metadata, "is_full_context": True}],
                ids=[full_context_id]
            )
            
            logger.debug(f"Stored context: {context_id}")
            
        except Exception as e:
            logger.error(f"Failed to store context {context_id}: {e}")
            raise
    
    async def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context by ID"""
        try:
            full_context_id = f"{context_id}_full"
            
            # Query for full context
            results = self.collection.get(
                ids=[full_context_id],
                include=["documents", "metadatas"]
            )
            
            if results["documents"]:
                context_json = results["documents"][0]
                return json.loads(context_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve context {context_id}: {e}")
            return None
    
    async def store_pattern(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any]
    ):
        """Store enhancement or learning pattern"""
        try:
            # Create searchable text for the pattern
            text_content = self._create_pattern_text(pattern_data)
            
            metadata = {
                "pattern_type": pattern_data.get("pattern_type", "unknown"),
                "layer_name": pattern_data.get("layer_name", ""),
                "model": pattern_data.get("model", ""),
                "confidence_improvement": pattern_data.get("confidence_improvement", 0.0),
                "stored_at": pattern_data.get("stored_at", "")
            }
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            # Store pattern
            self.collection.add(
                documents=[text_content],
                metadatas=[metadata],
                ids=[pattern_id]
            )
            
            # Store full pattern data
            full_pattern_id = f"{pattern_id}_full"
            self.collection.add(
                documents=[json.dumps(pattern_data, default=str)],
                metadatas=[{**metadata, "is_full_pattern": True}],
                ids=[full_pattern_id]
            )
            
            logger.debug(f"Stored pattern: {pattern_id}")
            
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern_id}: {e}")
            raise
    
    async def find_similar_patterns(
        self,
        query_text: str,
        pattern_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar patterns using semantic search"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if pattern_type:
                where_clause["pattern_type"] = pattern_type
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=limit * 2,  # Get more results to filter
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            patterns = []
            seen_base_ids = set()
            
            # Process results and get full pattern data
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # Skip full context documents in similarity search
                if metadata.get("is_full_pattern"):
                    continue
                
                # Extract base pattern ID
                pattern_id = results["ids"][0][i]
                if pattern_id in seen_base_ids:
                    continue
                seen_base_ids.add(pattern_id)
                
                # Get full pattern data
                full_pattern = await self.get_pattern(pattern_id)
                if full_pattern:
                    full_pattern["similarity_score"] = 1.0 - distance  # Convert distance to similarity
                    patterns.append(full_pattern)
                
                if len(patterns) >= limit:
                    break
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
            return []
    
    async def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get full pattern data by ID"""
        try:
            full_pattern_id = f"{pattern_id}_full"
            
            results = self.collection.get(
                ids=[full_pattern_id],
                include=["documents"]
            )
            
            if results["documents"]:
                pattern_json = results["documents"][0]
                return json.loads(pattern_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get pattern {pattern_id}: {e}")
            return None
    
    async def semantic_search(
        self,
        query: str,
        context_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform semantic search across all stored contexts"""
        try:
            # Build where clause
            where_clause = {}
            if context_types:
                where_clause["context_type"] = {"$in": context_types}
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit * 2,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            seen_base_ids = set()
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Skip full context documents in search results
                if metadata.get("is_full_context") or metadata.get("is_full_pattern"):
                    continue
                
                context_id = results["ids"][0][i]
                if context_id in seen_base_ids:
                    continue
                seen_base_ids.add(context_id)
                
                # Get full context
                full_context = await self.get_context(context_id)
                if full_context:
                    search_result = {
                        "context_id": context_id,
                        "similarity_score": 1.0 - distance,
                        "context_type": metadata.get("context_type"),
                        "preview": doc[:200] + "..." if len(doc) > 200 else doc,
                        "full_context": full_context
                    }
                    search_results.append(search_result)
                
                if len(search_results) >= limit:
                    break
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored contexts"""
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get breakdown by context type
            all_metadata = self.collection.get(include=["metadatas"])["metadatas"]
            
            context_type_counts = {}
            pattern_type_counts = {}
            
            for metadata in all_metadata:
                if metadata.get("is_full_context") or metadata.get("is_full_pattern"):
                    continue  # Skip full documents
                
                context_type = metadata.get("context_type", "unknown")
                context_type_counts[context_type] = context_type_counts.get(context_type, 0) + 1
                
                if context_type == "enhancement":
                    pattern_type = metadata.get("pattern_type", "unknown")
                    pattern_type_counts[pattern_type] = pattern_type_counts.get(pattern_type, 0) + 1
            
            return {
                "total_documents": collection_count,
                "context_type_breakdown": context_type_counts,
                "pattern_type_breakdown": pattern_type_counts,
                "collection_name": settings.CHROMA_COLLECTION_NAME
            }
            
        except Exception as e:
            logger.error(f"Failed to get context statistics: {e}")
            return {}
    
    def _create_searchable_text(self, context_data: Dict[str, Any], context_type: str) -> str:
        """Create searchable text from context data"""
        try:
            text_parts = []
            
            # Add context type
            text_parts.append(f"Context type: {context_type}")
            
            # Add key fields based on context type
            if context_type == "file":
                if "file_name" in context_data:
                    text_parts.append(f"File: {context_data['file_name']}")
                if "project_context" in context_data:
                    project_ctx = context_data["project_context"]
                    if isinstance(project_ctx, dict):
                        if "project_type" in project_ctx:
                            text_parts.append(f"Project type: {project_ctx['project_type']}")
                        if "project_description" in project_ctx:
                            text_parts.append(f"Description: {project_ctx['project_description']}")
            
            elif context_type == "feedback":
                text_parts.extend([
                    f"Layer: {context_data.get('layer_name', '')}",
                    f"Original: {context_data.get('original_prediction', '')}",
                    f"Correction: {context_data.get('user_correction', '')}",
                    f"Comments: {context_data.get('comments', '')}"
                ])
            
            elif context_type == "session":
                if "conversation_history" in context_data:
                    history = context_data["conversation_history"]
                    if isinstance(history, list) and history:
                        recent_queries = [h.get("user_query", "") for h in history[-3:]]
                        text_parts.append(f"Recent queries: {' '.join(recent_queries)}")
            
            # Add any additional text content
            for key in ["content", "description", "notes", "reasoning"]:
                if key in context_data and context_data[key]:
                    text_parts.append(str(context_data[key]))
            
            return " | ".join(filter(None, text_parts))
            
        except Exception as e:
            logger.warning(f"Failed to create searchable text: {e}")
            return json.dumps(context_data, default=str)[:500]  # Fallback
    
    def _create_pattern_text(self, pattern_data: Dict[str, Any]) -> str:
        """Create searchable text for patterns"""
        try:
            text_parts = []
            
            # Add pattern type
            pattern_type = pattern_data.get("pattern_type", "unknown")
            text_parts.append(f"Pattern: {pattern_type}")
            
            # Add layer information
            if "layer_name" in pattern_data:
                text_parts.append(f"Layer: {pattern_data['layer_name']}")
            
            # Add prediction information
            if "original_predictions" in pattern_data:
                predictions = pattern_data["original_predictions"]
                if isinstance(predictions, list):
                    pred_text = ", ".join([
                        f"{p.get('model', '')}: {p.get('prediction', '')}"
                        for p in predictions
                    ])
                    text_parts.append(f"Original predictions: {pred_text}")
            
            # Add enhanced prediction
            if "enhanced_prediction" in pattern_data:
                text_parts.append(f"Enhanced: {pattern_data['enhanced_prediction']}")
            
            # Add reasoning
            if "reasoning" in pattern_data:
                reasoning = pattern_data["reasoning"]
                if isinstance(reasoning, list):
                    text_parts.append(f"Reasoning: {' '.join(reasoning)}")
                elif isinstance(reasoning, str):
                    text_parts.append(f"Reasoning: {reasoning}")
            
            # Add context factors
            if "context_factors" in pattern_data:
                factors = pattern_data["context_factors"]
                if isinstance(factors, dict):
                    factor_text = ", ".join([
                        f"{k}: {v}" for k, v in factors.items() if v
                    ])
                    text_parts.append(f"Context: {factor_text}")
            
            return " | ".join(filter(None, text_parts))
            
        except Exception as e:
            logger.warning(f"Failed to create pattern text: {e}")
            return json.dumps(pattern_data, default=str)[:500]  # Fallback
    
    async def cleanup(self):
        """Cleanup vector store resources"""
        try:
            # ChromaDB doesn't require explicit cleanup for HTTP client
            self.client = None
            self.collection = None
            logger.info("✅ Vector store cleaned up")
            
        except Exception as e:
            logger.error(f"Error during vector store cleanup: {e}")