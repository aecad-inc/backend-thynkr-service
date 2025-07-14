import logging
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Include, IncludeEnum, Where
from typing import Dict, List, Any, Optional, Union, cast
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
            if not self.collection:
                raise RuntimeError("Vector store not initialized")
                
            # Create text representation for embedding
            text_content = self._create_searchable_text(context_data, context_type)
            
            # Prepare metadata - ChromaDB requires string values
            metadata = {
                "context_type": str(context_type),
                "stored_at": str(context_data.get("stored_at", "")),
                "file_id": str(context_data.get("file_id", "")),
                "project_id": str(context_data.get("project_id", "")),
                "user_id": str(context_data.get("user_id", ""))
            }
            
            # Remove empty values from metadata
            metadata = {k: v for k, v in metadata.items() if v and v.strip() and v != "None"}
            
            # Store in ChromaDB
            self.collection.add(
                documents=[text_content],
                metadatas=[metadata],
                ids=[context_id]
            )
            
            # Also store full context data as JSON in a separate document
            full_context_id = f"{context_id}_full"
            full_metadata = {**metadata, "is_full_context": "true"}
            
            self.collection.add(
                documents=[json.dumps(context_data, default=str)],
                metadatas=[full_metadata],
                ids=[full_context_id]
            )
            
            logger.debug(f"Stored context: {context_id}")
            
        except Exception as e:
            logger.error(f"Failed to store context {context_id}: {e}")
            raise
    
    async def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context by ID"""
        try:
            if not self.collection:
                raise RuntimeError("Vector store not initialized")
                
            full_context_id = f"{context_id}_full"
            
            # Query for full context using proper include types
            results = self.collection.get(
                ids=[full_context_id],
                include=[IncludeEnum.documents, IncludeEnum.metadatas]
            )
            
            # Check if results and documents exist
            if not results or "documents" not in results:
                return None
                
            documents = results["documents"]
            if not documents or len(documents) == 0:
                return None
                
            context_json = documents[0]
            if not context_json:
                return None
                
            return json.loads(context_json)
            
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
            if not self.collection:
                raise RuntimeError("Vector store not initialized")
                
            # Create searchable text for the pattern
            text_content = self._create_pattern_text(pattern_data)
            
            # Prepare metadata - ensure all values are strings
            metadata = {
                "pattern_type": str(pattern_data.get("pattern_type", "unknown")),
                "layer_name": str(pattern_data.get("layer_name", "")),
                "model": str(pattern_data.get("model", "")),
                "confidence_improvement": str(pattern_data.get("confidence_improvement", 0.0)),
                "stored_at": str(pattern_data.get("stored_at", ""))
            }
            
            # Clean metadata - only keep non-empty values
            metadata = {k: v for k, v in metadata.items() if v and v.strip() and v != "None"}
            
            # Store pattern
            self.collection.add(
                documents=[text_content],
                metadatas=[metadata],
                ids=[pattern_id]
            )
            
            # Store full pattern data
            full_pattern_id = f"{pattern_id}_full"
            full_metadata = {**metadata, "is_full_pattern": "true"}
            
            self.collection.add(
                documents=[json.dumps(pattern_data, default=str)],
                metadatas=[full_metadata],
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
            if not self.collection:
                raise RuntimeError("Vector store not initialized")
                
            # Build where clause for filtering with proper typing
            where_clause: Optional[Where] = None
            if pattern_type:
                where_clause = cast(Where, {"pattern_type": pattern_type})
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(limit * 2, 100),  # Get more results to filter, but cap at 100
                where=where_clause,
                include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances]
            )
            
            # Validate results structure
            if not results:
                logger.warning("No query results returned")
                return []
            
            # Check required keys exist
            required_keys = ["documents", "metadatas", "distances", "ids"]
            if not all(key in results for key in required_keys):
                logger.warning(f"Missing required keys in results: {results.keys()}")
                return []
            
            # Extract result arrays with proper null checking
            documents = results.get("documents")
            metadatas = results.get("metadatas")
            distances = results.get("distances")
            ids = results.get("ids")
            
            # Validate all arrays exist and have content
            if not all([documents, metadatas, distances, ids]):
                logger.warning("One or more result arrays are empty or None")
                return []
            
            # Check if nested arrays exist and are not empty
            if (not documents or not documents[0] or 
                not metadatas or not metadatas[0] or
                not distances or not distances[0] or 
                not ids or not ids[0]):
                logger.warning("One or more nested result arrays are empty")
                return []
            
            patterns = []
            seen_base_ids = set()
            
            # Process results safely with type guards
            if not (documents and len(documents) > 0 and documents[0] and
                    metadatas and len(metadatas) > 0 and metadatas[0] and
                    distances and len(distances) > 0 and distances[0] and
                    ids and len(ids) > 0 and ids[0]):
                return []
                
            doc_list = documents[0]
            meta_list = metadatas[0]
            dist_list = distances[0]
            id_list = ids[0]
            
            # Ensure all lists have the same length
            min_length = min(len(doc_list), len(meta_list), len(dist_list), len(id_list))
            
            for i in range(min_length):
                try:
                    doc = doc_list[i]
                    metadata = meta_list[i]
                    distance = dist_list[i]
                    pattern_id = id_list[i]
                    
                    # Skip full pattern documents in similarity search
                    if metadata and metadata.get("is_full_pattern") == "true":
                        continue
                    
                    # Skip duplicates
                    if pattern_id in seen_base_ids:
                        continue
                    seen_base_ids.add(pattern_id)
                    
                    # Get full pattern data
                    full_pattern = await self.get_pattern(pattern_id)
                    if full_pattern:
                        full_pattern["similarity_score"] = max(0.0, 1.0 - distance)  # Ensure non-negative
                        patterns.append(full_pattern)
                    
                    if len(patterns) >= limit:
                        break
                        
                except (IndexError, TypeError, KeyError) as e:
                    logger.warning(f"Error processing result {i}: {e}")
                    continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
            return []
    
    async def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get full pattern data by ID"""
        try:
            if not self.collection:
                raise RuntimeError("Vector store not initialized")
                
            full_pattern_id = f"{pattern_id}_full"
            
            results = self.collection.get(
                ids=[full_pattern_id],
                include=[IncludeEnum.documents]
            )
            
            if (results and 
                "documents" in results and 
                results["documents"] and 
                len(results["documents"]) > 0):
                
                pattern_json = results["documents"][0]
                if pattern_json:
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
            if not self.collection:
                raise RuntimeError("Vector store not initialized")
                
            # Build where clause with proper typing
            where_clause: Optional[Where] = None
            if context_types:
                where_clause = cast(Where, {"context_type": {"$in": context_types}})
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=min(limit * 2, 100),  # Cap at 100 results
                where=where_clause,
                include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances]
            )
            
            # Validate results structure
            if not results:
                return []
                
            required_keys = ["documents", "metadatas", "distances", "ids"]
            if not all(key in results for key in required_keys):
                logger.warning(f"Missing required keys in search results: {results.keys()}")
                return []
                
            # Extract result arrays with proper null checking
            documents = results.get("documents")
            metadatas = results.get("metadatas")
            distances = results.get("distances")
            ids = results.get("ids")
            
            # Validate arrays exist and are not None
            if not all([documents, metadatas, distances, ids]):
                return []
            
            # Check if nested arrays exist and are not empty
            if (not documents or not documents[0] or 
                not metadatas or not metadatas[0] or
                not distances or not distances[0] or 
                not ids or not ids[0]):
                return []
            
            search_results = []
            seen_base_ids = set()
            
            # Process results safely with type guards
            if not (documents and len(documents) > 0 and documents[0] and
                    metadatas and len(metadatas) > 0 and metadatas[0] and
                    distances and len(distances) > 0 and distances[0] and
                    ids and len(ids) > 0 and ids[0]):
                return []
                
            doc_list = documents[0]
            meta_list = metadatas[0]
            dist_list = distances[0]
            id_list = ids[0]
            
            min_length = min(len(doc_list), len(meta_list), len(dist_list), len(id_list))
            
            for i in range(min_length):
                try:
                    doc = doc_list[i]
                    metadata = meta_list[i]
                    distance = dist_list[i]
                    context_id = id_list[i]
                    
                    # Skip full context documents in search results
                    if (metadata and 
                        (metadata.get("is_full_context") == "true" or 
                         metadata.get("is_full_pattern") == "true")):
                        continue
                    
                    if context_id in seen_base_ids:
                        continue
                    seen_base_ids.add(context_id)
                    
                    # Get full context
                    full_context = await self.get_context(context_id)
                    if full_context:
                        search_result = {
                            "context_id": context_id,
                            "similarity_score": max(0.0, 1.0 - distance),  # Ensure non-negative
                            "context_type": metadata.get("context_type") if metadata else None,
                            "preview": doc[:200] + "..." if len(doc) > 200 else doc,
                            "full_context": full_context
                        }
                        search_results.append(search_result)
                    
                    if len(search_results) >= limit:
                        break
                        
                except (IndexError, TypeError, KeyError) as e:
                    logger.warning(f"Error processing search result {i}: {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored contexts"""
        try:
            if not self.collection:
                raise RuntimeError("Vector store not initialized")
                
            # Get collection info
            collection_count = self.collection.count()
            
            # Get breakdown by context type
            try:
                all_results = self.collection.get(include=[IncludeEnum.metadatas])
                all_metadata = all_results.get("metadatas") if all_results else None
            except Exception as e:
                logger.warning(f"Failed to get all metadata: {e}")
                all_metadata = None
            
            context_type_counts = {}
            pattern_type_counts = {}
            
            if all_metadata:
                for metadata in all_metadata:
                    if not metadata:
                        continue
                        
                    # Skip full documents in statistics
                    if (metadata.get("is_full_context") == "true" or 
                        metadata.get("is_full_pattern") == "true"):
                        continue
                    
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
            return {
                "total_documents": 0,
                "context_type_breakdown": {},
                "pattern_type_breakdown": {},
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "error": str(e)
            }
    
    def _create_searchable_text(self, context_data: Dict[str, Any], context_type: str) -> str:
        """Create searchable text from context data"""
        try:
            text_parts = []
            
            # Add context type
            text_parts.append(f"Context type: {context_type}")
            
            # Add key fields based on context type
            if context_type == "file":
                if "file_name" in context_data and context_data["file_name"]:
                    text_parts.append(f"File: {context_data['file_name']}")
                if "project_context" in context_data:
                    project_ctx = context_data["project_context"]
                    if isinstance(project_ctx, dict):
                        if project_ctx.get("project_type"):
                            text_parts.append(f"Project type: {project_ctx['project_type']}")
                        if project_ctx.get("project_description"):
                            text_parts.append(f"Description: {project_ctx['project_description']}")
            
            elif context_type == "feedback":
                feedback_parts = []
                if context_data.get('layer_name'):
                    feedback_parts.append(f"Layer: {context_data['layer_name']}")
                if context_data.get('original_prediction'):
                    feedback_parts.append(f"Original: {context_data['original_prediction']}")
                if context_data.get('user_correction'):
                    feedback_parts.append(f"Correction: {context_data['user_correction']}")
                if context_data.get('comments'):
                    feedback_parts.append(f"Comments: {context_data['comments']}")
                text_parts.extend(feedback_parts)
            
            elif context_type == "session":
                if "conversation_history" in context_data:
                    history = context_data["conversation_history"]
                    if isinstance(history, list) and history:
                        recent_queries = []
                        for h in history[-3:]:  # Last 3 interactions
                            if isinstance(h, dict) and h.get("user_query"):
                                recent_queries.append(h["user_query"])
                        if recent_queries:
                            text_parts.append(f"Recent queries: {' '.join(recent_queries)}")
            
            # Add any additional text content
            for key in ["content", "description", "notes", "reasoning"]:
                if key in context_data and context_data[key]:
                    text_parts.append(str(context_data[key]))
            
            result = " | ".join(filter(None, text_parts))
            return result if result else "No searchable content"
            
        except Exception as e:
            logger.warning(f"Failed to create searchable text: {e}")
            # Safe fallback
            try:
                fallback = json.dumps(context_data, default=str)[:500]
                return fallback if fallback else "No content available"
            except Exception:
                return "No content available"
    
    def _create_pattern_text(self, pattern_data: Dict[str, Any]) -> str:
        """Create searchable text for patterns"""
        try:
            text_parts = []
            
            # Add pattern type
            pattern_type = pattern_data.get("pattern_type", "unknown")
            text_parts.append(f"Pattern: {pattern_type}")
            
            # Add layer information
            if pattern_data.get("layer_name"):
                text_parts.append(f"Layer: {pattern_data['layer_name']}")
            
            # Add prediction information
            if "original_predictions" in pattern_data:
                predictions = pattern_data["original_predictions"]
                if isinstance(predictions, list) and predictions:
                    pred_parts = []
                    for p in predictions:
                        if isinstance(p, dict):
                            model = p.get('model', '')
                            prediction = p.get('prediction', '')
                            if model and prediction:
                                pred_parts.append(f"{model}: {prediction}")
                    if pred_parts:
                        text_parts.append(f"Original predictions: {', '.join(pred_parts)}")
            
            # Add enhanced prediction
            if pattern_data.get("enhanced_prediction"):
                text_parts.append(f"Enhanced: {pattern_data['enhanced_prediction']}")
            
            # Add reasoning
            if "reasoning" in pattern_data:
                reasoning = pattern_data["reasoning"]
                if isinstance(reasoning, list) and reasoning:
                    text_parts.append(f"Reasoning: {' '.join(str(r) for r in reasoning)}")
                elif isinstance(reasoning, str) and reasoning:
                    text_parts.append(f"Reasoning: {reasoning}")
            
            # Add context factors
            if "context_factors" in pattern_data:
                factors = pattern_data["context_factors"]
                if isinstance(factors, dict):
                    factor_parts = []
                    for k, v in factors.items():
                        if v:
                            factor_parts.append(f"{k}: {v}")
                    if factor_parts:
                        text_parts.append(f"Context: {', '.join(factor_parts)}")
            
            result = " | ".join(filter(None, text_parts))
            return result if result else "No pattern content"
            
        except Exception as e:
            logger.warning(f"Failed to create pattern text: {e}")
            # Safe fallback
            try:
                fallback = json.dumps(pattern_data, default=str)[:500]
                return fallback if fallback else "No content available"
            except Exception:
                return "No content available"
    
    async def cleanup(self):
        """Cleanup vector store resources"""
        try:
            # ChromaDB doesn't require explicit cleanup for HTTP client
            self.client = None
            self.collection = None
            logger.info("✅ Vector store cleaned up")
            
        except Exception as e:
            logger.error(f"Error during vector store cleanup: {e}")