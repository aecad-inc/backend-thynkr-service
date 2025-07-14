import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from app.core.bedrock_client import BedrockClient
from app.core.context_manager import ContextManager
from app.models.requests import LayerPredictions, ModelPrediction, FileContext
from app.config.settings import settings
import asyncio

logger = logging.getLogger(__name__)

class EnhancementService:
    """Service for enhancing model predictions using LLM intelligence"""
    
    def __init__(self, bedrock_client: BedrockClient, context_manager: ContextManager):
        self.bedrock_client = bedrock_client
        self.context_manager = context_manager
        
    async def enhance_low_confidence_predictions(
        self,
        layer_predictions: List[LayerPredictions],
        file_context: Optional[FileContext] = None
    ) -> Dict[str, Any]:
        """Enhance predictions with confidence below threshold"""
        
        enhanced_results = []
        enhancement_summary = {
            "total_layers": len(layer_predictions),
            "enhanced_count": 0,
            "skipped_count": 0,
            "improvements": []
        }
        
        for layer_pred in layer_predictions:
            # Determine if enhancement is needed
            max_confidence = max(pred.confidence for pred in layer_pred.predictions)
            
            if max_confidence < settings.LOW_CONFIDENCE_THRESHOLD:
                # Enhance this prediction
                enhanced_result = await self._enhance_single_prediction(
                    layer_pred, file_context
                )
                enhanced_results.append(enhanced_result)
                enhancement_summary["enhanced_count"] += 1
                
                if enhanced_result.get("improved", False):
                    enhancement_summary["improvements"].append({
                        "layer_name": layer_pred.layer_name,
                        "original_confidence": max_confidence,
                        "enhanced_confidence": enhanced_result.get("final_confidence", max_confidence),
                        "reasoning": enhanced_result.get("reasoning", "")
                    })
            else:
                # Skip enhancement for high-confidence predictions
                enhanced_results.append({
                    "layer_name": layer_pred.layer_name,
                    "enhanced": False,
                    "reason": "High confidence - no enhancement needed",
                    "original_predictions": layer_pred.predictions,
                    "final_prediction": max(layer_pred.predictions, key=lambda x: x.confidence)
                })
                enhancement_summary["skipped_count"] += 1
        
        return {
            "enhanced_predictions": enhanced_results,
            "summary": enhancement_summary
        }
    
    async def _enhance_single_prediction(
        self,
        layer_pred: LayerPredictions,
        file_context: Optional[FileContext] = None
    ) -> Dict[str, Any]:
        """Enhance a single layer prediction using LLM analysis"""
        
        try:
            # Prepare context for LLM
            context_data = await self._prepare_enhancement_context(layer_pred, file_context)
            
            # Create enhancement prompt
            prompt = self._create_enhancement_prompt(layer_pred, context_data)
            
            # Get LLM analysis
            llm_response = await self.bedrock_client.invoke_structured_output(
                prompt=prompt,
                schema=self._get_enhancement_schema(),
                system_prompt=self._get_enhancement_system_prompt()
            )
            
            if llm_response["success"]:
                structured_result = llm_response["structured_content"]
                
                # Process LLM recommendation
                enhanced_result = await self._process_llm_recommendation(
                    layer_pred, structured_result, context_data
                )
                
                return enhanced_result
            else:
                logger.error(f"LLM enhancement failed: {llm_response.get('error')}")
                return {
                    "layer_name": layer_pred.layer_name,
                    "enhanced": False,
                    "error": llm_response.get("error"),
                    "original_predictions": layer_pred.predictions
                }
                
        except Exception as e:
            logger.error(f"Enhancement error for {layer_pred.layer_name}: {e}")
            return {
                "layer_name": layer_pred.layer_name,
                "enhanced": False,
                "error": str(e),
                "original_predictions": layer_pred.predictions
            }
    
    async def _prepare_enhancement_context(
        self,
        layer_pred: LayerPredictions,
        file_context: Optional[FileContext] = None
    ) -> Dict[str, Any]:
        """Prepare comprehensive context for enhancement"""
        
        context = {
            "layer_name": layer_pred.layer_name,
            "geometric_context": layer_pred.geometric_context or {},
            "model_predictions": [
                {
                    "model": pred.model_type,
                    "prediction": pred.predicted_class,
                    "confidence": pred.confidence,
                    "probabilities": pred.probabilities
                }
                for pred in layer_pred.predictions
            ]
        }
        
        # Add file/project context if available
        if file_context:
            context["project_type"] = file_context.project_context.project_type if file_context.project_context else None
            context["file_name"] = file_context.file_name
            
            # Add document context insights
            if file_context.document_contexts:
                context["document_insights"] = [
                    {
                        "type": doc.document_type,
                        "content_summary": doc.content[:500] if doc.content else None
                    }
                    for doc in file_context.document_contexts[:3]  # Limit to first 3 docs
                ]
        
        # Retrieve similar patterns from vector store
        try:
            similar_patterns = await self.context_manager.find_similar_patterns(
                layer_pred.layer_name,
                limit=5
            )
            context["similar_patterns"] = similar_patterns
        except Exception as e:
            logger.warning(f"Could not retrieve similar patterns: {e}")
            context["similar_patterns"] = []
        
        return context
    
    def _create_enhancement_prompt(
        self,
        layer_pred: LayerPredictions,
        context_data: Dict[str, Any]
    ) -> str:
        """Create detailed prompt for LLM enhancement"""
        
        predictions_summary = "\n".join([
            f"- {pred['model'].upper()}: {pred['prediction']} (confidence: {pred['confidence']:.3f})"
            for pred in context_data["model_predictions"]
        ])
        
        similar_patterns_summary = ""
        if context_data.get("similar_patterns"):
            similar_patterns_summary = "\nSimilar patterns from historical data:\n" + "\n".join([
                f"- {pattern.get('layer_name', 'Unknown')}: {pattern.get('classification', 'Unknown')}"
                for pattern in context_data["similar_patterns"][:3]
            ])
        
        project_context = ""
        if context_data.get("project_type"):
            project_context = f"\nProject Type: {context_data['project_type']}"
        
        document_context = ""
        if context_data.get("document_insights"):
            document_context = "\nDocument Context:\n" + "\n".join([
                f"- {doc['type']}: {doc['content_summary'][:200]}..." if doc['content_summary'] else f"- {doc['type']}: No content"
                for doc in context_data["document_insights"]
            ])
        
        prompt = f"""
Analyze this CAD layer classification scenario that requires confidence enhancement:

LAYER INFORMATION:
Layer Name: "{layer_pred.layer_name}"
Geometric Context: {json.dumps(context_data.get('geometric_context', {}), indent=2)}

CURRENT MODEL PREDICTIONS:
{predictions_summary}

ADDITIONAL CONTEXT:
{project_context}
{document_context}
{similar_patterns_summary}

ANALYSIS REQUIRED:
1. Analyze the layer name for industry-standard patterns and conventions
2. Consider the geometric context and how it relates to layer classification
3. Evaluate the consistency and reliability of model predictions
4. Factor in project type and document context if relevant
5. Compare with similar historical patterns

Please provide your enhanced classification with detailed reasoning.
The goal is to improve prediction confidence for layers below {settings.LOW_CONFIDENCE_THRESHOLD} threshold.
"""
        
        return prompt
    
    def _get_enhancement_schema(self) -> Dict[str, Any]:
        """Get JSON schema for structured LLM response"""
        
        return {
            "type": "object",
            "properties": {
                "recommended_classification": {
                    "type": "string",
                    "description": "The recommended layer classification"
                },
                "confidence_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the recommendation"
                },
                "reasoning": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step-by-step reasoning for the classification"
                },
                "key_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key factors that influenced the decision"
                },
                "alternative_classifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "classification": {"type": "string"},
                            "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        }
                    },
                    "description": "Alternative classifications with probabilities"
                },
                "improvement_explanation": {
                    "type": "string",
                    "description": "Explanation of how this improves upon model predictions"
                },
                "uncertainty_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Factors that contribute to uncertainty"
                }
            },
            "required": [
                "recommended_classification",
                "confidence_score", 
                "reasoning",
                "key_factors"
            ]
        }
    
    def _get_enhancement_system_prompt(self) -> str:
        """Get system prompt for enhancement"""
        
        return f"""You are an expert CAD layer classification assistant specialized in enhancing low-confidence predictions for the AECAD system.

Your expertise includes:
- Industry-standard layer naming conventions across multiple domains
- Geometric context interpretation for CAD entities
- Pattern recognition in engineering drawings
- Project-specific classification standards

Available layer classifications:
{', '.join(settings.LAYER_LABELS)}

Your goal is to provide enhanced classifications with higher confidence than the original model predictions, backed by clear reasoning and industry knowledge.

Focus on:
1. Layer naming pattern analysis using industry standards
2. Geometric context clues that indicate layer purpose
3. Project type considerations
4. Historical pattern matching
5. Cross-model consensus analysis

Provide actionable improvements with confidence scores between 0.0 and 1.0."""
    
    async def _process_llm_recommendation(
        self,
        layer_pred: LayerPredictions,
        llm_result: Dict[str, Any],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process LLM recommendation and create final result"""
        
        original_max_confidence = max(pred.confidence for pred in layer_pred.predictions)
        enhanced_confidence = llm_result.get("confidence_score", original_max_confidence)
        
        # Determine if this is an improvement
        is_improvement = enhanced_confidence > original_max_confidence + 0.05  # 5% improvement threshold
        
        result = {
            "layer_name": layer_pred.layer_name,
            "enhanced": True,
            "improved": is_improvement,
            "original_max_confidence": original_max_confidence,
            "final_confidence": enhanced_confidence,
            "recommended_classification": llm_result.get("recommended_classification"),
            "reasoning": llm_result.get("reasoning", []),
            "key_factors": llm_result.get("key_factors", []),
            "alternative_classifications": llm_result.get("alternative_classifications", []),
            "improvement_explanation": llm_result.get("improvement_explanation", ""),
            "uncertainty_factors": llm_result.get("uncertainty_factors", []),
            "original_predictions": [
                {
                    "model": pred.model_type,
                    "prediction": pred.predicted_class,
                    "confidence": pred.confidence
                }
                for pred in layer_pred.predictions
            ],
            "enhancement_method": "llm_analysis"
        }
        
        # Store this enhancement for future learning
        try:
            await self._store_enhancement_result(layer_pred, result, context_data)
        except Exception as e:
            logger.warning(f"Could not store enhancement result: {e}")
        
        return result
    
    async def _store_enhancement_result(
        self,
        layer_pred: LayerPredictions,
        enhancement_result: Dict[str, Any],
        context_data: Dict[str, Any]
    ):
        """Store enhancement result for future learning"""
        
        storage_data = {
            "layer_name": layer_pred.layer_name,
            "original_predictions": context_data["model_predictions"],
            "enhanced_prediction": enhancement_result["recommended_classification"],
            "confidence_improvement": enhancement_result["final_confidence"] - enhancement_result["original_max_confidence"],
            "reasoning": enhancement_result["reasoning"],
            "context_factors": {
                "project_type": context_data.get("project_type"),
                "geometric_context": context_data.get("geometric_context"),
                "document_context_available": bool(context_data.get("document_insights"))
            },
            "timestamp": "2025-01-14T10:00:00Z"
        }
        
        await self.context_manager.store_enhancement_pattern(storage_data)
    
    async def analyze_model_consensus(
        self,
        layer_predictions: List[LayerPredictions],
        file_context: Optional[FileContext] = None
    ) -> Dict[str, Any]:
        """Analyze consensus across different models"""
        
        consensus_results = []
        
        for layer_pred in layer_predictions:
            if len(layer_pred.predictions) < 2:
                continue
                
            # Analyze agreement between models
            predictions_by_class = {}
            for pred in layer_pred.predictions:
                if pred.predicted_class not in predictions_by_class:
                    predictions_by_class[pred.predicted_class] = []
                predictions_by_class[pred.predicted_class].append(pred)
            
            # Check for consensus
            max_agreement = max(len(preds) for preds in predictions_by_class.values())
            has_consensus = max_agreement >= 2
            
            if not has_consensus:
                # Use LLM to resolve disagreement
                resolution_result = await self._resolve_model_disagreement(
                    layer_pred, file_context
                )
                consensus_results.append(resolution_result)
            else:
                # Models agree - validate consensus quality
                consensus_class = max(predictions_by_class.keys(), 
                                    key=lambda k: len(predictions_by_class[k]))
                consensus_preds = predictions_by_class[consensus_class]
                avg_confidence = sum(p.confidence for p in consensus_preds) / len(consensus_preds)
                
                consensus_results.append({
                    "layer_name": layer_pred.layer_name,
                    "consensus_achieved": True,
                    "consensus_classification": consensus_class,
                    "consensus_confidence": avg_confidence,
                    "agreeing_models": [p.model_type for p in consensus_preds],
                    "resolution_method": "model_agreement"
                })
        
        return {
            "consensus_analysis": consensus_results,
            "summary": {
                "total_analyzed": len(consensus_results),
                "consensus_achieved": sum(1 for r in consensus_results if r.get("consensus_achieved")),
                "llm_resolutions": sum(1 for r in consensus_results if r.get("resolution_method") == "llm_analysis")
            }
        }
    
    async def _resolve_model_disagreement(
        self,
        layer_pred: LayerPredictions,
        file_context: Optional[FileContext] = None
    ) -> Dict[str, Any]:
        """Use LLM to resolve disagreement between models"""
        
        try:
            # Prepare disagreement analysis context
            context_data = await self._prepare_enhancement_context(layer_pred, file_context)
            
            # Create disagreement resolution prompt
            prompt = self._create_disagreement_prompt(layer_pred, context_data)
            
            # Get LLM resolution
            llm_response = await self.bedrock_client.invoke_structured_output(
                prompt=prompt,
                schema=self._get_disagreement_schema(),
                system_prompt=self._get_disagreement_system_prompt()
            )
            
            if llm_response["success"]:
                structured_result = llm_response["structured_content"]
                
                return {
                    "layer_name": layer_pred.layer_name,
                    "consensus_achieved": True,
                    "consensus_classification": structured_result["resolved_classification"],
                    "consensus_confidence": structured_result["confidence_score"],
                    "resolution_method": "llm_analysis",
                    "resolution_reasoning": structured_result.get("reasoning", []),
                    "model_analysis": structured_result.get("model_analysis", {}),
                    "disagreement_factors": structured_result.get("disagreement_factors", [])
                }
            else:
                return {
                    "layer_name": layer_pred.layer_name,
                    "consensus_achieved": False,
                    "error": llm_response.get("error"),
                    "resolution_method": "failed"
                }
                
        except Exception as e:
            logger.error(f"Disagreement resolution error for {layer_pred.layer_name}: {e}")
            return {
                "layer_name": layer_pred.layer_name,
                "consensus_achieved": False,
                "error": str(e),
                "resolution_method": "failed"
            }
    
    def _create_disagreement_prompt(
        self,
        layer_pred: LayerPredictions,
        context_data: Dict[str, Any]
    ) -> str:
        """Create prompt for resolving model disagreements"""
        
        model_disagreements = "\n".join([
            f"- {pred['model'].upper()}: {pred['prediction']} (confidence: {pred['confidence']:.3f})"
            for pred in context_data["model_predictions"]
        ])
        
        prompt = f"""
Resolve this model disagreement for CAD layer classification:

LAYER: "{layer_pred.layer_name}"

MODEL DISAGREEMENTS:
{model_disagreements}

CONTEXT:
{json.dumps(context_data.get('geometric_context', {}), indent=2)}

The models disagree on classification. Analyze each model's prediction considering:
1. Layer naming conventions and patterns
2. Geometric context appropriateness 
3. Model-specific strengths and weaknesses
4. Industry standards and best practices

Provide a definitive resolution with clear reasoning for why one classification is most appropriate.
"""
        
        return prompt
    
    def _get_disagreement_schema(self) -> Dict[str, Any]:
        """Schema for disagreement resolution response"""
        
        return {
            "type": "object",
            "properties": {
                "resolved_classification": {
                    "type": "string",
                    "description": "The final resolved classification"
                },
                "confidence_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reasoning": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "model_analysis": {
                    "type": "object",
                    "description": "Analysis of each model's prediction"
                },
                "disagreement_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Factors causing the disagreement"
                }
            },
            "required": ["resolved_classification", "confidence_score", "reasoning"]
        }
    
    def _get_disagreement_system_prompt(self) -> str:
        """System prompt for disagreement resolution"""
        
        return """You are a senior CAD classification expert resolving disagreements between AI models.

Your role is to:
1. Analyze each model's prediction and reasoning
2. Consider the strengths/weaknesses of different model types (CNN, GNN, BERT)
3. Apply industry expertise to determine the most accurate classification
4. Provide clear justification for your decision

Model characteristics:
- CNN: Strong with visual/geometric patterns
- GNN: Good with spatial relationships and graph structures  
- BERT: Excellent with naming conventions and text patterns

Resolve disagreements based on which model's strength best matches the available evidence."""
    
    async def process_natural_language_query(
        self,
        query: str,
        file_context: Optional[Any] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process natural language queries about predictions"""
        try:
            # Build context for the query
            context_parts = []
            
            if file_context:
                context_parts.append(f"File context: {file_context}")
            
            if conversation_history:
                recent_history = conversation_history[-3:]  # Last 3 interactions
                history_text = "\n".join([
                    f"User: {h.get('user_query', '')}\nSystem: {h.get('system_response', '')}"
                    for h in recent_history
                ])
                context_parts.append(f"Recent conversation:\n{history_text}")
            
            if session_context:
                context_parts.append(f"Session context: {session_context}")
            
            context_str = "\n\n".join(context_parts)
            
            # Create query processing prompt
            system_prompt = """You are an expert assistant for the AECAD CAD layer classification system.
Your role is to answer user questions about predictions, classifications, and system decisions.

Provide clear, helpful explanations that:
1. Answer the user's specific question
2. Explain the reasoning behind classifications
3. Reference relevant context when available
4. Suggest actionable next steps when appropriate
5. Maintain a professional but friendly tone"""

            prompt = f"""
User Query: "{query}"

Available Context:
{context_str}

Please provide a comprehensive answer to the user's question, leveraging any available context to give specific, actionable information.
"""

            # Get LLM response
            llm_response = await self.bedrock_client.invoke_claude(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            if llm_response["success"]:
                return {
                    "success": True,
                    "response": llm_response["content"],
                    "confidence": 0.9,  # High confidence for direct Q&A
                    "supporting_evidence": [],
                    "related_suggestions": [
                        "Ask about specific layer classifications",
                        "Request explanation of confidence scores",
                        "Inquire about alternative classifications"
                    ]
                }
            else:
                return {
                    "success": False,
                    "response": "I'm sorry, I couldn't process your query at this time.",
                    "confidence": 0.0,
                    "error": llm_response.get("error")
                }
                
        except Exception as e:
            logger.error(f"Natural language query processing failed: {e}")
            return {
                "success": False,
                "response": "An error occurred while processing your query.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def generate_prediction_explanation(
        self,
        file_version_id: str,
        layer_name: str,
        prediction: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate detailed explanation for a specific prediction"""
        try:
            # Build explanation prompt
            system_prompt = """You are an expert CAD classification system explainer.
Your role is to provide clear, detailed explanations of classification decisions that help users understand:
1. Why a specific classification was chosen
2. What factors influenced the decision
3. How confident the system is and why
4. What alternatives were considered"""

            prompt = f"""
Explain this CAD layer classification decision:

Layer Name: "{layer_name}"
Predicted Classification: "{prediction}"
Confidence Score: {confidence:.2f}
File Version ID: {file_version_id}

Please provide a comprehensive explanation that covers:
1. The reasoning behind this classification
2. Key factors that influenced the decision (naming patterns, context, etc.)
3. Why the confidence is at this level
4. Any alternative classifications that were considered
5. Visual or contextual cues that support this decision

Make the explanation accessible to engineering professionals who may not be familiar with AI systems.
"""

            # Get explanation from LLM
            llm_response = await self.bedrock_client.invoke_claude(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )
            
            if llm_response["success"]:
                # Parse and structure the response
                explanation_text = llm_response["content"]
                
                return {
                    "success": True,
                    "explanation": explanation_text,
                    "key_factors": [
                        "Layer naming pattern analysis",
                        "Geometric context evaluation", 
                        "Industry standard comparison",
                        f"Confidence assessment ({confidence:.1%})"
                    ],
                    "confidence_breakdown": {
                        "naming_pattern": min(0.9, confidence + 0.1),
                        "context_match": confidence,
                        "historical_pattern": max(0.5, confidence - 0.1)
                    },
                    "alternatives": [
                        {"classification": "Alternative classification", "reasoning": "Would be considered if..."}
                    ],
                    "visual_cues": [
                        "Layer name indicates specific infrastructure type",
                        "Geometric properties align with classification",
                        "Context supports decision"
                    ]
                }
            else:
                return {
                    "success": False,
                    "explanation": "Unable to generate explanation at this time.",
                    "error": llm_response.get("error")
                }
                
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {
                "success": False,
                "explanation": "An error occurred while generating the explanation.",
                "error": str(e)
            }