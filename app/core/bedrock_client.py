import boto3
import json
import logging
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config.settings import settings

logger = logging.getLogger(__name__)

class BedrockClient:
    """AWS Bedrock client for LLM interactions"""
    
    def __init__(self):
        self.bedrock_client = None
        self.bedrock_runtime = None
        self.embedding_client = None
        
    async def initialize(self):
        """Initialize Bedrock clients"""
        try:
            # Initialize Bedrock client for model management
            self.bedrock_client = boto3.client(
                'bedrock',
                region_name=settings.BEDROCK_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            
            # Initialize Bedrock Runtime client for inference
            self.bedrock_runtime = boto3.client(
                'bedrock-runtime',
                region_name=settings.BEDROCK_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            
            logger.info("✅ Bedrock clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bedrock clients: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def invoke_claude(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Invoke Claude model with retry logic"""
        
        if not self.bedrock_runtime:
            raise Exception("Bedrock runtime client not initialized")
        
        try:
            # Prepare the request body
            messages = [{"role": "user", "content": prompt}]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or settings.MAX_TOKENS,
                "temperature": temperature or settings.TEMPERATURE,
                "top_p": settings.TOP_P,
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            # Make the API call
            response = self.bedrock_runtime.invoke_model(
                modelId=settings.BEDROCK_MODEL_CLAUDE,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            return {
                "success": True,
                "content": response_body['content'][0]['text'],
                "usage": response_body.get('usage', {}),
                "model": settings.BEDROCK_MODEL_CLAUDE
            }
            
        except Exception as e:
            logger.error(f"❌ Claude invocation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings using Titan model"""
        
        if not self.bedrock_runtime:
            raise Exception("Bedrock runtime client not initialized")
        
        try:
            embeddings = []
            
            for text in texts:
                body = {
                    "inputText": text
                }
                
                response = self.bedrock_runtime.invoke_model(
                    modelId=settings.BEDROCK_MODEL_TITAN,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json"
                )
                
                response_body = json.loads(response['body'].read())
                embeddings.append(response_body['embedding'])
            
            return {
                "success": True,
                "embeddings": embeddings,
                "model": settings.BEDROCK_MODEL_TITAN
            }
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "embeddings": []
            }
    
    async def invoke_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke Claude with structured JSON output"""
        
        structured_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Response:
"""
        
        response = await self.invoke_claude(
            prompt=structured_prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Lower temperature for structured output
        )
        
        if response["success"]:
            try:
                # Parse JSON response
                json_content = json.loads(response["content"])
                response["structured_content"] = json_content
                return response
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse structured output: {e}")
                response["success"] = False
                response["error"] = f"Invalid JSON response: {e}"
                
        return response
    
    async def analyze_layer_confidence(
        self,
        layer_name: str,
        predictions: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze layer predictions and provide confidence enhancement"""
        
        system_prompt = """You are an expert CAD layer classification assistant for the AECAD system. 
Your role is to analyze layer naming patterns, geometric context, and model predictions to provide 
accurate layer classifications with confidence scores."""
        
        context_str = ""
        if context:
            context_str = f"\nProject Context: {json.dumps(context, indent=2)}"
        
        prompt = f"""
Analyze this layer classification scenario:

Layer Name: "{layer_name}"
Model Predictions:
- CNN: {predictions.get('cnn', 'N/A')}
- GNN: {predictions.get('gnn', 'N/A')} 
- BERT: {predictions.get('bert', 'N/A')}
{context_str}

Please provide your analysis in the following format:
1. Most likely classification with confidence score (0.0-1.0)
2. Reasoning based on layer name patterns and context
3. Alternative classifications if confidence is low
4. Recommendations for improving classification accuracy

Focus on industry-standard layer naming conventions and geometric context clues.
"""
        
        return await self.invoke_claude(prompt=prompt, system_prompt=system_prompt)
    
    async def explain_prediction(
        self,
        layer_name: str,
        final_prediction: str,
        confidence: float,
        reasoning_chain: List[str]
    ) -> Dict[str, Any]:
        """Generate user-friendly explanation for predictions"""
        
        system_prompt = """You are explaining CAD layer classification decisions to users. 
Provide clear, concise explanations that help users understand why certain predictions were made."""
        
        prompt = f"""
Explain this layer classification decision:

Layer Name: "{layer_name}"
Final Prediction: "{final_prediction}"
Confidence: {confidence:.2f}

Reasoning Chain:
{chr(10).join(f"- {step}" for step in reasoning_chain)}

Provide a clear, user-friendly explanation of:
1. Why this classification was chosen
2. What factors influenced the decision
3. How confident the system is and why
4. Any alternative interpretations considered

Keep the explanation concise but informative for engineering professionals.
"""
        
        return await self.invoke_claude(prompt=prompt, system_prompt=system_prompt)
    
    async def cleanup(self):
        """Cleanup resources"""
        self.bedrock_client = None
        self.bedrock_runtime = None
        logger.info("✅ Bedrock clients cleaned up")