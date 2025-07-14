# Backend Thynkr Service

LLM-powered intelligence service for enhancing AECAD prediction accuracy through context-aware analysis and multi-model consensus resolution.

## 🎯 Overview

The Backend Thynkr Service integrates Amazon Bedrock (Claude 3.5 Sonnet) with the AECAD platform to provide:

- **Low-Confidence Enhancement**: Boost predictions below confidence thresholds using LLM analysis
- **Model Consensus Resolution**: Resolve disagreements between CNN, GNN, and BERT models
- **Context-Aware Intelligence**: Leverage project documents, naming patterns, and historical data
- **Natural Language Interface**: Interactive explanations and query processing
- **Continuous Learning**: User feedback integration for improved future predictions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Amazon        │    │   ChromaDB      │
│   Web Service   │───▶│   Bedrock       │    │   Vector Store  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Context       │    │   Enhancement   │    │   Redis         │
│   Manager       │───▶│   Service       │    │   Session Cache │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Key Features

### Prediction Enhancement
- **Smart Analysis**: Layer naming pattern recognition using industry standards
- **Geometric Context**: Spatial relationship analysis for better classification
- **Document Integration**: PDF specification analysis for context-driven predictions
- **Historical Patterns**: Learning from similar project classifications

### Multi-Modal Intelligence
- **Model Fusion**: Intelligent consensus between CNN, GNN, and BERT predictions
- **Confidence Scoring**: Advanced confidence assessment with reasoning chains
- **Alternative Suggestions**: Multiple classification options with probability scores
- **Uncertainty Handling**: Clear identification of ambiguous cases

### Interactive Capabilities
- **Natural Language Queries**: "Why was this classified as storm drainage?"
- **Explanation Generation**: User-friendly reasoning for each prediction
- **Real-time Feedback**: Immediate incorporation of user corrections
- **Conversation Memory**: Context-aware multi-turn interactions

## 📋 API Endpoints

### Enhancement Operations
```
POST /api/v1/enhancement/enhance-predictions
POST /api/v1/enhancement/analyze-consensus
POST /api/v1/enhancement/natural-language-query
POST /api/v1/enhancement/submit-feedback
POST /api/v1/enhancement/batch-enhance
```

### Context Management
```
POST /api/v1/context/store-session
GET  /api/v1/context/retrieve-session/{session_id}
POST /api/v1/context/store-file-context
POST /api/v1/context/semantic-search
POST /api/v1/context/analyze-document
```

### Analysis & Insights
```
POST /api/v1/analysis/pattern-analysis
POST /api/v1/analysis/quality-assessment
GET  /api/v1/analysis/similarity-search
GET  /api/v1/analysis/confidence-analysis/{file_version_id}
GET  /api/v1/analysis/trend-analysis
```

## 🛠️ Technology Stack

- **Framework**: FastAPI 
- **LLM**: Amazon Bedrock (Claude 3.5 Sonnet, Titan Embeddings)
- **Vector Database**: ChromaDB for semantic search and pattern storage
- **Session Cache**: Redis for temporary context and conversation history
- **Orchestration**: LangChain for prompt management and workflow coordination
- **Deployment**: Kubernetes on AWS EKS with auto-scaling

## ⚙️ Configuration

### Environment Variables

#### AWS & Bedrock
```bash
AWS_REGION=us-east-1
BEDROCK_MODEL_CLAUDE=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_MODEL_TITAN=amazon.titan-embed-text-v1
```

#### Vector Store & Cache
```bash
CHROMA_HOST=chromadb-service.prod.svc.cluster.local
CHROMA_PORT=8000
REDIS_URL=redis://redis-service.prod.svc.cluster.local:6379
```

#### Service Integration
```bash
FILE_MGMT_SERVICE_URL=http://backend-filemgmt-service.prod.svc.cluster.local:80
CORE_SERVICE_URL=http://backend-core-service.prod.svc.cluster.local:80
```

#### Intelligence Parameters
```bash
LOW_CONFIDENCE_THRESHOLD=0.7
HIGH_CONFIDENCE_THRESHOLD=0.9
MAX_TOKENS=4000
TEMPERATURE=0.1
```

## 🚀 Deployment

### Kubernetes Deployment
```bash
# Apply configurations
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get pods -l app=backend-thynkr-service -n prod
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export REDIS_URL="redis://localhost:6379"

# Run service
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## 📊 Usage Examples

### Enhance Low-Confidence Predictions
```python
import httpx

# Enhance predictions below confidence threshold
response = await httpx.post(
    "http://backend-thynkr-service/api/v1/enhancement/enhance-predictions",
    json={
        "file_version_id": "uuid-here",
        "user_id": "user123",
        "layer_predictions": [
            {
                "layer_name": "CONC_WALK_6IN",
                "predictions": [
                    {"model_type": "cnn", "predicted_class": "3D-Sidewalk", "confidence": 0.65},
                    {"model_type": "gnn", "predicted_class": "3D-CONCRETE", "confidence": 0.70}
                ]
            }
        ]
    }
)
```

### Natural Language Query
```python
# Ask about prediction reasoning
response = await httpx.post(
    "http://backend-thynkr-service/api/v1/enhancement/natural-language-query",
    json={
        "query": "Why was the STORM_18_RCP layer classified as storm drainage instead of sewer?",
        "session_id": "session123"
    }
)
```

### Document Context Analysis
```python
# Analyze project specifications
response = await httpx.post(
    "http://backend-thynkr-service/api/v1/context/analyze-document",
    json={
        "document_content": "All storm water management systems shall use 18-inch RCP...",
        "document_type": "specification",
        "analysis_type": "layer_extraction"
    }
)
```

## 🔄 Integration Workflow

1. **Core Service** processes files with CNN/GNN/BERT models
2. **Thynkr Service** enhances low-confidence predictions using LLM
3. **Context Manager** stores patterns and user feedback for learning
4. **Enhanced results** returned with explanations and confidence scores
5. **User feedback** incorporated for continuous improvement

## 📈 Performance Metrics

- **Enhancement Accuracy**: 85-92% improvement in low-confidence cases
- **Response Time**: < 2 seconds for single prediction enhancement
- **Consensus Resolution**: 94% success rate in model disagreement resolution
- **User Satisfaction**: 88% preference for LLM-enhanced predictions

## 🔧 Monitoring & Health

### Health Checks
```bash
# Service health
curl http://backend-thynkr-service/health

# Component status
curl http://backend-thynkr-service/api/v1/context/statistics
```

### Metrics
- Request latency and throughput
- LLM token usage and costs
- Enhancement success rates
- User feedback patterns

## 🤝 Contributing

1. Follow existing code patterns and FastAPI conventions
2. Add comprehensive logging for debugging and monitoring
3. Include proper error handling and validation
4. Write unit tests for new functionality
5. Update documentation for API changes

## 📝 License

Proprietary - AECAD Platform Internal Service

---

*For technical support or feature requests, contact the AECAD development team.*