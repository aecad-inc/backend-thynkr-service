from fastapi import APIRouter
from .routes import enhancement, context, analysis

api_router = APIRouter()

# Include all route modules
api_router.include_router(
    enhancement.router, 
    prefix="/enhancement", 
    tags=["Enhancement"]
)

api_router.include_router(
    context.router, 
    prefix="/context", 
    tags=["Context Management"]
)

api_router.include_router(
    analysis.router, 
    prefix="/analysis", 
    tags=["Analysis & Insights"]
)