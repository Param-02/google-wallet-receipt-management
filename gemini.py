"""
Receipt Chatbot Backend - FastAPI Implementation (Gemini + Local JSON)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
import os
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Receipt Chatbot API - Gemini", version="1.0.1")

# Available expense categories
EXPENSE_CATEGORIES = [
    "Groceries", "Food", "Transportation", "Travel", "Utilities",
    "Subscriptions", "Healthcare", "Shopping", "Entertainment",
    "Education", "Maintenance", "Financial", "Others"
]

# Load the parsed_receipt.json as the database
try:
    with open("parsed_receipt.json", "r", encoding="utf-8") as f:
        RECEIPT_DATA = json.load(f)
    if not isinstance(RECEIPT_DATA, list):
        raise ValueError("parsed_receipt.json must contain a JSON array of receipts.")
    logger.info(f"Loaded {len(RECEIPT_DATA)} receipts from parsed_receipt.json.")
except Exception as e:
    logger.error(f"Failed to load parsed_receipt.json: {e}")
    RECEIPT_DATA = []

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    categories_analyzed: List[str]
    receipts_count: int
    timestamp: str

class ReceiptAnalysisService:
    """Service class for receipt analysis and chatbot functionality"""

    def __init__(self):
        vertexai.init(
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "amplified-album-464909-m5"),
            location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        )
        self.gemini = GenerativeModel("gemini-2.5-flash")

    def _build_context(self) -> str:
        """Builds context string from RECEIPT_DATA"""
        if not RECEIPT_DATA:
            return "No receipts found."

        summaries = []
        for idx, receipt in enumerate(RECEIPT_DATA, start=1):
            if "Summary" in receipt:
                summaries.append(f"{idx}. {receipt['Summary']}")
            else:
                summaries.append(f"{idx}. {json.dumps(receipt)}")

        return "\n".join(summaries)

    async def classify_categories(self, user_query: str) -> List[str]:
        """
        Classify user query to determine relevant expense categories
        """
        prompt = f"""You are an expense category classifier. Given a user query about expenses or receipts, determine which expense categories are relevant to answer the question.

Available Categories:
{', '.join(EXPENSE_CATEGORIES)}

Query: "{user_query}"

Respond only with a JSON array of category names.
"""
        try:
            response = self.gemini.generate_content(prompt)
            result = response.text.strip()
            categories = json.loads(result)
            valid_categories = [cat for cat in categories if cat in EXPENSE_CATEGORIES]
            return valid_categories if valid_categories else ["Others"]
        except Exception as e:
            logger.error(f"Category classification error: {e}")
            return ["Others"]

    async def generate_answer(self, user_query: str, context: str) -> str:
        """
        Generate final answer using context and user query
        """
        prompt = f"""You are an intelligent expense analysis assistant. You help users understand their spending patterns by analyzing their receipt data.

Receipt Data:
{context}

User Question:
{user_query}

Be conversational but precise, include details from the receipt(s), show calculations if needed, and say if data is insufficient."""
        try:
            response = self.gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "I encountered an error while analyzing your receipts. Please try again."

    async def process_chat_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to process chat query through the full pipeline
        """
        try:
            relevant_categories = await self.classify_categories(query)
            context = self._build_context()
            answer = await self.generate_answer(query, context)

            return {
                "response": answer,
                "categories_analyzed": relevant_categories,
                "receipts_count": len(RECEIPT_DATA),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "categories_analyzed": [],
                "receipts_count": 0,
                "timestamp": datetime.now().isoformat()
            }

# Initialize the service
analysis_service = ReceiptAnalysisService()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for receipt analysis queries
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Processing chat query: {request.query}")
    result = await analysis_service.process_chat_query(request.query)
    return ChatResponse(**result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/categories")
async def get_categories():
    """Get available expense categories"""
    return {"categories": EXPENSE_CATEGORIES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
