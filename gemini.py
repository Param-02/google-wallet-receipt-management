"""
Receipt Chatbot Backend - FastAPI Implementation (Gemini + Firebase)
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
import firebase_admin
from firebase_admin import credentials, firestore

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
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "splendid-yeti-464913-j2"),
            location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        )
        self.gemini = GenerativeModel("gemini-2.5-flash")
        self.db = self._init_firestore()
        self.receipt_data = self._load_receipt_data()

    def list_receipts(self) -> str:
        """Return a human readable list of all receipts."""
        if not self.receipt_data:
            return "No receipts found."

        lines = []
        for r in self.receipt_data:
            store = r.get("store_name", "Unknown")
            category = r.get("receipt_category", "Unknown")
            total = r.get("total_amount", 0)
            currency = r.get("currency", "")
            date = r.get("date", "Unknown")
            lines.append(f"* {category}: {currency}{total} ({store} - {date})")

        # Calculate total per currency
        totals = {}
        for r in self.receipt_data:
            currency = r.get("currency", "")
            try:
                amount = float(r.get("total_amount", 0))
            except (ValueError, TypeError):
                amount = 0
            totals[currency] = totals.get(currency, 0) + amount

        total_str = " + ".join(
            f"{cur}{amt}" for cur, amt in totals.items()
        )

        return "\n".join(lines) + f"\n\n📊 **Total:** {total_str} 💰"


    def _init_firestore(self):
        """Initialize Firebase and return Firestore client."""
        try:
            if not firebase_admin._apps:
                cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            raise

    def _load_receipt_data(self) -> List[Dict[str, Any]]:
        """Load receipt data from Firestore."""
        try:
            docs = list(self.db.collection("receipts").stream())
            data = [doc.to_dict() for doc in docs]
            logger.info(f"Loaded {len(data)} receipts from Firestore.")
            return data
        except Exception as e:
            logger.error(f"Failed to load receipts from Firestore: {e}")
            return []

    def reload_receipt_data(self):
        """Reload receipt data from Firestore."""
        self.receipt_data = self._load_receipt_data()
        return len(self.receipt_data)

    def _build_context(self) -> str:
        """Builds context string from receipt data"""
        if not self.receipt_data:
            return "No receipts found."

        summaries = []
        for idx, receipt in enumerate(self.receipt_data, start=1):
            # Include receipt category in context
            store = receipt.get('store_name', 'Unknown Store')
            category = receipt.get('receipt_category', 'Unknown Category')
            total = receipt.get('total_amount', '0')
            currency = receipt.get('currency', '')
            date = receipt.get('date', 'Unknown Date')
            
            context_line = f"{idx}. {store} ({category}) - {currency}{total} on {date}"
            if receipt.get('Summary'):
                context_line += f" - {receipt['Summary']}"
            summaries.append(context_line)

        return "\n".join(summaries)

    async def classify_categories(self, user_query: str) -> List[str]:
        """
        Classify user query to determine relevant expense categories
        """
        prompt = f"""You are an expense category classifier. Analyze the user query and determine which specific expense categories are being asked about.

Available Categories:
{', '.join(EXPENSE_CATEGORIES)}

User Query: "{user_query}"

Instructions:
- If the query mentions specific categories (like "groceries", "transportation", "food"), return those exact category names
- If the query asks about "total spending" or "all expenses", return ALL relevant categories found in the data
- If the query is about comparing categories, return the categories being compared
- Only use "Others" if the query is truly about miscellaneous/unspecified expenses
- Return categories that are actually relevant to answering the question

Examples:
- "transportation expenses" → ["Transportation"]
- "grocery and food spending" → ["Groceries", "Food"] 
- "total spend on groceries and transportation" → ["Transportation", "Groceries"]
- "how much did I spend total" → (analyze data to find all relevant categories)

Respond with ONLY a JSON array of category names from the available list above."""
        
        try:
            response = self.gemini.generate_content(prompt)
            result = response.text.strip()
            
            # Clean the response text (remove any markdown formatting)
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            categories = json.loads(result)
            valid_categories = [cat for cat in categories if cat in EXPENSE_CATEGORIES]
            
            # If no valid categories found or empty, try to extract from receipt data
            if not valid_categories:
                # Analyze receipt data to find relevant categories
                receipt_categories = set()
                for receipt in self.receipt_data:
                    if receipt.get('receipt_category'):
                        receipt_categories.add(receipt['receipt_category'])
                
                # Check if query is asking about total/all spending
                query_lower = user_query.lower()
                if any(word in query_lower for word in ['total', 'all', 'spent', 'expenses']):
                    valid_categories = list(receipt_categories) if receipt_categories else ["Others"]
                else:
                    valid_categories = ["Others"]
            
            return valid_categories
            
        except Exception as e:
            logger.error(f"Category classification error: {e}")
            # Fallback: try to extract categories from receipt data
            receipt_categories = set()
            for receipt in self.receipt_data:
                if receipt.get('receipt_category'):
                    receipt_categories.add(receipt['receipt_category'])
            return list(receipt_categories) if receipt_categories else ["Others"]

    async def generate_answer(self, user_query: str, context: str) -> str:
        """
        Generate final answer using context and user query
        """
        prompt = f"""You are a concise expense analysis assistant. Provide crisp, structured answers about spending patterns.

Receipt Data:
{context}

User Question: {user_query}

Instructions:
- Be brief and direct
- Use bullet points or structured format
- Show totals clearly (e.g., "Transportation: ₹2000.0")
- For currency differences, note "Cannot combine (different currencies)"
- Include key details but avoid lengthy explanations
- Use emojis sparingly (💰 for totals, 📊 for summaries)

Format examples:
- Single category: "Transportation: ₹2000.0 (Fuel - July 3, 2025)"
- Multiple categories: "Transportation: ₹2000.0 | Groceries: $25.97"
- Total with same currency: "Total: $50.00"
- Total with different currencies: "Total: Cannot combine - ₹2000.0 + $25.97 (different currencies)"

Be concise and helpful."""

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
            query_lower = query.lower()

            # If the user is requesting a list of receipts, handle locally
            if "receipt" in query_lower and any(k in query_lower for k in ["all", "list", "show"]):
                summary = self.list_receipts()
                return {
                    "response": summary,
                    "categories_analyzed": [r.get("receipt_category", "Unknown") for r in self.receipt_data],
                    "receipts_count": len(self.receipt_data),
                    "timestamp": datetime.now().isoformat(),
                }

            relevant_categories = await self.classify_categories(query)
            context = self._build_context()
            answer = await self.generate_answer(query, context)

            return {
                "response": answer,
                "categories_analyzed": relevant_categories,
                "receipts_count": len(self.receipt_data),
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

@app.post("/reload")
async def reload_receipts():
    """Reload receipt data from Firestore"""
    count = analysis_service.reload_receipt_data()
    return {"message": f"Reloaded {count} receipts", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/categories")
async def get_categories():
    """Get available expense categories"""
    return {"categories": EXPENSE_CATEGORIES}

@app.get("/receipts/count")
async def get_receipts_count():
    """Get current number of receipts"""
    return {"count": len(analysis_service.receipt_data), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
