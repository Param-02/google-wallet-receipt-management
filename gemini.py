"""
Receipt Chatbot Backend - FastAPI Implementation (Gemini + Firebase)
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
import os
import uuid
import hashlib
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
import firebase_admin
from firebase_admin import credentials, firestore
from receipt_pipeline import ReceiptPipeline

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

# Token tracking (user authentication handled via Firestore)
TOKENS: Dict[str, str] = {}
auth_scheme = HTTPBearer()

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class ProcessReceiptRequest(BaseModel):
    image_path: str

class ProcessReceiptResponse(BaseModel):
    receipt: Dict[str, Any]

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
        self.receipt_data_cache: Dict[str, List[Dict[str, Any]]] = {}

    def list_receipts(self, user_id: str) -> str:
        """Return a human readable list of all receipts for the user."""
        data = self._get_receipt_data(user_id)
        if not data:
            return "No receipts found."
        lines = []
        for r in data:
            store = r.get("store_name", "Unknown")
            category = r.get("receipt_category", "Unknown")
            total = r.get("total_amount", 0)
            currency = r.get("currency", "")
            date = r.get("date", "Unknown")
            lines.append(f"* {category}: {currency}{total} ({store} - {date})")

        # Calculate total per currency
        totals = {}
        for r in data:
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

    def _load_receipt_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Load receipt data for a specific user from Firestore."""
        try:
            collection = (
                self.db.collection("users")
                .document(user_id)
                .collection("receipts")
            )
            docs = list(collection.stream())
            data = [doc.to_dict() for doc in docs]
            logger.info(
                f"Loaded {len(data)} receipts from Firestore for user {user_id}."
            )
            return data
        except Exception as e:
            logger.error(f"Failed to load receipts for {user_id}: {e}")
            return []

    def reload_receipt_data(self, user_id: str) -> int:
        """Reload receipt data from Firestore for the given user."""
        data = self._load_receipt_data(user_id)
        self.receipt_data_cache[user_id] = data
        return len(data)

    def _get_receipt_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve cached receipt data for a user, loading if necessary."""
        if user_id not in self.receipt_data_cache:
            self.receipt_data_cache[user_id] = self._load_receipt_data(user_id)
        return self.receipt_data_cache[user_id]

    def _build_context(self, user_id: str) -> str:
        """Builds context string from receipt data"""
        data = self._get_receipt_data(user_id)
        if not data:
            return "No receipts found."

        summaries = []
        for idx, receipt in enumerate(data, start=1):
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

    async def classify_categories(self, user_query: str, user_id: str) -> List[str]:
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
                for receipt in self._get_receipt_data(user_id):
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
            for receipt in self._get_receipt_data(user_id):
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

    async def process_chat_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Main method to process chat query through the full pipeline
        """
        try:
            query_lower = query.lower()

            # If the user is requesting a list of receipts, handle locally
            if "receipt" in query_lower and any(k in query_lower for k in ["all", "list", "show"]):
                summary = self.list_receipts(user_id)
                return {
                    "response": summary,
                    "categories_analyzed": [
                        r.get("receipt_category", "Unknown")
                        for r in self._get_receipt_data(user_id)
                    ],
                    "receipts_count": len(self._get_receipt_data(user_id)),
                    "timestamp": datetime.now().isoformat(),
                }

            relevant_categories = await self.classify_categories(query, user_id)
            context = self._build_context(user_id)
            answer = await self.generate_answer(query, context)

            return {
                "response": answer,
                "categories_analyzed": relevant_categories,
                "receipts_count": len(self._get_receipt_data(user_id)),
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


def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Return a salted SHA-256 hash and the salt."""
    if salt is None:
        salt = uuid.uuid4().hex
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000,
    ).hex()
    return hashed, salt


def _user_doc(username: str):
    """Return Firestore document reference for a user."""
    return analysis_service.db.collection("users").document(username)


@app.post("/register")
async def register(request: RegisterRequest):
    """Register a new user in Firestore."""
    doc_ref = _user_doc(request.username)
    doc = doc_ref.get()
    if doc.exists and doc.to_dict().get("password"):
        raise HTTPException(status_code=400, detail="User already exists")
    try:
        pwd_hash, salt = _hash_password(request.password)
        doc_ref.set({
            "password": pwd_hash,
            "salt": salt,
            "created_at": datetime.now().isoformat(),
        })
        return {"message": "User registered"}
    except Exception as e:
        logger.error(f"Failed to register user {request.username}: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user via Firestore and return token."""
    doc = _user_doc(request.username).get()
    if not doc.exists:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    data = doc.to_dict()
    stored_hash = data.get("password")
    salt = data.get("salt")
    if not stored_hash or not salt:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    pwd_hash, _ = _hash_password(request.password, salt)
    if stored_hash != pwd_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    TOKENS[token] = request.username
    return LoginResponse(token=token)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """Main chat endpoint for receipt analysis queries."""
    token = credentials.credentials
    if token not in TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    user_id = TOKENS[token]

    logger.info(
        f"Processing chat query for {user_id}: {request.query}")
    result = await analysis_service.process_chat_query(request.query, user_id)
    return ChatResponse(**result)


@app.post("/process_receipt", response_model=ProcessReceiptResponse)
async def process_receipt_endpoint(
    request: ProcessReceiptRequest,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """Process a receipt image for the authenticated user."""
    token = credentials.credentials
    if token not in TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = TOKENS[token]
    pipeline = ReceiptPipeline(user_id=user_id)
    result = pipeline.process_receipt(request.image_path)
    if not result:
        raise HTTPException(status_code=500, detail="Receipt processing failed")
    analysis_service.reload_receipt_data(user_id)
    return ProcessReceiptResponse(receipt=result)

@app.post("/reload")
async def reload_receipts(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """Reload receipt data from Firestore for the authenticated user"""
    token = credentials.credentials
    if token not in TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = TOKENS[token]
    count = analysis_service.reload_receipt_data(user_id)
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
async def get_receipts_count(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Get current number of receipts for the authenticated user"""
    token = credentials.credentials
    if token not in TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = TOKENS[token]
    count = len(analysis_service._get_receipt_data(user_id))
    return {"count": count, "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
