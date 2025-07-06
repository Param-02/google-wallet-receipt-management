# 🧾 Receipt Processing System

An intelligent receipt processing system that extracts structured data from receipt images using computer vision and AI, with a chatbot interface for natural language queries.

## 🚀 Features

- **📸 Image Processing**: Converts receipt images to clean PDFs
- **🤖 AI-Powered Extraction**: Uses Google Vertex AI Gemini to extract structured data
- **📂 Smart Categorization**: Categorizes both receipts and individual items
- **💾 Data Accumulation**: Stores all receipts in Firebase Firestore
- **🤖 Chatbot Interface**: Ask questions about your receipts in natural language
- **🌐 REST API**: FastAPI-based backend for integration
- **🛠️ Pipeline Orchestration**: Seamless workflow management

## 📋 System Architecture

```
📸 Receipt Image → 🔄 Image Processing → 🤖 AI Parsing → 🔥 Firestore Storage → 💬 Chatbot
     (main2.py)      (receipt_pipeline.py)   (ai.py)     (Firestore)  (gemini.py)
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Google Cloud account with Vertex AI enabled
- Service account key with appropriate permissions

### Setup
1. **Clone/Download the project**
   ```bash
   cd /path/to/bill_recepit
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Google Cloud Setup**
   - Place your service account key file in the project directory
   - The system is configured for: `splendid-yeti-464913-j2-83b565740502.json`
   - Environment variable is automatically set via `.zshrc`

## 🎯 Usage

### 1. Process Receipt Images

**Using the convenience script:**
```bash
# Process a single receipt
./process_receipt.sh 33.jpg

# Process with custom output
./process_receipt.sh receipt.jpg --output my_receipts.json

# Show all stored receipts
./process_receipt.sh --show-all
```

**Using Python directly:**
```bash
# Process receipt
python3 receipt_pipeline.py --input 33.jpg

# View all receipts
python3 receipt_pipeline.py --show-all
```

### 2. Start the Chatbot

**Using the convenience script:**
```bash
./process_receipt.sh --chatbot
```

**Using Python directly:**
```bash
python3 gemini.py
```

### 3. Interactive Chatbot Testing

```bash
python3 test_chatbot.py
```

Then ask questions like:
- "How much did I spend in total?"
- "What were my transportation expenses?"
- "Show me my grocery receipts"
- "What was my most expensive purchase?"

### 4. API Endpoints

With the chatbot running on `http://localhost:8000`:

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How much did I spend on groceries?"}'

# Get receipts count
curl http://localhost:8000/receipts/count

# Reload receipt data
curl -X POST http://localhost:8000/reload
```

## 📁 File Structure

### Core Components
```
📂 bill_recepit/
├── 🔧 main2.py                    # Image processing (image → PDF)
├── 🤖 ai.py                       # AI parsing (PDF → structured JSON)
├── 🔄 receipt_pipeline.py         # Workflow orchestration
├── 💬 gemini.py                   # Chatbot API server
├── 🧪 test_chatbot.py             # Interactive testing interface
├── 🚀 process_receipt.sh          # Convenience script
├── 📋 requirements.txt            # Dependencies
├── 🔑 splendid-yeti-464913-j2...json  # Google Cloud credentials
└── 🔥 Firestore collection      # Accumulated receipt data
```

### Sample Data
```
├── 📸 33.JPG                      # Fuel receipt sample
├── 📸 22.JPG                      # Additional sample
├── 📸 11.JPG                      # Additional sample
├── 📸 samplebill.jpg              # Grocery receipt sample
└── 📸 receipt.jpg                 # Additional sample
```

## 🏗️ System Components

### 1. Image Processing (`main2.py`)
- **Input**: Receipt images (JPG, PNG, etc.)
- **Processing**: Contour detection, perspective correction, noise reduction
- **Output**: Clean, optimized PDF files
- **Features**: Multiple fallback methods, size optimization

### 2. AI Parsing (`ai.py`)
- **Input**: PDF files
- **AI Model**: Google Vertex AI Gemini 2.5-flash
- **Output**: Structured JSON with categorization
- **Features**: Multilingual support, currency detection, smart categorization

### 3. Pipeline Orchestration (`receipt_pipeline.py`)
- **Function**: Coordinates image processing and AI parsing
- **Features**: Error handling, progress tracking, data accumulation
- **Output**: Consolidated receipt database

### 4. Chatbot Interface (`gemini.py`)
- **API**: FastAPI-based REST service
- **AI**: Natural language understanding for receipt queries
- **Features**: Category classification, contextual responses
- **Endpoints**: `/chat`, `/health`, `/reload`, `/receipts/count`

## 📊 Data Format

### Receipt JSON Structure
```json
{
  "store_name": "The Shop",
  "store_address": "Store #100 Chicago, IL",
  "date": "2025-07-03",
  "time": "12:57",
  "receipt_category": "Groceries",
  "total_amount": "25.97",
  "currency": "$",
  "items": [
    {
      "item_name": "Large Eggs",
      "quantity": "1",
      "unit_price": "0.99",
      "total_price": "0.99",
      "category": "Groceries"
    }
  ],
  "processed_at": "2025-07-04T21:23:48.198979",
  "source_image": "samplebill.jpg"
}
```

### Available Categories
- **Groceries** - Supermarkets, food markets
- **Food** - Restaurants, cafes, food delivery
- **Transportation** - Gas stations, fuel, parking
- **Travel** - Hotels, airlines, car rentals
- **Utilities** - Bills, internet, phone
- **Shopping** - Retail, electronics, clothing
- **Healthcare** - Medical services, pharmacy
- **Entertainment** - Movies, events, recreation
- **Education** - Schools, courses, books
- **Maintenance** - Repairs, services
- **Financial** - Banking, insurance
- **Others** - Miscellaneous expenses

## 🔧 Configuration

### Google Cloud Setup
1. Enable Vertex AI API in your Google Cloud project
2. Create a service account with appropriate permissions
3. Download the service account key
4. Update `PROJECT_ID` in `ai.py` and `gemini.py` if needed

### Environment Variables
The system automatically sets:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

## 🧪 Testing

### Sample Questions for Chatbot
- "How much did I spend in total?"
- "What were my transportation expenses?"
- "Show me my grocery receipts"
- "What was my most expensive purchase?"
- "How many receipts do I have?"
- "What categories did I spend on?"
- "Compare my spending between groceries and transportation"

### Test Commands
```bash
# Test image processing
python3 main2.py --input 33.jpg --output-pdf test.pdf

# Test AI parsing
python3 ai.py test.pdf

# Test full pipeline
python3 receipt_pipeline.py --input samplebill.jpg

# Test chatbot interactively
python3 test_chatbot.py
```

## 🚨 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'fastapi'"**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

**2. "Your default credentials were not found"**
```bash
# Check if environment variable is set
echo $GOOGLE_APPLICATION_CREDENTIALS
# If not set, restart terminal or run:
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/splendid-yeti-464913-j2-83b565740502.json"
```

**3. "403 Permission denied" for Vertex AI**
- Ensure Vertex AI API is enabled in Google Cloud Console
- Check service account permissions
- Verify project ID in configuration

**4. Chatbot not responding**
```bash
# Check if server is running
curl http://localhost:8000/health
# If not running, start with:
python3 gemini.py
```

## 🔄 Workflow Examples

### Processing Multiple Receipts
```bash
# Process several receipts
./process_receipt.sh 33.jpg
./process_receipt.sh samplebill.jpg
./process_receipt.sh receipt.jpg

# View accumulated data
./process_receipt.sh --show-all

# Start chatbot to analyze
./process_receipt.sh --chatbot
```

### API Integration
```python
import requests

# Ask the chatbot
response = requests.post("http://localhost:8000/chat", 
    json={"query": "How much did I spend on transportation?"})
result = response.json()
print(result['response'])
```

## 🤝 Contributing

1. Follow the existing code structure
2. Add error handling for new features
3. Update this README for significant changes
4. Test with various receipt formats

## 📝 License

This project is for personal/educational use. Ensure compliance with Google Cloud terms when using Vertex AI services.

---

**🎉 Happy Receipt Processing!** 

For questions or issues, check the troubleshooting section or review the logs for detailed error information. 