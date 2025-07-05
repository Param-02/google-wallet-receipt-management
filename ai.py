#!/usr/bin/env python3
"""
Receipt Parser using Google Cloud Vertex AI Gemini
Extracts structured information from receipt PDFs using multimodal AI.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Vertex AI configuration
PROJECT_ID = "splendid-yeti-464913-j2"  # Updated to match service account
LOCATION = "us-central1"  # Replace with your preferred location
MODEL_NAME = "gemini-2.5-flash"  # or "gemini-1.5-pro"

# Authentication methods (choose one):
# Method 1: Use environment variable (recommended)
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Method 2: Use gcloud CLI authentication (easiest for development)
# Run: gcloud auth application-default login

# Method 3: Set service account key path directly (less secure)
# SERVICE_ACCOUNT_KEY_PATH = "/path/to/your/service-account-key.json"

class ReceiptParser:
    """Receipt parser using Vertex AI Gemini multimodal model."""
    
    def __init__(self, project_id: str, location: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the receipt parser.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location (e.g., 'us-central1')
            model_name: Gemini model name
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI with authentication
        try:
            # Method 3: If using direct service account key path
            # import os
            # if 'SERVICE_ACCOUNT_KEY_PATH' in globals():
            #     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_KEY_PATH
            
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(model_name)
            logger.info(f"Successfully initialized Vertex AI with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            raise
        
        # Receipt parsing prompt
        self.prompt = """You are an expert receipt parser. Your task is to extract structured information from a receipt image and output it as a valid minified JSON. You must translate any non-English text into English. You will be provided with an image of a bill or receipt.

Extract as much structured information as possible from the receipt image and output it as a valid minified JSON object, following the schema below. Use ISO-8601 date format (YYYY-MM-DD), 24-hour time (HH:MM), and include the detected currency symbol. Translate any non-English text into English. Leave any missing fields as null. Also include a summary under 100-200 tokens for the receipt.

For the `receipt_category` field, categorize the entire receipt based on the primary type of business/transaction. Choose from:
- Groceries (grocery stores, supermarkets, food markets)
- Food (restaurants, cafes, food delivery, dining)
- Transportation (gas stations, fuel, parking, transport services)
- Travel (hotels, airlines, car rentals, travel bookings)
- Utilities (electricity, water, gas, internet, phone bills)
- Subscriptions (streaming services, memberships, recurring services)
- Healthcare (hospitals, clinics, pharmacy, medical services)
- Shopping (retail stores, clothing, electronics, general merchandise)
- Entertainment (movies, concerts, events, recreation)
- Education (schools, courses, books, educational services)
- Maintenance (repairs, services, home improvement)
- Financial (banking, insurance, investments, financial services)
- Others (anything that doesn't fit the above categories)

For each `item` in the `items` list, the `category` field **must** be one of the same categories above.

If the appropriate category is unclear or does not fit, use `"Others"`.

Here is the schema:
{
    "store_name": "",
    "store_address": "",
    "store_phone": "",
    "date": "",
    "time": "",
    "bill_number": "",
    "receipt_category": "",
    "Summary": "",
    "payment_method": "",
    "currency": "",
    "subtotal_amount": "",
    "tax_amount": "",
    "tip_amount": "",
    "total_amount": "",
    "items": [
        {
            "item_name": "",
            "quantity": "",
            "unit_price": "",
            "total_price": "",
            "category": ""
        }
    ],
    "tax_breakdown": [
        {
            "tax_name": "",
            "tax_rate": "",
            "tax_amount": ""
        }
    ],
    "footer_notes": ""
}

Output only the JSON. Ensure the output is a valid minified JSON."""


    def load_pdf_bytes(self, pdf_path: str) -> Optional[bytes]:
        """
        Load PDF file as bytes.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDF bytes or None if loading fails
        """
        try:
            logger.info(f"Loading PDF file: {pdf_path}")
            
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            logger.info(f"Successfully loaded PDF ({len(pdf_bytes)} bytes)")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Error loading PDF file: {str(e)}")
            return None

    def parse_receipt(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse receipt from PDF using Gemini multimodal model.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Parsed receipt data as dictionary or None if parsing fails
        """
        try:
            # Load PDF as bytes
            pdf_bytes = self.load_pdf_bytes(pdf_path)
            if not pdf_bytes:
                return None
                
            logger.info("Sending PDF to Gemini for parsing...")
            
            # Create PDF part for Gemini
            pdf_part = Part.from_data(pdf_bytes, mime_type="application/pdf")
            
            # Generate content with Gemini
            response = self.model.generate_content([self.prompt, pdf_part])
            
            if not response.text:
                logger.error("No response from Gemini")
                return None
                
            logger.info("Received response from Gemini")
            
            # Parse JSON response
            try:
                # Clean the response text (remove any markdown formatting)
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                parsed_data = json.loads(response_text)
                logger.info("Successfully parsed JSON response")
                return parsed_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Response text: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing receipt: {str(e)}")
            return None

    def save_result(self, data: Dict[str, Any], output_path: str = "parsed_receipt.json"):
        """
        Save parsed receipt data to JSON file.
        
        Args:
            data: Parsed receipt data
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
            
            logger.info(f"Saved parsed receipt to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")

def main():
    """Main function to run the receipt parser."""
    
    if len(sys.argv) != 2:
        print("Usage: python parse_receipt_with_vertexai.py <path_to_receipt.pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Validate input file
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"Input file must be a PDF: {pdf_path}")
        sys.exit(1)
    
    # Check if project ID is set
    if PROJECT_ID == "your-project-id":
        logger.error("Please set your Google Cloud PROJECT_ID in the script")
        print("\n📋 Setup Instructions:")
        print("1. Replace 'your-project-id' with your actual Google Cloud project ID")
        print("2. Ensure you have Vertex AI API enabled in your project")
        print("3. Choose one authentication method:")
        print("   Method 1 (Recommended): Service Account Key")
        print("   - Create a service account in Google Cloud Console")
        print("   - Download the JSON key file")
        print("   - Set environment variable: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
        print("   Method 2 (Development): gcloud CLI")
        print("   - Run: gcloud auth application-default login")
        print("   - Follow the browser authentication flow")
        print("   Method 3 (Direct): Set SERVICE_ACCOUNT_KEY_PATH in script")
        print("   - Uncomment and set the SERVICE_ACCOUNT_KEY_PATH variable")
        sys.exit(1)
    
    try:
        # Initialize parser
        logger.info("Initializing receipt parser...")
        parser = ReceiptParser(PROJECT_ID, LOCATION, MODEL_NAME)
        
        # Parse receipt
        logger.info(f"Processing receipt: {pdf_path}")
        result = parser.parse_receipt(pdf_path)
        
        if result:
            # Save result
            parser.save_result(result)
            print("✅ Receipt parsed successfully!")
            print(f"✅ Output saved to: parsed_receipt.json")
            
            # Print summary
            if 'Summary' in result and result['Summary']:
                print(f"\n📋 Summary: {result['Summary']}")
            
            if 'total_amount' in result and result['total_amount']:
                currency = result.get('currency', '')
                print(f"💰 Total: {currency}{result['total_amount']}")
                
        else:
            logger.error("Failed to parse receipt")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()