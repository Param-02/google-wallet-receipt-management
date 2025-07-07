#!/usr/bin/env python3
"""
Interactive Receipt Chatbot Tester
Type your own questions to the chatbot!
"""

import requests
import json
import time
from typing import Dict, Any

class ReceiptChatbotTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token = None

    def login(self, username: str = "admin", password: str = "password") -> bool:
        """Obtain auth token from the API"""
        try:
            response = requests.post(
                f"{self.base_url}/login",
                json={"username": username, "password": password},
            )
            if response.status_code == 200:
                self.token = response.json().get("token")
                return True
            return False
        except Exception:
            return False

    def register(self, username: str, password: str) -> bool:
        """Register a new user"""
        try:
            response = requests.post(
                f"{self.base_url}/register",
                json={"username": username, "password": password},
            )
            return response.status_code == 200
        except Exception:
            return False
        
    def test_connection(self) -> bool:
        """Test if the chatbot API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def ask_question(self, query: str) -> Dict[str, Any]:
        """Ask a question to the chatbot"""
        try:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            response = requests.post(
                f"{self.base_url}/chat",
                json={"query": query},
                headers=headers,
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_receipts_count(self) -> Dict[str, Any]:
        """Get current number of receipts"""
        try:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            response = requests.get(f"{self.base_url}/receipts/count", headers=headers)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def print_response(result: Dict[str, Any]):
    """Pretty print the chatbot response"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("\n" + "="*60)
        print("🤖 CHATBOT RESPONSE")
        print("="*60)
        print(result.get('response', 'No response'))
        print("\n" + "-"*60)
        print(f"📂 Categories analyzed: {result.get('categories_analyzed', [])}")
        print(f"📊 Receipts analyzed: {result.get('receipts_count', 0)}")
        print(f"⏰ Timestamp: {result.get('timestamp', 'N/A')}")
        print("="*60)

def main():
    print("🤖 Interactive Receipt Chatbot")
    print("=" * 60)
    print("Type your questions about your receipts!")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chatbot")
    print("  'help' - Show sample questions")
    print("  'count' - Show receipt count")
    print("=" * 60)
    
    tester = ReceiptChatbotTester()

    # Test connection
    print("\n🔌 Testing connection...")
    if not tester.test_connection():
        print("❌ Chatbot API is not running!")
        print("Please start the chatbot with: python3 gemini.py")
        return

    print("✅ Connected to chatbot API")

    # Ensure user exists then login for auth token
    tester.register("admin", "password")
    if not tester.login():
        print("❌ Login failed - check credentials")
        return

    # Get receipts count
    count_result = tester.get_receipts_count()
    print(f"📊 Current receipts loaded: {count_result.get('count', 0)}")
    
    print("\n🎯 Ready! Type your questions below:")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            question = input("\n🙋 Your question: ").strip()
            
            if not question:
                continue
                
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            elif question.lower() == 'help':
                print("\n💡 Sample questions you can ask:")
                print("  • How much did I spend in total?")
                print("  • What were my transportation expenses?")
                print("  • Show me my grocery receipts")
                print("  • What was my most expensive purchase?")
                print("  • How many receipts do I have?")
                print("  • What categories did I spend on?")
                print("  • Which store did I visit most?")
                print("  • Show me receipts from July 2025")
                continue
                
            elif question.lower() == 'count':
                count_result = tester.get_receipts_count()
                print(f"\n📊 Current receipts: {count_result.get('count', 0)}")
                continue
            
            # Ask the chatbot
            print("\n🤔 Thinking...")
            result = tester.ask_question(question)
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (Ctrl+C pressed)")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 
