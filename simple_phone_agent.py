#!/usr/bin/env python3
"""
Simple Phone Call Agent
Works without ngrok by using Twilio's built-in TwiML
"""

import os
import json
import time
from twilio.rest import Client
import ollama
from agent_config import get_system_instructions

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

class SimplePhoneAgent:
    """Simple phone call agent using Twilio's built-in TwiML"""
    
    def __init__(self):
        """Initialize simple phone agent"""
        self.account_sid = os.getenv("TWILIO_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.to_number = os.getenv("TO_PHONE_NUMBER")
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        
        # Load clients
        self.load_clients()
        
        print(f"📞 Simple Phone Agent initialized")
        print(f"   From: {self.from_number}")
        print(f"   To: {self.to_number}")
    
    def load_clients(self):
        """Load client data"""
        try:
            with open('clients.json', 'r') as f:
                self.clients = json.load(f)
            print(f"✅ Loaded {len(self.clients)} clients")
        except FileNotFoundError:
            print("⚠️  clients.json not found")
            self.clients = []
        except Exception as e:
            print(f"❌ Error loading clients: {e}")
            self.clients = []
    
    def find_client_by_phone(self, phone_number):
        """Find client by phone number"""
        for client in self.clients:
            if client.get('Phone') == phone_number:
                return client
        return None
    
    def make_simple_call(self, to_number=None, message="Hello! This is your AI sales assistant. How can I help you today?"):
        """Make a simple call with pre-recorded message"""
        try:
            target_number = to_number or self.to_number
            
            print(f"📞 Making simple call to: {target_number}")
            print(f"   Message: {message}")
            
            # Create TwiML for the call
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">{message}</Say>
    <Pause length="2"/>
    <Say voice="alice">Thank you for calling. Goodbye!</Say>
    <Hangup/>
</Response>"""
            
            # Make the call
            call = self.client.calls.create(
                to=target_number,
                from_=self.from_number,
                twiml=twiml
            )
            
            print(f"✅ Call initiated: {call.sid}")
            return call.sid
            
        except Exception as e:
            print(f"❌ Error making call: {e}")
            return None
    
    def make_interactive_call(self, to_number=None):
        """Make an interactive call (requires webhook setup)"""
        try:
            target_number = to_number or self.to_number
            
            print(f"📞 Making interactive call to: {target_number}")
            print("   Note: This requires webhook setup with ngrok")
            
            # For interactive calls, you need a webhook URL
            # This is a placeholder - you'll need to set up ngrok first
            webhook_url = "https://your-ngrok-url.ngrok.io/call"
            
            call = self.client.calls.create(
                to=target_number,
                from_=self.from_number,
                url=webhook_url,
                method='POST'
            )
            
            print(f"✅ Interactive call initiated: {call.sid}")
            return call.sid
            
        except Exception as e:
            print(f"❌ Error making interactive call: {e}")
            return None
    
    def list_calls(self):
        """List recent calls"""
        try:
            calls = self.client.calls.list(limit=10)
            print("\n📞 Recent Calls:")
            print("=" * 50)
            
            for call in calls:
                status = call.status
                duration = call.duration
                from_num = call.from_
                to_num = call.to
                
                print(f"   {call.sid}")
                print(f"   Status: {status}")
                print(f"   Duration: {duration}s")
                print(f"   From: {from_num} -> To: {to_num}")
                print(f"   Date: {call.date_created}")
                print("-" * 30)
                
        except Exception as e:
            print(f"❌ Error listing calls: {e}")
    
    def test_voice(self, to_number=None):
        """Test voice with different messages"""
        messages = [
            "Hello! This is a test call from your AI sales assistant.",
            "I'm here to help you with information about our products and services.",
            "How can I assist you today?",
            "Thank you for your time. Have a great day!"
        ]
        
        target_number = to_number or self.to_number
        
        print(f"🎤 Testing voice with {target_number}")
        print("=" * 40)
        
        for i, message in enumerate(messages, 1):
            print(f"\n{i}. Testing: {message}")
            call_sid = self.make_simple_call(target_number, message)
            if call_sid:
                print(f"   ✅ Call {i} initiated")
                time.sleep(5)  # Wait between calls
            else:
                print(f"   ❌ Call {i} failed")
                break

def main():
    """Main function"""
    print("📞 Simple Phone Call Agent")
    print("=" * 40)
    
    # Check credentials
    if not os.getenv("TWILIO_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("❌ Twilio credentials not found")
        print("   Please set TWILIO_SID and TWILIO_AUTH_TOKEN")
        return
    
    # Create agent
    agent = SimplePhoneAgent()
    
    # Show menu
    while True:
        print("\n📞 Simple Phone Call Agent Menu:")
        print("1. Make simple call (pre-recorded message)")
        print("2. Make interactive call (requires webhook)")
        print("3. Test voice with multiple messages")
        print("4. List recent calls")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            phone_number = input("Enter phone number to call (or press Enter for default): ").strip()
            message = input("Enter message to say (or press Enter for default): ").strip()
            
            if not message:
                message = "Hello! This is your AI sales assistant. How can I help you today?"
            
            if phone_number:
                agent.make_simple_call(phone_number, message)
            else:
                agent.make_simple_call(message=message)
        
        elif choice == "2":
            print("\n⚠️  Interactive calls require webhook setup with ngrok")
            print("   Please set up ngrok first, then use phone_call_agent.py")
            phone_number = input("Enter phone number to call (or press Enter for default): ").strip()
            if phone_number:
                agent.make_interactive_call(phone_number)
            else:
                agent.make_interactive_call()
        
        elif choice == "3":
            phone_number = input("Enter phone number to test (or press Enter for default): ").strip()
            if phone_number:
                agent.test_voice(phone_number)
            else:
                agent.test_voice()
        
        elif choice == "4":
            agent.list_calls()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 