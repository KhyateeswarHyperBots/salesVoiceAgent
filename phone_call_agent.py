#!/usr/bin/env python3
"""
Phone Call Integration for Sales Voice Agent
Uses Twilio to make and receive actual phone calls
"""

import os
import json
import time
import threading
from flask import Flask, request, Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
import ollama
from agent_config import DEFAULT_SYSTEM_INSTRUCTIONS

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

class PhoneCallAgent:
    """Phone call integration using Twilio"""
    
    def __init__(self):
        """Initialize phone call agent"""
        self.account_sid = os.getenv("TWILIO_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.to_number = os.getenv("TO_PHONE_NUMBER")
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        
        # Conversation state
        self.conversation_history = []
        self.current_client = None
        
        # Load clients
        self.load_clients()
        
        # Flask app for webhooks
        self.app = Flask(__name__)
        self.setup_routes()
        
        print(f"📞 Phone Call Agent initialized")
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
    
    def setup_routes(self):
        """Setup Flask routes for Twilio webhooks"""
        
        @self.app.route('/call', methods=['POST'])
        def handle_call():
            """Handle incoming call"""
            call_sid = request.form.get('CallSid')
            from_number = request.form.get('From')
            
            print(f"📞 Incoming call from: {from_number}")
            
            # Find client
            self.current_client = self.find_client_by_phone(from_number)
            
            # Create TwiML response
            response = VoiceResponse()
            
            if self.current_client:
                welcome_message = f"Hello {self.current_client.get('Full Name')}! I'm your AI sales assistant. I'm here to help you with information about Hyprbots' innovative automation solutions. How can I assist you today?"
            else:
                welcome_message = "Hello! I'm your AI sales assistant. I'm here to help you with information about Hyprbots' innovative automation solutions. How can I assist you today?"
            
            # Speak welcome message
            response.say(welcome_message, voice='alice')
            
            # Gather user input
            gather = Gather(
                input='speech',
                timeout=10,
                speech_timeout='auto',
                action='/process_speech',
                method='POST'
            )
            gather.say("Please speak after the beep.", voice='alice')
            response.append(gather)
            
            # If no input, end call
            response.say("Thank you for calling. Goodbye!", voice='alice')
            response.hangup()
            
            return Response(str(response), mimetype='text/xml')
        
        @self.app.route('/process_speech', methods=['POST'])
        def process_speech():
            """Process speech input and generate response"""
            speech_result = request.form.get('SpeechResult', '')
            confidence = request.form.get('Confidence', '0')
            
            print(f"🎤 Speech: {speech_result}")
            print(f"   Confidence: {confidence}")
            
            if not speech_result:
                response = VoiceResponse()
                response.say("I didn't catch that. Could you please repeat?", voice='alice')
                gather = Gather(
                    input='speech',
                    timeout=10,
                    speech_timeout='auto',
                    action='/process_speech',
                    method='POST'
                )
                gather.say("Please speak after the beep.", voice='alice')
                response.append(gather)
                response.say("Thank you for calling. Goodbye!", voice='alice')
                response.hangup()
                return Response(str(response), mimetype='text/xml')
            
            # Generate AI response
            ai_response = self.generate_response(speech_result)
            
            # Create TwiML response
            response = VoiceResponse()
            response.say(ai_response, voice='alice')
            
            # Continue conversation
            gather = Gather(
                input='speech',
                timeout=10,
                speech_timeout='auto',
                action='/process_speech',
                method='POST'
            )
            gather.say("Please speak after the beep.", voice='alice')
            response.append(gather)
            
            # End call if no more input
            response.say("Thank you for calling. Goodbye!", voice='alice')
            response.hangup()
            
            return Response(str(response), mimetype='text/xml')
    
    def generate_response(self, user_input):
        """Generate AI response using Ollama"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Create context
            context = ""
            if self.current_client:
                context = f"""
Current Client Information:
- Name: {self.current_client.get('Full Name', 'Unknown')}
- Title: {self.current_client.get('Title', 'Unknown')}
- Company: {self.current_client.get('Company', 'Unknown')}
- Location: {self.current_client.get('City', '')}, {self.current_client.get('State', '')}
"""
            
            # Create prompt
            prompt = f"""{system_instructions}

{context}
User Input: {user_input}

Please provide a concise, helpful response suitable for a phone conversation."""
            
            # Get response from Ollama
            response = ollama.chat(model='llama3.2:3b', messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            ai_response = response['message']['content']
            
            # Add to conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': ai_response
            })
            
            print(f"🤖 AI Response: {ai_response}")
            return ai_response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Could you please try again?"
    
    def make_call(self, to_number=None, client_phone=None):
        """Make an outbound call"""
        try:
            target_number = to_number or client_phone or self.to_number
            
            print(f"📞 Making call to: {target_number}")
            
            # Make the call
            call = self.client.calls.create(
                to=target_number,
                from_=self.from_number,
                url=f"{self.get_webhook_url()}/call",
                method='POST'
            )
            
            print(f"✅ Call initiated: {call.sid}")
            return call.sid
            
        except Exception as e:
            print(f"❌ Error making call: {e}")
            return None
    
    def get_webhook_url(self):
        """Get webhook URL (you'll need to set this up with ngrok or similar)"""
        # For development, you can use ngrok
        # ngrok http 5000
        return "https://your-ngrok-url.ngrok.io"
    
    def start_server(self, port=5000):
        """Start Flask server for webhooks"""
        print(f"🌐 Starting webhook server on port {port}")
        print(f"   Webhook URL: {self.get_webhook_url()}")
        print("   Use ngrok to expose this server: ngrok http 5000")
        
        self.app.run(host='0.0.0.0', port=port, debug=False)
    
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


def main():
    """Main function"""
    print("📞 Phone Call Agent")
    print("=" * 40)
    
    # Check credentials
    if not os.getenv("TWILIO_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("❌ Twilio credentials not found")
        print("   Please set TWILIO_SID and TWILIO_AUTH_TOKEN")
        return
    
    # Create agent
    agent = PhoneCallAgent()
    
    # Show menu
    while True:
        print("\n📞 Phone Call Agent Menu:")
        print("1. Make a call")
        print("2. Start webhook server")
        print("3. List recent calls")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            phone_number = input("Enter phone number to call (or press Enter for default): ").strip()
            if phone_number:
                agent.make_call(to_number=phone_number)
            else:
                agent.make_call()
        
        elif choice == "2":
            print("\n🌐 Starting webhook server...")
            print("   You'll need to expose this server using ngrok:")
            print("   ngrok http 5000")
            print("   Then update the webhook URL in the code")
            agent.start_server()
        
        elif choice == "3":
            agent.list_calls()
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main() 