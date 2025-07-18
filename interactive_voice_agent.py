#!/usr/bin/env python3
"""
Interactive Voice Agent with Ollama
Full conversational AI phone agent using Twilio + Ollama
"""

import os
import json
import time
import threading
from flask import Flask, request, Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
import ollama
from agent_config import get_system_instructions

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

class InteractiveVoiceAgent:
    """Interactive voice agent using Twilio + Ollama"""
    
    def __init__(self):
        """Initialize interactive voice agent"""
        self.account_sid = os.getenv("TWILIO_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.to_number = os.getenv("TO_PHONE_NUMBER")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        
        # Conversation state
        self.conversation_history = {}
        self.current_client = None
        
        # Load clients
        self.load_clients()
        
        # Flask app for webhooks
        self.app = Flask(__name__)
        self.setup_routes()
        
        print(f"🎤 Interactive Voice Agent initialized")
        print(f"   From: {self.from_number}")
        print(f"   To: {self.to_number}")
        print(f"   Model: {self.model_name}")
    
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
            print(f"   Call SID: {call_sid}")
            
            # Find client
            self.current_client = self.find_client_by_phone(from_number)
            
            # Initialize conversation history for this call
            self.conversation_history[call_sid] = []
            
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
            """Process speech input and generate AI response"""
            call_sid = request.form.get('CallSid')
            speech_result = request.form.get('SpeechResult', '')
            confidence = request.form.get('Confidence', '0')
            
            print(f"🎤 Speech from {call_sid}: {speech_result}")
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
            
            # Check for end call keywords
            if any(word in speech_result.lower() for word in ['goodbye', 'bye', 'end call', 'hang up', 'stop']):
                response = VoiceResponse()
                response.say("Thank you for calling. Have a great day!", voice='alice')
                response.hangup()
                return Response(str(response), mimetype='text/xml')
            
            # Generate AI response using Ollama
            ai_response = self.generate_ai_response(speech_result, call_sid)
            
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
    
    def generate_ai_response(self, user_input, call_sid):
        """Generate AI response using Ollama"""
        try:
            # Add to conversation history
            if call_sid not in self.conversation_history:
                self.conversation_history[call_sid] = []
            
            self.conversation_history[call_sid].append({
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
            
            # Get conversation history for context
            conversation_context = ""
            if len(self.conversation_history[call_sid]) > 1:
                recent_exchanges = self.conversation_history[call_sid][-6:]  # Last 3 exchanges
                conversation_context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_exchanges[:-1]  # Exclude current user input
                ])
            
            # Create prompt
            system_instructions = get_system_instructions()
            prompt = f"""{system_instructions}

{context}
Previous conversation:
{conversation_context}

Current user input: {user_input}

Please provide a concise, helpful response suitable for a phone conversation. Keep it under 2-3 sentences."""
            
            print(f"🤖 Generating AI response...")
            
            # Get response from Ollama
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            ai_response = response['message']['content']
            
            # Add to conversation history
            self.conversation_history[call_sid].append({
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
            
            print(f"📞 Making interactive call to: {target_number}")
            print("   Note: This requires webhook server to be running")
            
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
        print(f"🌐 Starting interactive voice agent server on port {port}")
        print(f"   Webhook URL: {self.get_webhook_url()}")
        print("   Use ngrok to expose this server: ngrok http 5000")
        print("   Then update the webhook URL in get_webhook_url() method")
        
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
    
    def save_conversation(self, call_sid):
        """Save conversation to file"""
        if call_sid not in self.conversation_history:
            return
        
        try:
            os.makedirs('phone_conversations', exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'phone_conversations/conversation_{call_sid}_{timestamp}.json'
            
            conversation_data = {
                'call_sid': call_sid,
                'timestamp': timestamp,
                'client': self.current_client.get('Full Name') if self.current_client else 'Unknown',
                'conversation': self.conversation_history[call_sid]
            }
            
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            print(f"💾 Conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")


def main():
    """Main function"""
    print("🎤 Interactive Voice Agent with Ollama")
    print("=" * 50)
    
    # Check credentials
    if not os.getenv("TWILIO_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("❌ Twilio credentials not found")
        print("   Please set TWILIO_SID and TWILIO_AUTH_TOKEN")
        return
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        print(f"✅ Ollama connected with {len(models)} models")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   Please start Ollama: ollama serve")
        return
    
    # Create agent
    agent = InteractiveVoiceAgent()
    
    # Show menu
    while True:
        print("\n🎤 Interactive Voice Agent Menu:")
        print("1. Start webhook server (for incoming calls)")
        print("2. Make outbound call")
        print("3. List recent calls")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🌐 Starting webhook server...")
            print("   You'll need to expose this server using ngrok:")
            print("   ngrok http 5000")
            print("   Then update the webhook URL in the code")
            agent.start_server()
        
        elif choice == "2":
            phone_number = input("Enter phone number to call (or press Enter for default): ").strip()
            if phone_number:
                agent.make_call(to_number=phone_number)
            else:
                agent.make_call()
        
        elif choice == "3":
            agent.list_calls()
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main() 