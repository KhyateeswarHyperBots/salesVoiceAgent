#!/usr/bin/env python3
"""
Enhanced Interactive Voice Agent with Ollama
Full conversational AI phone agent using Twilio + Ollama + RAG + Sentiment Analysis
"""

import os
import json
import time
import threading
import queue
import numpy as np
import datetime
import warnings
from flask import Flask, request, Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
import ollama
from agent_config import get_system_instructions
import faiss
from sentence_transformers import SentenceTransformer
import textblob
from textblob import TextBlob
import librosa
import soundfile as sf
from scipy import signal

# Suppress warnings
warnings.filterwarnings('ignore')

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

# Set environment variables to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), 'models')

class RAG:
    """RAG system for document retrieval"""
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []
        self.index_path = "faiss_index"
        self.embeddings_path = "embeddings.pkl"

    def build_index(self, docs):
        """Build FAISS index from documents"""
        self.documents = docs
        embeddings = self.model.encode(docs)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        with open(self.embeddings_path, "wb") as f:
            import pickle
            pickle.dump(docs, f)

    def load_index(self):
        """Load existing FAISS index"""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.embeddings_path, "rb") as f:
                import pickle
                self.documents = pickle.load(f)
            print(f"✅ Loaded RAG index with {len(self.documents)} documents")
        except Exception as e:
            print(f"❌ Error loading RAG index: {e}")
            print("🔄 Rebuilding index...")
            # If loading fails, rebuild the index
            try:
                with open('documents.json') as f:
                    docs = json.load(f)
                self.build_index(docs)
            except Exception as e2:
                print(f"❌ Error rebuilding index: {e2}")

    def retrieve(self, query, top_k=3):
        """Retrieve relevant documents"""
        if self.index is None:
            print("⚠️ RAG index not loaded, attempting to load...")
            self.load_index()
            if self.index is None:
                return ["No documents available for retrieval."]
        
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in I[0]]

class ClientRAGSystem:
    """Client-specific RAG system"""
    def __init__(self):
        self.clients = []
        self.load_clients()

    def load_clients(self):
        """Load client data"""
        try:
            with open('clients.json', 'r') as f:
                self.clients = json.load(f)
            print(f"✅ Loaded {len(self.clients)} clients")
        except FileNotFoundError:
            print("⚠️ clients.json not found")
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

    def get_client_context(self, client):
        """Get formatted client context"""
        if not client:
            return ""
        
        return f"""
Client Information:
- Name: {client.get('Full Name', 'Unknown')}
- Title: {client.get('Title', 'Unknown')}
- Company: {client.get('Company', 'Unknown')}
- Location: {client.get('City', '')}, {client.get('State', '')}
- Industry: {client.get('Industry', 'Unknown')}
- Phone: {client.get('Phone', 'Unknown')}
"""

class EnhancedInteractiveVoiceAgent:
    """Enhanced interactive voice agent with RAG, sentiment analysis, and client data"""
    
    def __init__(self):
        """Initialize enhanced interactive voice agent"""
        self.account_sid = os.getenv("TWILIO_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.to_number = os.getenv("TO_PHONE_NUMBER")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        
        # Initialize RAG systems
        self.rag = RAG()
        self.client_rag = ClientRAGSystem()
        
        # Conversation state
        self.conversation_history = {}
        self.current_client = None
        
        # Sentiment analysis tracking
        self._sentiment_history = {}
        self._buying_signals = {}
        self._objection_signals = {}
        self._engagement_score = {}
        self._buying_probability = {}
        
        # Audio sentiment analysis
        self._audio_sentiment_history = {}
        self._voice_emotion_scores = {}
        self._speech_patterns = {}
        self._real_time_audio_buffer = queue.Queue()
        self._audio_analysis_active = False
        
        # System instructions
        self.system_instructions = get_system_instructions()
        
        # Flask app for webhooks
        self.app = Flask(__name__)
        self.setup_routes()
        
        print(f"🎤 Enhanced Interactive Voice Agent initialized")
        print(f"   From: {self.from_number}")
        print(f"   To: {self.to_number}")
        print(f"   Model: {self.model_name}")
    
    def analyze_sentiment(self, text):
        """Analyze text sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            
            # Classify sentiment
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                'sentiment': sentiment,
                'polarity': sentiment_score,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'sentiment': 'neutral',
                'polarity': 0.0,
                'subjectivity': 0.0
            }
    
    def detect_buying_signals(self, text):
        """Detect buying signals in text"""
        buying_keywords = [
            'interested', 'pricing', 'cost', 'price', 'quote', 'proposal',
            'demo', 'trial', 'purchase', 'buy', 'implement', 'deploy',
            'solution', 'benefits', 'roi', 'investment', 'budget',
            'timeline', 'schedule', 'next steps', 'decision', 'approval'
        ]
        
        objection_keywords = [
            'expensive', 'costly', 'budget', 'expensive', 'not sure',
            'concerned', 'worried', 'risk', 'security', 'integration',
            'complex', 'difficult', 'time', 'resources', 'staff',
            'training', 'support', 'maintenance'
        ]
        
        text_lower = text.lower()
        buying_signals = [word for word in buying_keywords if word in text_lower]
        objections = [word for word in objection_keywords if word in text_lower]
        
        return {
            'buying_signals': buying_signals,
            'objections': objections,
            'buying_score': len(buying_signals),
            'objection_score': len(objections)
        }
    
    def update_buying_probability(self, user_input, sentiment_data, buying_signals, call_sid):
        """Update buying probability based on conversation"""
        if call_sid not in self._buying_probability:
            self._buying_probability[call_sid] = 0.0
        
        current_prob = self._buying_probability[call_sid]
        
        # Sentiment impact
        sentiment_impact = sentiment_data['polarity'] * 10
        
        # Buying signals impact
        buying_impact = buying_signals['buying_score'] * 5
        
        # Objection impact (negative)
        objection_impact = -buying_signals['objection_score'] * 3
        
        # Update probability
        new_prob = current_prob + sentiment_impact + buying_impact + objection_impact
        
        # Clamp between 0 and 100
        self._buying_probability[call_sid] = max(0, min(100, new_prob))
        
        print(f"💰 Buying probability: {self._buying_probability[call_sid]:.1f}%")
    
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
            self.current_client = self.client_rag.find_client_by_phone(from_number)
            
            # Initialize conversation history for this call
            self.conversation_history[call_sid] = []
            self._sentiment_history[call_sid] = []
            self._buying_signals[call_sid] = []
            self._objection_signals[call_sid] = []
            self._engagement_score[call_sid] = 0.0
            self._buying_probability[call_sid] = 0.0
            
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
                # Save conversation before ending
                self.save_conversation(call_sid)
                
                response = VoiceResponse()
                response.say("Thank you for calling. Have a great day!", voice='alice')
                response.hangup()
                return Response(str(response), mimetype='text/xml')
            
            # Analyze sentiment and buying signals
            sentiment_data = self.analyze_sentiment(speech_result)
            buying_signals = self.detect_buying_signals(speech_result)
            
            # Update buying probability
            self.update_buying_probability(speech_result, sentiment_data, buying_signals, call_sid)
            
            # Generate AI response using Ollama with RAG
            ai_response = self.generate_enhanced_ai_response(speech_result, call_sid)
            
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
    
    def generate_enhanced_ai_response(self, user_input, call_sid):
        """Generate AI response using Ollama with RAG and context"""
        try:
            # Add to conversation history
            if call_sid not in self.conversation_history:
                self.conversation_history[call_sid] = []
            
            self.conversation_history[call_sid].append({
                'role': 'user',
                'content': user_input
            })
            
            # Get RAG context
            rag_context = ""
            try:
                relevant_docs = self.rag.retrieve(user_input, top_k=2)
                rag_context = "\n".join(relevant_docs)
            except Exception as e:
                print(f"RAG retrieval error: {e}")
            
            # Get client context
            client_context = ""
            if self.current_client:
                client_context = self.client_rag.get_client_context(self.current_client)
            
            # Get conversation history for context
            conversation_context = ""
            if len(self.conversation_history[call_sid]) > 1:
                recent_exchanges = self.conversation_history[call_sid][-6:]  # Last 3 exchanges
                conversation_context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_exchanges[:-1]  # Exclude current user input
                ])
            
            # Get buying probability context
            buying_context = ""
            if call_sid in self._buying_probability:
                prob = self._buying_probability[call_sid]
                if prob > 70:
                    buying_context = "The customer shows strong buying interest. Focus on next steps and closing."
                elif prob > 40:
                    buying_context = "The customer shows moderate interest. Continue building value and addressing concerns."
                else:
                    buying_context = "The customer shows low interest. Focus on understanding needs and building rapport."
            
            # Create enhanced prompt
            prompt = f"""{self.system_instructions}

{client_context}

Relevant Information from Knowledge Base:
{rag_context}

Previous conversation:
{conversation_context}

Buying Context: {buying_context}

Current user input: {user_input}

Please provide a helpful, conversational response that:
1. Addresses the user's question or concern
2. Uses relevant information from the knowledge base
3. Maintains a professional but friendly tone
4. Keeps the response concise (2-3 sentences) for phone conversation
5. Adapts to the buying probability context provided"""
            
            print(f"🤖 Generating enhanced AI response...")
            
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
            
            print(f"📞 Making enhanced interactive call to: {target_number}")
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
        # You can also use localtunnel: npx localtunnel --port 5000
        return "https://your-ngrok-url.ngrok.io"
    
    def start_server(self, port=5000):
        """Start Flask server for webhooks"""
        print(f"🌐 Starting enhanced interactive voice agent server on port {port}")
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
        """Save conversation to file with enhanced data"""
        if call_sid not in self.conversation_history:
            return
        
        try:
            os.makedirs('phone_conversations', exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'phone_conversations/enhanced_conversation_{call_sid}_{timestamp}.json'
            
            conversation_data = {
                'call_sid': call_sid,
                'timestamp': timestamp,
                'client': self.current_client.get('Full Name') if self.current_client else 'Unknown',
                'conversation': self.conversation_history[call_sid],
                'sentiment_history': self._sentiment_history.get(call_sid, []),
                'buying_signals': self._buying_signals.get(call_sid, []),
                'objection_signals': self._objection_signals.get(call_sid, []),
                'final_buying_probability': self._buying_probability.get(call_sid, 0.0),
                'engagement_score': self._engagement_score.get(call_sid, 0.0)
            }
            
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            print(f"💾 Enhanced conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")


def main():
    """Main function"""
    print("🎤 Enhanced Interactive Voice Agent with Ollama + RAG + Sentiment Analysis")
    print("=" * 70)
    
    # Check credentials
    if not os.getenv("TWILIO_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("❌ Twilio credentials not found")
        print("   Please set TWILIO_SID and TWILIO_AUTH_TOKEN")
        return
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        model_count = len(list(models))  # Convert to list to get length
        print(f"✅ Ollama connected with {model_count} models")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   Please start Ollama: ollama serve")
        return
    
    # Create agent
    agent = EnhancedInteractiveVoiceAgent()
    
    # Show menu
    while True:
        print("\n🎤 Enhanced Interactive Voice Agent Menu:")
        print("1. Start webhook server (for incoming calls)")
        print("2. Make outbound call")
        print("3. List recent calls")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🌐 Starting enhanced webhook server...")
            print("   Features: RAG, Sentiment Analysis, Client Data, Buying Signals")
            print("   You'll need to expose this server using ngrok:")
            print("   ngrok http 5001")
            print("   Then update the webhook URL in the code")
            agent.start_server(port=5001)
        
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