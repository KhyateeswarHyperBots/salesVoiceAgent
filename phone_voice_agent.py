#!/usr/bin/env python3
"""
Phone Voice Agent with Original Architecture
Full phone call agent using Twilio + Ollama + Web Scraping + ChatGPT Web Search + RAG
"""

import os
import json
import time
import threading
import queue
import numpy as np
import datetime
import warnings
import re
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

# Import the original components from main.py
try:
    from client_rag_system import ClientRAGSystem
    from chatgpt_web_search import ChatGPTWebSearch
    from comprehensive_company_research import ComprehensiveCompanyResearch
    from company_website_scraper import CompanyWebsiteScraper
except ImportError:
    print("⚠️ Some advanced components not available, using basic versions")

class RAG:
    """RAG system for document retrieval (from main.py)"""
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

class PhoneVoiceAgent:
    """Phone voice agent using original main.py architecture"""
    
    def __init__(self):
        """Initialize phone voice agent with original architecture"""
        self.account_sid = os.getenv("TWILIO_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.to_number = os.getenv("TO_PHONE_NUMBER")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        
        # Initialize original architecture components
        self.rag = RAG()
        self.client_rag = ClientRAGSystem() if 'ClientRAGSystem' in globals() else None
        self.web_search = ChatGPTWebSearch() if 'ChatGPTWebSearch' in globals() else None
        self.comprehensive_research = ComprehensiveCompanyResearch() if 'ComprehensiveCompanyResearch' in globals() else None
        self.website_scraper = CompanyWebsiteScraper() if 'CompanyWebsiteScraper' in globals() else None
        
        # Conversation state (from main.py)
        self.conversation_history = {}
        self.current_client = None
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sentiment analysis and buying signals tracking (from main.py)
        self._sentiment_history = {}
        self._buying_signals = {}
        self._objection_signals = {}
        self._engagement_score = {}
        self._buying_probability = {}
        
        # Audio sentiment analysis (from main.py)
        self._audio_sentiment_history = {}
        self._voice_emotion_scores = {}
        self._speech_patterns = {}
        self._real_time_audio_buffer = queue.Queue()
        self._audio_analysis_active = False
        
        # Private attributes to store fetched data (from main.py)
        self._web_data = {}
        self._website_data = {}
        self._web_context = {}
        self._website_context = {}
        
        # System instructions
        self.system_instructions = get_system_instructions()
        
        # Flask app for webhooks
        self.app = Flask(__name__)
        self.setup_routes()
        
        print(f"🎤 Phone Voice Agent initialized with original architecture")
        print(f"   From: {self.from_number}")
        print(f"   To: {self.to_number}")
        print(f"   Model: {self.model_name}")
        print(f"   Features: RAG, Web Search, Web Scraping, Sentiment Analysis, Client Data")
    
    def analyze_sentiment(self, text):
        """Analyze text sentiment using TextBlob (from main.py)"""
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
        """Detect buying signals in text (from main.py)"""
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
        """Update buying probability based on conversation (from main.py)"""
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
    
    def fetch_web_data_for_client(self, client):
        """Fetch web data for client (from main.py)"""
        if not self.comprehensive_research or not client:
            return None
        
        try:
            print(f"🌐 Fetching web data for {client.get('Company', 'Unknown')}...")
            research_data = self.comprehensive_research.research_company(client)
            if research_data:
                self._web_data[client.get('Phone')] = research_data
                self._web_context[client.get('Phone')] = self.format_web_data_for_context(client, research_data)
                print(f"✅ Web data fetched for {client.get('Company')}")
            return research_data
        except Exception as e:
            print(f"❌ Error fetching web data: {e}")
            return None
    
    def scrape_company_website(self, client):
        """Scrape company website (from main.py)"""
        if not self.website_scraper or not client:
            return None
        
        try:
            company_name = client.get('Company', '')
            if company_name:
                print(f"🌐 Scraping website for {company_name}...")
                scraped_data = self.website_scraper.scrape_company_website(company_name)
                if scraped_data:
                    self._website_data[client.get('Phone')] = scraped_data
                    self._website_context[client.get('Phone')] = self.format_website_data_for_context(scraped_data)
                    print(f"✅ Website data scraped for {company_name}")
                return scraped_data
        except Exception as e:
            print(f"❌ Error scraping website: {e}")
            return None
    
    def format_web_data_for_context(self, client, research_data):
        """Format research data for conversation context (from main.py)"""
        if not research_data:
            return ""
        
        web_context = []
        
        # Company overview
        company = research_data.get('company_overview', {})
        if company.get('annual_revenue'):
            web_context.append(f"Financial Data: {client.get('Company')} has annual revenue of {company['annual_revenue']}")
        
        if company.get('number_of_employees'):
            web_context.append(f"Company Size: {company['number_of_employees']}")
        
        if company.get('industry'):
            web_context.append(f"Industry: {company['industry']}")
        
        # Financial & AP Operations
        financial = research_data.get('financial_ap_operations', {})
        if financial.get('erp_systems_used'):
            web_context.append(f"ERP Systems: {', '.join(financial['erp_systems_used'])}")
        
        if financial.get('ap_automation_maturity'):
            web_context.append(f"AP Automation Maturity: {financial['ap_automation_maturity']}")
        
        if financial.get('known_pain_points'):
            web_context.append("Key Pain Points:")
            for point in financial['known_pain_points'][:3]:
                web_context.append(f"  • {point}")
        
        # Recent news and trigger events
        news = research_data.get('recent_news_trigger_events', {})
        if news.get('trigger_events'):
            web_context.append("Recent Developments:")
            for event in news['trigger_events'][:2]:
                web_context.append(f"  • {event}")
        
        return "\n".join(web_context)
    
    def format_website_data_for_context(self, scraped_data):
        """Format website data for conversation context (from main.py)"""
        if not scraped_data:
            return ""
        
        website_context = []
        
        # Company description
        if scraped_data.get('company_description'):
            website_context.append(f"Company Description: {scraped_data['company_description'][:200]}...")
        
        # Key products/services
        if scraped_data.get('products_services'):
            website_context.append("Products/Services:")
            for item in scraped_data['products_services'][:3]:
                website_context.append(f"  • {item}")
        
        # Technology stack
        if scraped_data.get('technology_stack'):
            website_context.append(f"Technology Stack: {', '.join(scraped_data['technology_stack'][:5])}")
        
        # Recent updates
        if scraped_data.get('recent_updates'):
            website_context.append("Recent Updates:")
            for update in scraped_data['recent_updates'][:2]:
                website_context.append(f"  • {update}")
        
        return "\n".join(website_context)
    
    def setup_client_session(self, client, call_sid):
        """Setup client session with web data (from main.py)"""
        if not client:
            return
        
        print(f"👤 Setting up session for {client.get('Full Name')} from {client.get('Company')}")
        
        # Initialize conversation history for this call
        self.conversation_history[call_sid] = []
        self._sentiment_history[call_sid] = []
        self._buying_signals[call_sid] = []
        self._objection_signals[call_sid] = []
        self._engagement_score[call_sid] = 0.0
        self._buying_probability[call_sid] = 0.0
        
        # Fetch web data in background
        def fetch_data():
            self.fetch_web_data_for_client(client)
            self.scrape_company_website(client)
        
        threading.Thread(target=fetch_data, daemon=True).start()
    
    def setup_routes(self):
        """Setup Flask routes for Twilio webhooks"""
        
        @self.app.route('/call', methods=['POST'])
        def handle_call():
            """Handle incoming call"""
            call_sid = request.form.get('CallSid')
            from_number = request.form.get('From')
            
            print(f"📞 Incoming call from: {from_number}")
            print(f"   Call SID: {call_sid}")
            
            # Find client using original client RAG system
            if self.client_rag:
                self.current_client = self.client_rag.find_client_by_phone(from_number)
            else:
                # Fallback to basic client lookup
                try:
                    with open('clients.json', 'r') as f:
                        clients = json.load(f)
                    self.current_client = next((c for c in clients if c.get('Phone') == from_number), None)
                except:
                    self.current_client = None
            
            # Setup client session with web data
            self.setup_client_session(self.current_client, call_sid)
            
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
            
            # Analyze sentiment and buying signals (from main.py)
            sentiment_data = self.analyze_sentiment(speech_result)
            buying_signals = self.detect_buying_signals(speech_result)
            
            # Update buying probability
            self.update_buying_probability(speech_result, sentiment_data, buying_signals, call_sid)
            
            # Generate AI response using original architecture
            ai_response = self.generate_enhanced_response(speech_result, call_sid)
            
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
    
    def generate_enhanced_response(self, user_input, call_sid):
        """Generate AI response using original main.py architecture"""
        try:
            # Add to conversation history
            if call_sid not in self.conversation_history:
                self.conversation_history[call_sid] = []
            
            self.conversation_history[call_sid].append({
                'role': 'user',
                'content': user_input
            })
            
            # Get RAG context from documents (from main.py)
            retrieved_docs = self.rag.retrieve(user_input)
            context_parts = []
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    content = doc.get('content', str(doc))
                else:
                    content = str(doc)
                context_parts.append(content)
            
            # Get conversation history context
            conversation_context = ""
            if len(self.conversation_history[call_sid]) > 1:
                recent_exchanges = self.conversation_history[call_sid][-6:]  # Last 3 exchanges
                conversation_context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_exchanges[:-1]  # Exclude current user input
                ])
            
            # Combine RAG context
            rag_context = "\n".join(context_parts)
            
            # Get web data for current client (from main.py)
            web_data = ""
            website_data = ""
            if self.current_client:
                client_phone = self.current_client.get('Phone')
                web_data = self._web_context.get(client_phone, "")
                website_data = self._website_context.get(client_phone, "")
            
            # Create client info
            if self.current_client:
                client_info = f"""
Current Client Information:
- Name: {self.current_client.get('Full Name', 'Unknown')}
- Title: {self.current_client.get('Title', 'Unknown')}
- Company: {self.current_client.get('Company', 'Unknown')}
- Location: {self.current_client.get('City', '')}, {self.current_client.get('State', '')}
"""
            else:
                client_info = ""
            
            # Include web data in context (from main.py)
            web_context = f"\nChatGPT Web Research Data:\n{web_data}" if web_data else ""
            website_context = f"\nWebsite Scraping Data:\n{website_data}" if website_data else ""
            combined_web_data = f"{web_context}{website_context}"
            
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
            
            # Create enhanced prompt (from main.py)
            if conversation_context:
                prompt = f"""{self.system_instructions}

{client_info}
Context from documents: {rag_context}{combined_web_data}

Previous conversation:
{conversation_context}

Buying Context: {buying_context}

Current question: {user_input}

Please provide a personalized response that takes into account the client's role, company, web research data, website scraping data, and our conversation history. Keep it concise for phone conversation."""
            else:
                prompt = f"""{self.system_instructions}

{client_info}
Context: {rag_context}{combined_web_data}

Buying Context: {buying_context}

Question: {user_input}

Please provide a personalized response based on the context, web research, website data, and your role as a sales executive. Keep it concise for phone conversation."""
            
            print(f"🤖 Generating enhanced response with web data...")
            
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
            
            print(f"📞 Making enhanced call to: {target_number}")
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
        """Get webhook URL"""
        return "https://your-ngrok-url.ngrok.io"
    
    def start_server(self, port=5002):
        """Start Flask server for webhooks"""
        print(f"🌐 Starting phone voice agent server on port {port}")
        print(f"   Webhook URL: {self.get_webhook_url()}")
        print("   Use ngrok to expose this server: ngrok http 5002")
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
            filename = f'phone_conversations/phone_conversation_{call_sid}_{timestamp}.json'
            
            conversation_data = {
                'call_sid': call_sid,
                'timestamp': timestamp,
                'client': self.current_client.get('Full Name') if self.current_client else 'Unknown',
                'conversation': self.conversation_history[call_sid],
                'sentiment_history': self._sentiment_history.get(call_sid, []),
                'buying_signals': self._buying_signals.get(call_sid, []),
                'objection_signals': self._objection_signals.get(call_sid, []),
                'final_buying_probability': self._buying_probability.get(call_sid, 0.0),
                'engagement_score': self._engagement_score.get(call_sid, 0.0),
                'web_data_used': bool(self._web_context.get(self.current_client.get('Phone') if self.current_client else None)),
                'website_data_used': bool(self._website_context.get(self.current_client.get('Phone') if self.current_client else None))
            }
            
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            print(f"💾 Phone conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")


def main():
    """Main function"""
    print("🎤 Phone Voice Agent with Original Architecture")
    print("=" * 60)
    print("Features: Web Scraping + ChatGPT Web Search + RAG + Sentiment Analysis")
    print("=" * 60)
    
    # Check credentials
    if not os.getenv("TWILIO_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("❌ Twilio credentials not found")
        print("   Please set TWILIO_SID and TWILIO_AUTH_TOKEN")
        return
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        model_count = len(list(models))
        print(f"✅ Ollama connected with {model_count} models")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   Please start Ollama: ollama serve")
        return
    
    # Create agent
    agent = PhoneVoiceAgent()
    
    # Show menu
    while True:
        print("\n🎤 Phone Voice Agent Menu:")
        print("1. Start webhook server (for incoming calls)")
        print("2. Make outbound call")
        print("3. List recent calls")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🌐 Starting phone voice agent server...")
            print("   Features: Web Scraping, ChatGPT Web Search, RAG, Sentiment Analysis")
            print("   You'll need to expose this server using ngrok:")
            print("   ngrok http 5002")
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