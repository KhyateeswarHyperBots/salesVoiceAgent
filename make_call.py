#!/usr/bin/env python3
"""
Make Outbound Call with Original Architecture
Simple script to make a phone call using the original main.py architecture
"""

import os
import json
import time
import threading
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
from enhanced_sentiment_analysis import EnhancedSentimentAnalysis

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
        self.index_path = "vector.index"
        self.embeddings_path = "embeddings.pkl"

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

    def retrieve(self, query, top_k=3):
        """Retrieve relevant documents"""
        if self.index is None:
            self.load_index()
            if self.index is None:
                return ["No documents available for retrieval."]
        
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in I[0]]

class OutboundCallAgent:
    """Outbound call agent using original main.py architecture"""
    
    def __init__(self):
        """Initialize outbound call agent"""
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
        
        # Conversation state
        self.conversation_history = []
        self.current_client = None
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced sentiment analysis
        self.sentiment_analyzer = EnhancedSentimentAnalysis(model_type="ensemble")
        
        # Sentiment analysis and buying signals tracking
        self._sentiment_history = []
        self._buying_signals = []
        self._objection_signals = []
        self._engagement_score = 0.0
        self._buying_probability = 0.0
        
        # Private attributes to store fetched data
        self._web_data = None
        self._website_data = None
        self._web_context = None
        self._website_context = None
        
        # System instructions
        self.system_instructions = get_system_instructions()
        
        # Flask app for webhooks
        self.app = Flask(__name__)
        self.setup_routes()
        
        print(f"🎤 Outbound Call Agent initialized")
        print(f"   From: {self.from_number}")
        print(f"   To: {self.to_number}")
        print(f"   Model: {self.model_name}")
        print(f"   Features: Web Scraping, ChatGPT Web Search, RAG, Enhanced Sentiment Analysis")
        print(f"   Sentiment Models: {list(self.sentiment_analyzer.models.keys())}")
    
    def analyze_sentiment(self, text):
        """Analyze text sentiment using enhanced models"""
        try:
            # Use enhanced sentiment analysis
            result = self.sentiment_analyzer.analyze_sentiment(text)
            
            # Print detailed intent analysis
            sales_signals = result.get('sales_signals', {})
            intent = sales_signals.get('intent', 'neutral')
            confidence = sales_signals.get('confidence', 0.5)
            buying_score = sales_signals.get('buying_score', 0)
            objection_score = sales_signals.get('objection_score', 0)
            positive_score = sales_signals.get('positive_score', 0)
            negative_score = sales_signals.get('negative_score', 0)
            
            # Calculate intent percentage
            intent_percentage = confidence * 100
            
            print(f"🎯 INTENT: {intent.upper()} ({intent_percentage:.1f}%)")
            print(f"   📈 Buying Score: {buying_score}, Objection Score: {objection_score}")
            print(f"   ✅ Positive Score: {positive_score}, ❌ Negative Score: {negative_score}")
            
            # Show detected keywords if any
            if sales_signals.get('buying_signals'):
                print(f"   🎯 Buying Signals: {', '.join(sales_signals['buying_signals'])}")
            if sales_signals.get('objections'):
                print(f"   ⚠️ Objections: {', '.join(sales_signals['objections'])}")
            if sales_signals.get('positive_words'):
                print(f"   ✅ Positive Words: {', '.join(sales_signals['positive_words'])}")
            if sales_signals.get('negative_words'):
                print(f"   ❌ Negative Words: {', '.join(sales_signals['negative_words'])}")
            
            return result
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'sentiment': 'neutral',
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sales_signals': {
                    'intent': 'neutral',
                    'confidence': 0.5
                }
            }
    

    
    def update_buying_probability(self, user_input, sentiment_data, buying_signals):
        """Update buying probability based on conversation (from main.py)"""
        # Use enhanced buying probability calculation
        new_prob = self.sentiment_analyzer.get_buying_probability(sentiment_data)
        
        # Update probability with some smoothing
        self._buying_probability = (self._buying_probability * 0.7) + (new_prob * 0.3)
        
        print(f"💰 Buying probability: {self._buying_probability:.1f}%")
    

    
    def fetch_web_data_for_client(self, client):
        """Fetch web data for client (from main.py)"""
        if not self.comprehensive_research or not client:
            return None
        
        try:
            print(f"🌐 Fetching web data for {client.get('Company', 'Unknown')}...")
            research_data = self.comprehensive_research.research_company(client)
            if research_data:
                self._web_data = research_data
                self._web_context = self.format_web_data_for_context(client, research_data)
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
                    self._website_data = scraped_data
                    self._website_context = self.format_website_data_for_context(scraped_data)
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
    
    def setup_client_session(self, client):
        """Setup client session with web data (from main.py)"""
        if not client:
            return
        
        print(f"👤 Setting up session for {client.get('Full Name')} from {client.get('Company')}")
        
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
            
            # Use the stored original client number for client data lookup
            original_client_number = getattr(self, 'original_client_number', self.to_number)

            # Find client using flexible phone matching (primary method)
            try:
                with open('clients.json', 'r') as f:
                    clients = json.load(f)
                
                # Normalize the input phone number (remove + and spaces)
                normalized_input = (original_client_number or '').replace('+', '').replace(' ', '')
                
                # Try to find client with flexible phone matching
                self.current_client = None
                for client in clients:
                    client_phone = client.get('Phone', '')
                    if client_phone:
                        # Normalize client phone number
                        normalized_client_phone = client_phone.replace('+', '').replace(' ', '')
                        if normalized_client_phone == normalized_input:
                            self.current_client = client
                            break
                
                # If not found with basic lookup, try client RAG system as fallback
                if not self.current_client and self.client_rag:
                    print("🔄 Trying client RAG system as fallback...")
                    self.current_client = self.client_rag.search_client_by_phone(original_client_number)
                
            except Exception as e:
                print(f"❌ Error in client lookup: {e}")
                self.current_client = None

            print(f"📞 Call from: {from_number}")
            print(f"👤 Using client data from: {original_client_number}")
            if self.current_client:
                print(f"   Client: {self.current_client.get('Full Name')} from {self.current_client.get('Company')}")
            else:
                print("   No client data found")
            
            # Setup client session with web data
            self.setup_client_session(self.current_client)
            
            # Track call start time
            self.call_start_time = datetime.datetime.now()
            
            # Create TwiML response
            response = VoiceResponse()
            
            if self.current_client:
                welcome_message = f"Hi {self.current_client.get('Full Name')}, this is Sunny from Hyprbots. I'm calling because companies like {self.current_client.get('Company', 'yours')} are losing thousands of dollars monthly on manual invoice processing. We've helped similar companies cut that cost by 90%. You're probably spending 10+ hours a week on this - am I right?"
            else:
                welcome_message = "Hi, this is Sunny from Hyprbots. I'm calling because companies are losing thousands of dollars monthly on manual invoice processing. We've helped similar companies cut that cost by 90%. You're probably spending 10+ hours a week on this - am I right?"
            
            # Speak welcome message
            response.say(welcome_message, voice='alice')
            
            # Gather user input
            gather = Gather(
                input='speech',
                timeout=15,
                speech_timeout=3,
                action='/process_speech',
                method='POST'
            )
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
                    timeout=15,
                    speech_timeout=3,
                    action='/process_speech',
                    method='POST'
                )
                response.append(gather)
                response.say("Thank you for calling. Goodbye!", voice='alice')
                response.hangup()
                return Response(str(response), mimetype='text/xml')
            
            # Check for end call keywords
            if any(word in speech_result.lower() for word in ['goodbye', 'bye', 'end call', 'hang up', 'stop']):
                # Analyze sentiment and buying signals before ending
                sentiment_data = self.analyze_sentiment(speech_result)
                buying_signals = sentiment_data.get('sales_signals', {})
                
                # Update buying probability one final time
                self.update_buying_probability(speech_result, sentiment_data, buying_signals)
                
                # Print final intent analysis
                if sentiment_data:
                    sales_signals = sentiment_data.get('sales_signals', {})
                    intent = sales_signals.get('intent', 'neutral')
                    confidence = sales_signals.get('confidence', 0.5)
                    buying_score = sales_signals.get('buying_score', 0)
                    objection_score = sales_signals.get('objection_score', 0)
                    positive_score = sales_signals.get('positive_score', 0)
                    negative_score = sales_signals.get('negative_score', 0)
                    
                    intent_percentage = confidence * 100
                    
                    print(f"🎯 FINAL INTENT: {intent.upper()} ({intent_percentage:.1f}%)")
                    print(f"   📈 Final Buying Score: {buying_score}, Objection Score: {objection_score}")
                    print(f"   ✅ Final Positive Score: {positive_score}, ❌ Negative Score: {negative_score}")
                    print(f"   💰 Final Buying Probability: {self._buying_probability:.1f}%")
                
                # Save call summary before ending
                self.save_call_summary("Call ended by user")
                
                # Create proper goodbye response
                response = VoiceResponse()
                response.say("Thank you for calling. Have a great day!", voice='alice')
                response.hangup()
                
                print("👋 Saying goodbye and ending call...")
                return Response(str(response), mimetype='text/xml')
            
            # Check for scheduling requests first
            scheduling_keywords = ['schedule', 'book', 'appointment', 'demo', 'meeting', 'show me', 'give me', 'set up']
            if any(keyword in speech_result.lower() for keyword in scheduling_keywords):
                schedule_response = self.schedule_event(speech_result)
                response = VoiceResponse()
                response.say(schedule_response, voice='alice')
                
                # Continue conversation
                gather = Gather(
                    input='speech',
                    timeout=15,
                    speech_timeout=3,
                    action='/process_speech',
                    method='POST'
                )
                response.append(gather)
                
                return Response(str(response), mimetype='text/xml')
            
            # Analyze sentiment and buying signals (from main.py)
            sentiment_data = self.analyze_sentiment(speech_result)
            buying_signals = sentiment_data.get('sales_signals', {})
            
            # Update buying probability
            self.update_buying_probability(speech_result, sentiment_data, buying_signals)
            
            # Generate AI response using original architecture
            ai_response = self.generate_enhanced_response(speech_result, sentiment_data)
            
            # Create TwiML response
            response = VoiceResponse()
            response.say(ai_response, voice='alice')
            
            # Continue conversation
            gather = Gather(
                input='speech',
                timeout=15,
                speech_timeout=3,
                action='/process_speech',
                method='POST'
            )
            response.append(gather)
            
            # Debug: Print the TwiML response
            twiml_response = str(response)
            print(f"📞 TwiML Response: {twiml_response}")
            
            # Only end call if no input is received (this will be handled by the gather timeout)
            # Don't add goodbye message here - let the gather handle the flow
            
            return Response(twiml_response, mimetype='text/xml')
        
        @self.app.route('/timeout', methods=['POST'])
        def handle_timeout():
            """Handle when user doesn't respond within timeout"""
            # Save call summary before ending
            self.save_call_summary("Call ended due to timeout")
            
            response = VoiceResponse()
            response.say("I didn't hear from you. Thank you for calling Hyprbots. Have a great day!", voice='alice')
            response.hangup()
            return Response(str(response), mimetype='text/xml')
    
    def generate_enhanced_response(self, user_input, sentiment_data=None):
        """Generate AI response using original main.py architecture"""
        try:
            # Add to conversation history
            self.conversation_history.append({
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
            if len(self.conversation_history) > 1:
                recent_exchanges = self.conversation_history[-6:]  # Last 3 exchanges
                conversation_context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_exchanges[:-1]  # Exclude current user input
                ])
            
            # Combine RAG context
            rag_context = "\n".join(context_parts)
            
            # Get web data for current client (from main.py)
            web_data = self._web_context if self._web_context else ""
            website_data = self._website_context if self._website_context else ""
            
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
            if self._buying_probability > 70:
                buying_context = "The customer shows strong buying interest. Focus on next steps and closing."
            elif self._buying_probability > 40:
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
            
            # Clean and limit the response for Twilio speech synthesis
            # Remove quotes and limit length for better speech synthesis
            ai_response = ai_response.replace('"', '').replace('"', '').replace('"', '')
            ai_response = ai_response.replace('\n', ' ').replace('  ', ' ')
            
            # Remove ALL meta-commentary and approach explanations
            meta_phrases = [
                "Here's my approach:", "This response aims to:", "Here's a personalized response:",
                "I'd be happy to help you craft", "Given Jane's low interest", "I'll focus on building",
                "This response aims to:", "Here's what I suggest:", "My approach would be:",
                "I'd recommend:", "Here's how I'd respond:", "Let me craft a response:"
            ]
            
            for phrase in meta_phrases:
                if phrase in ai_response:
                    ai_response = ai_response.split(phrase)[-1]
            
            # Ensure it ends properly
            if not ai_response.endswith(('.', '!', '?')):
                ai_response = ai_response.rstrip() + '.'
            
            # Add to conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': ai_response
            })
            
            print(f"🤖 AI Response: {ai_response}")
            
            # Print intent analysis after AI response
            if sentiment_data:
                sales_signals = sentiment_data.get('sales_signals', {})
                intent = sales_signals.get('intent', 'neutral')
                confidence = sales_signals.get('confidence', 0.5)
                buying_score = sales_signals.get('buying_score', 0)
                objection_score = sales_signals.get('objection_score', 0)
                positive_score = sales_signals.get('positive_score', 0)
                negative_score = sales_signals.get('negative_score', 0)
                
                # Calculate intent percentage
                intent_percentage = confidence * 100
                
                print(f"🎯 INTENT: {intent.upper()} ({intent_percentage:.1f}%)")
                print(f"   📈 Buying Score: {buying_score}, Objection Score: {objection_score}")
                print(f"   ✅ Positive Score: {positive_score}, ❌ Negative Score: {negative_score}")
                print(f"   💰 Buying probability: {self._buying_probability:.1f}%")
                
                # Show detected keywords if any
                if sales_signals.get('buying_signals'):
                    print(f"   🎯 Buying Signals: {', '.join(sales_signals['buying_signals'])}")
                if sales_signals.get('objections'):
                    print(f"   ⚠️ Objections: {', '.join(sales_signals['objections'])}")
                if sales_signals.get('positive_words'):
                    print(f"   ✅ Positive Words: {', '.join(sales_signals['positive_words'])}")
                if sales_signals.get('negative_words'):
                    print(f"   ❌ Negative Words: {', '.join(sales_signals['negative_words'])}")
            
            return ai_response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Could you please try again?"
    
    def make_call(self):
        """Make an outbound call"""
        try:
            # Ask for client phone number (for data lookup)
            print(f"📞 Outbound call will go to: +918688164030 (test number)")
            client_number = input("Enter client phone number for data lookup: ").strip()
            
            if client_number:
                self.original_client_number = client_number
                print(f"📞 Making outbound call to: +918688164030")
                print(f"👤 Using client data from: {self.original_client_number}")
            else:
                self.original_client_number = "+14048197966"  # Default client number
                print(f"📞 Making outbound call to: +918688164030")
                print(f"👤 Using client data from: {self.original_client_number}")
            
            print("   Starting webhook server...")
            
            # Start the server in background
            def start_server():
                self.app.run(host='0.0.0.0', port=5002, debug=False)
            
            server_thread = threading.Thread(target=start_server, daemon=True)
            server_thread.start()
            
            # Wait for server to start
            time.sleep(3)
            
            # Make the call (hardcoded to test number)
            call = self.client.calls.create(
                to="+918688164030",
                from_=self.from_number,
                url=f"{self.get_webhook_url()}/call",
                method='POST'
            )
            
            print(f"✅ Call initiated: {call.sid}")
            print("   Server is running on port 5002")
            print("   Use ngrok to expose: ngrok http 5002")
            print("   Then update the webhook URL in get_webhook_url() method")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Call ended")
            
        except Exception as e:
            print(f"❌ Error making call: {e}")
    
    def parse_schedule_request(self, user_input):
        """Parse user input to extract scheduling information"""
        user_input_lower = user_input.lower()
        
        # Default values
        event_title = "Sales Call"
        duration = 30
        start_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        attendees = []
        
        # Extract email addresses from user input
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, user_input)
        attendees.extend(emails)
        
        # Extract event title
        if "demo" in user_input_lower:
            event_title = "Product Demo Call"
        elif "meeting" in user_input_lower:
            event_title = "Sales Meeting"
        elif "call" in user_input_lower:
            event_title = "Sales Call"
        
        # Extract duration
        import re
        duration_match = re.search(r'(\d+)\s*(min|minute|minutes|hour|hours)', user_input_lower)
        if duration_match:
            value = int(duration_match.group(1))
            unit = duration_match.group(2)
            if unit in ['hour', 'hours']:
                duration = value * 60
            else:
                duration = value
        
        # Extract time information
        time_patterns = [
            r'tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
            r'today\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
            r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
            r'in\s+(\d+)\s*(hour|hours|minute|minutes)',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                if 'tomorrow' in pattern:
                    start_time = datetime.datetime.now() + datetime.timedelta(days=1)
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    ampm = match.group(3)
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                    start_time = start_time.replace(hour=hour, minute=minute)
                    break
                elif 'today' in pattern:
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    ampm = match.group(3)
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                    start_time = datetime.datetime.now().replace(hour=hour, minute=minute)
                    if start_time < datetime.datetime.now():
                        start_time += datetime.timedelta(days=1)
                    break
                elif 'in' in pattern:
                    value = int(match.group(1))
                    unit = match.group(2)
                    if unit in ['hour', 'hours']:
                        start_time = datetime.datetime.now() + datetime.timedelta(hours=value)
                    else:
                        start_time = datetime.datetime.now() + datetime.timedelta(minutes=value)
                    break
                else:
                    # Just time specified, assume today
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    ampm = match.group(3)
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                    start_time = datetime.datetime.now().replace(hour=hour, minute=minute)
                    if start_time < datetime.datetime.now():
                        start_time += datetime.timedelta(days=1)
                    break
        
        return event_title, start_time, duration, attendees

    def schedule_event(self, user_input):
        """Create a calendar event based on user input"""
        try:
            event_title, start_time, duration, parsed_attendees = self.parse_schedule_request(user_input)
            
            # Get client info for description
            client_info = ""
            if self.current_client:
                client_info = f"Client: {self.current_client.get('Full Name')} from {self.current_client.get('Company')}"
            
            # Create the calendar event
            from calendar_helper import create_event
            
            # Prepare attendees list with client contact info
            attendees = []
            
            # Add any emails mentioned in the conversation
            attendees.extend(parsed_attendees)
            
            # Add current client details
            if self.current_client:
                # Add client email if available and not already in attendees
                client_email = self.current_client.get('Email', '')
                if client_email and client_email not in attendees:
                    attendees.append(client_email)
                # Add client phone if no email and not already in attendees
                elif self.current_client.get('Phone', ''):
                    client_phone = f"Phone: {self.current_client.get('Phone')}"
                    if client_phone not in attendees:
                        attendees.append(client_phone)
                # Add client name as fallback
                else:
                    client_name = f"Client: {self.current_client.get('Full Name', 'Unknown')}"
                    if client_name not in attendees:
                        attendees.append(client_name)
            
            # Get current sales intelligence data
            sales_intelligence = {
                "buying_probability": self._buying_probability,
                "intent": "neutral",
                "intent_confidence": 0.0,
                "buying_signals": len(self._buying_signals),
                "objection_signals": len(self._objection_signals),
                "engagement_score": self._engagement_score
            }
            
            # Get latest sentiment data if available
            if self._sentiment_history:
                latest_sentiment = self._sentiment_history[-1]
                sales_intelligence.update({
                    "sentiment": latest_sentiment.get('sentiment', 'neutral'),
                    "polarity": latest_sentiment.get('polarity', 0.0),
                    "subjectivity": latest_sentiment.get('subjectivity', 0.0)
                })
            
            # Get sales signals from latest sentiment analysis
            if self._sentiment_history:
                latest_sales_signals = latest_sentiment.get('sales_signals', {})
                sales_intelligence.update({
                    "intent": latest_sales_signals.get('intent', 'neutral'),
                    "intent_confidence": latest_sales_signals.get('confidence', 0.0)
                })
            
            # Create enhanced description with sales intelligence
            enhanced_description = f"""Automatically scheduled based on: {user_input}
{client_info}

SALES INTELLIGENCE:
- Buying Probability: {self._buying_probability:.1f}%
- Intent: {sales_intelligence['intent'].upper()} ({sales_intelligence['intent_confidence']*100:.1f}% confidence)
- Sentiment: {sales_intelligence.get('sentiment', 'neutral')} (Polarity: {sales_intelligence.get('polarity', 0.0):.2f})
- Buying Signals Detected: {sales_intelligence['buying_signals']}
- Objection Signals Detected: {sales_intelligence['objection_signals']}
- Engagement Score: {sales_intelligence['engagement_score']:.2f}"""
            
            result = create_event(
                summary=event_title,
                start_time=start_time,
                duration_minutes=duration,
                description=enhanced_description,
                location="Phone Call",
                attendees=attendees,
                sales_intelligence=sales_intelligence
            )
            
            return result
            
        except Exception as e:
            print(f"Error creating calendar event: {e}")
            return "I'm sorry, I couldn't create the calendar event. Please try again."

    def save_call_summary(self, end_reason="Call ended normally"):
        """Save detailed call summary with sales intelligence"""
        try:
            # Calculate call duration
            call_start = getattr(self, 'call_start_time', datetime.datetime.now())
            call_duration = (datetime.datetime.now() - call_start).total_seconds()
            
            # Prepare call summary
            call_summary = {
                "session_id": self.session_id,
                "call_start_time": call_start.isoformat(),
                "call_end_time": datetime.datetime.now().isoformat(),
                "call_duration_seconds": call_duration,
                "end_reason": end_reason,
                "client_info": {
                    "name": self.current_client.get('Full Name', 'Unknown') if self.current_client else 'Unknown',
                    "company": self.current_client.get('Company', 'Unknown') if self.current_client else 'Unknown',
                    "phone": self.current_client.get('Phone', 'Unknown') if self.current_client else 'Unknown',
                    "email": self.current_client.get('Email', '') if self.current_client else ''
                },
                "sales_intelligence": {
                    "final_buying_probability": self._buying_probability,
                    "total_buying_signals": len(self._buying_signals),
                    "total_objection_signals": len(self._objection_signals),
                    "engagement_score": self._engagement_score,
                    "sentiment_history": self._sentiment_history[-5:] if self._sentiment_history else [],  # Last 5 sentiments
                    "conversation_turns": len(self.conversation_history)
                },
                "conversation_summary": {
                    "total_exchanges": len(self.conversation_history),
                    "user_inputs": [msg['content'] for msg in self.conversation_history if msg['role'] == 'user'],
                    "ai_responses": [msg['content'] for msg in self.conversation_history if msg['role'] == 'assistant']
                },
                "calendar_events_created": len([event for event in self._get_recent_calendar_events() if event.get('created_at', '').startswith(datetime.datetime.now().strftime('%Y-%m-%d'))])
            }
            
            # Save to call summaries file
            summaries_file = "call_summaries.json"
            summaries = []
            
            if os.path.exists(summaries_file):
                try:
                    with open(summaries_file, 'r') as f:
                        summaries = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    summaries = []
            
            summaries.append(call_summary)
            
            with open(summaries_file, 'w') as f:
                json.dump(summaries, f, indent=2)
            
            print(f"📊 Call summary saved: {call_summary['client_info']['name']} - {call_summary['sales_intelligence']['final_buying_probability']:.1f}% buying probability")
            
        except Exception as e:
            print(f"❌ Error saving call summary: {e}")
    
    def _get_recent_calendar_events(self):
        """Get recent calendar events for this session"""
        try:
            from calendar_helper import list_upcoming_events
            events = list_upcoming_events(days_ahead=30)
            return events
        except:
            return []

    def get_webhook_url(self):
        """Get webhook URL"""
        return "https://3a29b19f0015.ngrok-free.app"


def main():
    """Main function - just make the call"""
    print("🎤 Making Outbound Call with Original Architecture")
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
    
    # Create agent and make call
    agent = OutboundCallAgent()
    agent.make_call()


if __name__ == "__main__":
    main() 