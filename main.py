import os
import json
import warnings
import pyttsx3
import speech_recognition as sr
import faiss
import ollama
from sentence_transformers import SentenceTransformer
import re
import datetime
import time
import sys
import argparse
from calendar_helper import create_event
import textblob
from textblob import TextBlob
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import threading
import queue

# Suppress warnings
warnings.filterwarnings('ignore')
from config import (
    OPENAI_API_KEY, 
    VOICE_SETTINGS, 
    AI_MODEL, 
    SPEECH_RECOGNITION, 
    NON_INTERRUPTION_PERIOD,
    DOC_PATH,
    INDEX_PATH,
    EMBEDDINGS_PATH,
    CALENDAR_PATH
)

# Set environment variables to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), 'models')


from calendar_helper import create_event
from agent_config import get_system_instructions, AGENT_CONFIG
from client_rag_system import ClientRAGSystem
from chatgpt_web_search import ChatGPTWebSearch
from comprehensive_company_research import ComprehensiveCompanyResearch
from company_website_scraper import CompanyWebsiteScraper
import datetime

# Example usage:
now = datetime.datetime.now() + datetime.timedelta(hours=1)
# create_event("Call with Customer", now, duration_minutes=15)

class RAG:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def build_index(self, docs):
        self.documents = docs
        embeddings = self.model.encode(docs)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
        with open(EMBEDDINGS_PATH, "wb") as f:
            import pickle
            pickle.dump(docs, f)

    def load_index(self):
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(EMBEDDINGS_PATH, "rb") as f:
                import pickle
                self.documents = pickle.load(f)
            print(f"✅ Loaded RAG index with {len(self.documents)} documents")
        except Exception as e:
            print(f"❌ Error loading RAG index: {e}")
            print("🔄 Rebuilding index...")
            # If loading fails, rebuild the index
            with open(DOC_PATH) as f:
                docs = json.load(f)
            self.build_index(docs)

    def retrieve(self, query, top_k=3):
        if self.index is None:
            print("⚠️ RAG index not loaded, attempting to load...")
            self.load_index()
            if self.index is None:
                return ["No documents available for retrieval."]
        
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in I[0]]

class VoiceAgent:
    def __init__(self, phone_number=None):
        self.rag = RAG()
        self.client_rag = ClientRAGSystem()  # Add client RAG system
        self.web_search = ChatGPTWebSearch()  # Add ChatGPT web search
        self.comprehensive_research = ComprehensiveCompanyResearch()  # Add comprehensive research
        self.website_scraper = CompanyWebsiteScraper()  # Add website scraper
        self.model_name = AI_MODEL
        self.tts = pyttsx3.init()
        self.listener = sr.Recognizer()
        self.conversation_history = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_client = None  # Track current client
        self.phone_number = phone_number  # Store phone number from command line
        
        # Private attributes to store fetched data (avoid repeated calls)
        self._web_data = None  # Store ChatGPT web research data
        self._website_data = None  # Store website scraping data
        self._web_context = None  # Store formatted web context
        self._website_context = None  # Store formatted website context
        
        # Sentiment analysis and buying signals tracking
        self._sentiment_history = []  # Track sentiment over time
        self._buying_signals = []  # Track positive buying signals
        self._objection_signals = []  # Track objections/concerns
        self._engagement_score = 0.0  # Overall engagement score
        self._buying_probability = 0.0  # Probability of purchase (0-100%)
        
        # Audio sentiment analysis
        self._audio_sentiment_history = []  # Track audio sentiment
        self._voice_emotion_scores = []  # Track voice emotions
        self._speech_patterns = []  # Track speech patterns
        self._real_time_audio_buffer = queue.Queue()  # Real-time audio buffer
        self._audio_analysis_active = False  # Audio analysis state
        
        # System instructions for the AI model
        self.system_instructions = get_system_instructions()
        
        # Voice settings
        self.setup_voice()
        
        # Apply voice settings from config
        self.apply_voice_settings()
        
        # Start real-time audio analysis
        self.start_audio_analysis()
    
    def setup_voice(self):
        """Setup default voice settings"""
        # Get available voices
        voices = self.tts.getProperty('voices')
        
        # Set default voice (usually index 0 is the default)
        if voices:
            self.tts.setProperty('voice', voices[0].id)
        
        # Set default properties from config
        self.tts.setProperty('rate', VOICE_SETTINGS['rate'])
        self.tts.setProperty('volume', VOICE_SETTINGS['volume'])
        
        # Store voice settings
        self.voice_settings = {
            'rate': VOICE_SETTINGS['rate'],
            'volume': VOICE_SETTINGS['volume'],
            'voice_id': voices[0].id if voices else None
        }
        
        print(f"🎤 Voice initialized: Rate={self.voice_settings['rate']}, Volume={self.voice_settings['volume']}")
    
    def analyze_audio_sentiment(self, audio_data, sample_rate=16000):
        """Analyze sentiment from audio characteristics"""
        try:
            # Convert audio data to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            
            # Extract audio features
            features = self._extract_audio_features(audio_array, sample_rate)
            
            # Analyze voice characteristics
            voice_analysis = self._analyze_voice_characteristics(features)
            
            # Determine emotion from voice
            emotion = self._classify_voice_emotion(features)
            
            # Calculate confidence and engagement
            confidence_score = self._calculate_confidence_score(features)
            engagement_score = self._calculate_engagement_score(features)
            
            return {
                'emotion': emotion,
                'confidence': confidence_score,
                'engagement': engagement_score,
                'voice_characteristics': voice_analysis,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in audio sentiment analysis: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'engagement': 0.5,
                'voice_characteristics': {},
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _extract_audio_features(self, audio_array, sample_rate):
        """Extract audio features for analysis"""
        features = {}
        
        try:
            # Basic audio statistics
            features['rms_energy'] = np.sqrt(np.mean(audio_array**2))
            features['zero_crossing_rate'] = np.mean(np.diff(np.sign(audio_array)))
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate))
            
            # Pitch and frequency features
            pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sample_rate)
            features['pitch_mean'] = np.mean(pitches[magnitudes > 0.1])
            features['pitch_std'] = np.std(pitches[magnitudes > 0.1])
            
            # Spectral features
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            # MFCC features (voice timbre)
            mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sample_rate)
            features['tempo'] = tempo
            
            # Voice activity detection
            features['voice_activity'] = np.sum(np.abs(audio_array) > 0.01) / len(audio_array)
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            features = {
                'rms_energy': 0.0,
                'zero_crossing_rate': 0.0,
                'spectral_centroid': 0.0,
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'spectral_rolloff': 0.0,
                'mfcc_mean': np.zeros(13),
                'mfcc_std': np.zeros(13),
                'tempo': 0.0,
                'voice_activity': 0.0
            }
        
        return features
    
    def _analyze_voice_characteristics(self, features):
        """Analyze voice characteristics for sentiment"""
        characteristics = {}
        
        # Energy analysis
        if features['rms_energy'] > 0.1:
            characteristics['energy_level'] = 'high'
        elif features['rms_energy'] > 0.05:
            characteristics['energy_level'] = 'medium'
        else:
            characteristics['energy_level'] = 'low'
        
        # Pitch analysis
        if features['pitch_std'] > 50:
            characteristics['pitch_variation'] = 'high'
        elif features['pitch_std'] > 20:
            characteristics['pitch_variation'] = 'medium'
        else:
            characteristics['pitch_variation'] = 'low'
        
        # Speech rate analysis
        if features['tempo'] > 120:
            characteristics['speech_rate'] = 'fast'
        elif features['tempo'] > 80:
            characteristics['speech_rate'] = 'normal'
        else:
            characteristics['speech_rate'] = 'slow'
        
        # Voice clarity
        if features['spectral_centroid'] > 2000:
            characteristics['voice_clarity'] = 'clear'
        else:
            characteristics['voice_clarity'] = 'muffled'
        
        return characteristics
    
    def _classify_voice_emotion(self, features):
        """Classify emotion from voice features"""
        # Simple emotion classification based on audio features
        emotion_scores = {
            'excited': 0.0,
            'interested': 0.0,
            'neutral': 0.0,
            'concerned': 0.0,
            'frustrated': 0.0
        }
        
        # High energy + high pitch variation = excited
        if features['rms_energy'] > 0.08 and features['pitch_std'] > 40:
            emotion_scores['excited'] += 0.4
        
        # Medium energy + moderate pitch variation = interested
        if 0.04 < features['rms_energy'] < 0.08 and 20 < features['pitch_std'] < 40:
            emotion_scores['interested'] += 0.4
        
        # Low energy + low pitch variation = concerned/frustrated
        if features['rms_energy'] < 0.04 and features['pitch_std'] < 20:
            emotion_scores['concerned'] += 0.3
            emotion_scores['frustrated'] += 0.2
        
        # High spectral centroid = clear, engaged voice
        if features['spectral_centroid'] > 2000:
            emotion_scores['interested'] += 0.2
            emotion_scores['excited'] += 0.1
        
        # Return the emotion with highest score
        return max(emotion_scores, key=emotion_scores.get)
    
    def _calculate_confidence_score(self, features):
        """Calculate confidence score from voice features"""
        confidence = 0.5  # Base confidence
        
        # High energy indicates confidence
        if features['rms_energy'] > 0.08:
            confidence += 0.2
        
        # Clear voice indicates confidence
        if features['spectral_centroid'] > 2000:
            confidence += 0.15
        
        # Moderate pitch variation indicates confidence
        if 20 < features['pitch_std'] < 50:
            confidence += 0.15
        
        return min(1.0, confidence)
    
    def _calculate_engagement_score(self, features):
        """Calculate engagement score from voice features"""
        engagement = 0.5  # Base engagement
        
        # High voice activity indicates engagement
        if features['voice_activity'] > 0.7:
            engagement += 0.2
        
        # High energy indicates engagement
        if features['rms_energy'] > 0.06:
            engagement += 0.15
        
        # Pitch variation indicates engagement
        if features['pitch_std'] > 25:
            engagement += 0.15
        
        return min(1.0, engagement)
    
    def start_audio_analysis(self):
        """Start real-time audio analysis"""
        self._audio_analysis_active = True
        self._audio_analysis_thread = threading.Thread(target=self._audio_analysis_worker)
        self._audio_analysis_thread.daemon = True
        self._audio_analysis_thread.start()
        print("🎤 Real-time audio analysis started")
    
    def stop_audio_analysis(self):
        """Stop real-time audio analysis"""
        self._audio_analysis_active = False
        print("🎤 Real-time audio analysis stopped")
    
    def _audio_analysis_worker(self):
        """Background worker for audio analysis"""
        while self._audio_analysis_active:
            try:
                # Get audio data from buffer
                audio_data = self._real_time_audio_buffer.get(timeout=0.1)
                
                # Analyze audio sentiment
                audio_sentiment = self.analyze_audio_sentiment(audio_data)
                
                # Store results
                self._audio_sentiment_history.append(audio_sentiment)
                
                # Print live analysis
                self._print_live_audio_analysis(audio_sentiment)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio analysis worker: {e}")
    
    def _print_live_audio_analysis(self, audio_sentiment):
        """Print live audio sentiment analysis"""
        emotion = audio_sentiment['emotion']
        confidence = audio_sentiment['confidence']
        engagement = audio_sentiment['engagement']
        
        # Create emotion emoji mapping
        emotion_emojis = {
            'excited': '😃',
            'interested': '🤔',
            'neutral': '😐',
            'concerned': '😟',
            'frustrated': '😤'
        }
        
        emoji = emotion_emojis.get(emotion, '😐')
        
        # Print live analysis
        print(f"\n🎤 LIVE AUDIO ANALYSIS: {emoji} {emotion.upper()}")
        print(f"   Confidence: {confidence:.1%} | Engagement: {engagement:.1%}")
        
        # Print voice characteristics
        voice_chars = audio_sentiment['voice_characteristics']
        if voice_chars:
            print(f"   Voice: {voice_chars.get('energy_level', 'unknown')} energy, "
                  f"{voice_chars.get('speech_rate', 'unknown')} speech, "
                  f"{voice_chars.get('voice_clarity', 'unknown')} clarity")
        
        # Update buying probability with audio data
        self._update_buying_probability_with_audio(audio_sentiment)
    
    def _update_buying_probability_with_audio(self, audio_sentiment):
        """Update buying probability with audio sentiment data"""
        # Audio sentiment contribution
        audio_weight = 0.2
        
        # Emotion mapping to buying signals
        emotion_scores = {
            'excited': 0.8,
            'interested': 0.6,
            'neutral': 0.4,
            'concerned': 0.2,
            'frustrated': 0.1
        }
        
        emotion_score = emotion_scores.get(audio_sentiment['emotion'], 0.4)
        confidence_score = audio_sentiment['confidence']
        engagement_score = audio_sentiment['engagement']
        
        # Calculate audio contribution
        audio_contribution = (emotion_score + confidence_score + engagement_score) / 3
        
        # Update buying probability
        if self._buying_probability > 0:
            self._buying_probability = (self._buying_probability * 0.8) + (audio_contribution * 100 * audio_weight)
        else:
            self._buying_probability = audio_contribution * 100 * audio_weight
        
        # Ensure bounds
        self._buying_probability = max(0, min(100, self._buying_probability))
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of user input"""
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity  # -1 to 1
            subjectivity_score = blob.sentiment.subjectivity  # 0 to 1
            
            # Categorize sentiment
            if sentiment_score > 0.3:
                sentiment_category = "Positive"
            elif sentiment_score < -0.3:
                sentiment_category = "Negative"
            else:
                sentiment_category = "Neutral"
            
            return {
                'polarity': sentiment_score,
                'subjectivity': subjectivity_score,
                'category': sentiment_category,
                'timestamp': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'category': 'Neutral',
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def detect_buying_signals(self, text):
        """Detect buying signals in user input"""
        buying_keywords = [
            'interested', 'want to', 'need', 'looking for', 'solution', 'problem',
            'cost', 'price', 'budget', 'timeline', 'implementation', 'demo',
            'trial', 'pilot', 'start', 'begin', 'move forward', 'proceed',
            'schedule', 'meeting', 'call', 'discuss', 'learn more', 'details',
            'features', 'benefits', 'roi', 'return', 'save', 'efficient',
            'automate', 'improve', 'upgrade', 'modernize', 'digital'
        ]
        
        objection_keywords = [
            'expensive', 'costly', 'budget', 'money', 'afford', 'cheaper',
            'alternative', 'competitor', 'compare', 'think about', 'consider',
            'not sure', 'maybe', 'later', 'busy', 'time', 'complicated',
            'difficult', 'change', 'risk', 'concern', 'worry', 'doubt',
            'not ready', 'wait', 'delay', 'review', 'approval', 'committee'
        ]
        
        text_lower = text.lower()
        buying_count = sum(1 for keyword in buying_keywords if keyword in text_lower)
        objection_count = sum(1 for keyword in objection_keywords if keyword in text_lower)
        
        # Calculate signal strength
        total_words = len(text.split())
        buying_signal_strength = (buying_count / max(total_words, 1)) * 100
        objection_signal_strength = (objection_count / max(total_words, 1)) * 100
        
        return {
            'buying_signals': buying_count,
            'objection_signals': objection_count,
            'buying_strength': buying_signal_strength,
            'objection_strength': objection_signal_strength,
            'net_signal': buying_signal_strength - objection_signal_strength
        }
    
    def update_buying_probability(self, user_input, sentiment_data, buying_signals):
        """Update buying probability based on latest input"""
        # Base factors
        sentiment_weight = 0.3
        buying_signal_weight = 0.4
        engagement_weight = 0.3
        
        # Sentiment contribution
        sentiment_contribution = (sentiment_data['polarity'] + 1) * 50  # Convert -1,1 to 0,100
        
        # Buying signal contribution
        buying_contribution = min(buying_signals['buying_strength'], 100)
        objection_penalty = min(buying_signals['objection_strength'], 50)
        net_buying_contribution = max(0, buying_contribution - objection_penalty)
        
        # Engagement contribution (based on conversation length and quality)
        engagement_contribution = min(len(self.conversation_history) * 5, 50)
        
        # Calculate weighted probability
        new_probability = (
            sentiment_contribution * sentiment_weight +
            net_buying_contribution * buying_signal_weight +
            engagement_contribution * engagement_weight
        )
        
        # Smooth the transition (avoid sudden jumps)
        if self._buying_probability > 0:
            self._buying_probability = (self._buying_probability * 0.7) + (new_probability * 0.3)
        else:
            self._buying_probability = new_probability
        
        # Ensure bounds
        self._buying_probability = max(0, min(100, self._buying_probability))
        
        return self._buying_probability
    
    def get_buying_recommendation(self):
        """Get recommendation based on buying probability"""
        if self._buying_probability >= 80:
            return {
                'action': 'CLOSE',
                'confidence': 'High',
                'message': 'Strong buying signals detected. Focus on closing the deal.',
                'probability': self._buying_probability
            }
        elif self._buying_probability >= 60:
            return {
                'action': 'DEMO',
                'confidence': 'Medium-High',
                'message': 'Good buying signals. Schedule a demo or detailed presentation.',
                'probability': self._buying_probability
            }
        elif self._buying_probability >= 40:
            return {
                'action': 'NURTURE',
                'confidence': 'Medium',
                'message': 'Moderate interest. Continue building value and addressing concerns.',
                'probability': self._buying_probability
            }
        elif self._buying_probability >= 20:
            return {
                'action': 'EDUCATE',
                'confidence': 'Low-Medium',
                'message': 'Limited interest. Focus on education and problem identification.',
                'probability': self._buying_probability
            }
        else:
            return {
                'action': 'QUALIFY',
                'confidence': 'Low',
                'message': 'Low buying signals. Re-qualify or focus on relationship building.',
                'probability': self._buying_probability
            }
    
    def print_sentiment_analysis(self):
        """Print current sentiment analysis and buying signals"""
        if not self._sentiment_history:
            return
        
        print("\n" + "=" * 60)
        print("🧠 SENTIMENT ANALYSIS & BUYING SIGNALS")
        print("=" * 60)
        
        # Latest sentiment
        latest_sentiment = self._sentiment_history[-1]
        print(f"📊 Latest Sentiment: {latest_sentiment['category']} ({latest_sentiment['polarity']:.2f})")
        
        # Overall sentiment trend
        if len(self._sentiment_history) > 1:
            avg_sentiment = np.mean([s['polarity'] for s in self._sentiment_history])
            print(f"📈 Average Sentiment: {avg_sentiment:.2f}")
        
        # Buying probability
        print(f"🎯 Buying Probability: {self._buying_probability:.1f}%")
        
        # Recommendation
        recommendation = self.get_buying_recommendation()
        print(f"💡 Recommendation: {recommendation['action']} ({recommendation['confidence']})")
        print(f"📝 Action: {recommendation['message']}")
        
        # Recent signals
        if self._buying_signals:
            print(f"✅ Recent Buying Signals: {len(self._buying_signals)}")
        if self._objection_signals:
            print(f"⚠️ Recent Objections: {len(self._objection_signals)}")
        
        print("=" * 60)
    
    def print_web_search_details(self, client):
        """Print detailed web search information for a client"""
        print("\n" + "=" * 60)
        print("🌐 WEB SEARCH DETAILS")
        print("=" * 60)
        
        # Fetch comprehensive research data
        research_data = self.comprehensive_research.research_company(client)
        
        if research_data:
            print("✅ Real-time data fetched successfully")
            print()
            
            # Company Overview
            company = research_data.get('company_overview', {})
            if company:
                print("📊 COMPANY OVERVIEW:")
                if company.get('company_name'):
                    print(f"   Company: {company['company_name']}")
                if company.get('website'):
                    print(f"   Website: {company['website']}")
                if company.get('industry'):
                    print(f"   Industry: {company['industry']}")
                if company.get('annual_revenue'):
                    print(f"   Annual Revenue: {company['annual_revenue']}")
                if company.get('number_of_employees'):
                    print(f"   Employees: {company['number_of_employees']}")
                if company.get('headquarters_location'):
                    print(f"   Headquarters: {company['headquarters_location']}")
                print()
            
            # Financial & AP Operations
            financial = research_data.get('financial_ap_operations', {})
            if financial:
                print("💰 FINANCIAL & AP OPERATIONS:")
                if financial.get('erp_systems_used'):
                    print(f"   ERP Systems: {', '.join(financial['erp_systems_used'])}")
                if financial.get('ap_automation_maturity'):
                    print(f"   AP Automation Maturity: {financial['ap_automation_maturity']}")
                if financial.get('known_pain_points'):
                    print("   Known Pain Points:")
                    for point in financial['known_pain_points'][:3]:
                        print(f"     • {point}")
                print()
            
            # Recent News & Trigger Events
            news = research_data.get('recent_news_trigger_events', {})
            if news:
                print("📰 RECENT NEWS & TRIGGER EVENTS:")
                if news.get('trigger_events'):
                    for event in news['trigger_events'][:2]:
                        print(f"   • {event}")
                if news.get('funding_ma_activity'):
                    for activity in news['funding_ma_activity'][:1]:
                        print(f"   • {activity}")
                print()
            
            # Key Decision Makers
            teams = research_data.get('key_teams_decision_makers', {})
            if teams:
                print("👥 KEY DECISION MAKERS:")
                if teams.get('key_decision_makers'):
                    for person in teams['key_decision_makers'][:2]:
                        print(f"   • {person.get('name', 'N/A')} - {person.get('title', 'N/A')}")
                        if person.get('background_highlights'):
                            print(f"     Background: {person['background_highlights'][:100]}...")
                print()
            
            # Messaging Angle
            messaging = research_data.get('messaging_angle', {})
            if messaging:
                print("🎯 SALES MESSAGING ANGLE:")
                if messaging.get('why_hyprbots_can_help'):
                    print(f"   Why Hyprbots Can Help: {messaging['why_hyprbots_can_help']}")
                if messaging.get('key_value_propositions'):
                    print("   Key Value Propositions:")
                    for prop in messaging['key_value_propositions'][:2]:
                        print(f"     • {prop}")
                print()
        else:
            print("❌ No web data available")
        print("=" * 60)
        
    def print_web_search_details_from_data(self, research_data):
        """Print detailed web search information from existing research data"""
        print("\n" + "=" * 60)
        print("🌐 WEB SEARCH DETAILS")
        print("=" * 60)
        
        if research_data:
            print("✅ Real-time data fetched successfully")
            print()
            
            # Company Overview
            company = research_data.get('company_overview', {})
            if company:
                print("📊 COMPANY OVERVIEW:")
                if company.get('company_name'):
                    print(f"   Company: {company['company_name']}")
                if company.get('website'):
                    print(f"   Website: {company['website']}")
                if company.get('industry'):
                    print(f"   Industry: {company['industry']}")
                if company.get('annual_revenue'):
                    print(f"   Annual Revenue: {company['annual_revenue']}")
                if company.get('number_of_employees'):
                    print(f"   Employees: {company['number_of_employees']}")
                if company.get('headquarters_location'):
                    print(f"   Headquarters: {company['headquarters_location']}")
                print()
            
            # Financial & AP Operations
            financial = research_data.get('financial_ap_operations', {})
            if financial:
                print("💰 FINANCIAL & AP OPERATIONS:")
                if financial.get('erp_systems_used'):
                    print(f"   ERP Systems: {', '.join(financial['erp_systems_used'])}")
                if financial.get('ap_automation_maturity'):
                    print(f"   AP Automation Maturity: {financial['ap_automation_maturity']}")
                if financial.get('known_pain_points'):
                    print("   Known Pain Points:")
                    for point in financial['known_pain_points'][:3]:
                        print(f"     • {point}")
                print()
            
            # Recent News & Trigger Events
            news = research_data.get('recent_news_trigger_events', {})
            if news:
                print("📰 RECENT NEWS & TRIGGER EVENTS:")
                if news.get('trigger_events'):
                    for event in news['trigger_events'][:2]:
                        print(f"   • {event}")
                if news.get('funding_ma_activity'):
                    for activity in news['funding_ma_activity'][:1]:
                        print(f"   • {activity}")
                print()
            
            # Key Decision Makers
            teams = research_data.get('key_teams_decision_makers', {})
            if teams:
                print("👥 KEY DECISION MAKERS:")
                if teams.get('key_decision_makers'):
                    for person in teams['key_decision_makers'][:2]:
                        print(f"   • {person.get('name', 'N/A')} - {person.get('title', 'N/A')}")
                        if person.get('background_highlights'):
                            print(f"     Background: {person['background_highlights'][:100]}...")
                print()
            
            # Messaging Angle
            messaging = research_data.get('messaging_angle', {})
            if messaging:
                print("🎯 SALES MESSAGING ANGLE:")
                if messaging.get('why_hyprbots_can_help'):
                    print(f"   Why Hyprbots Can Help: {messaging['why_hyprbots_can_help']}")
                if messaging.get('key_value_propositions'):
                    print("   Key Value Propositions:")
                    for prop in messaging['key_value_propositions'][:2]:
                        print(f"     • {prop}")
                print()
        else:
            print("❌ No web data available")
        print("=" * 60)

    def format_web_data_for_context(self, client, research_data):
        """Format research data for conversation context"""
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
            for point in financial['known_pain_points'][:3]:  # Limit to 3 points
                web_context.append(f"  • {point}")
        
        # Recent news and trigger events
        news = research_data.get('recent_news_trigger_events', {})
        if news.get('trigger_events'):
            web_context.append("Recent Developments:")
            for event in news['trigger_events'][:2]:  # Limit to 2 events
                web_context.append(f"  • {event}")
        
        # Messaging angle for sales conversation
        messaging = research_data.get('messaging_angle', {})
        if messaging.get('key_value_propositions'):
            web_context.append("Value Propositions:")
            for prop in messaging['key_value_propositions'][:2]:  # Limit to 2 props
                web_context.append(f"  • {prop}")
        
        return "\n".join(web_context)

    def fetch_web_data_for_client(self, client):
        """Get web data for conversation context (uses stored data)"""
        # Return the stored web context if available
        if self._web_context:
            return self._web_context
        else:
            # Fallback: fetch data if not already available
            print(f"🌐 Fetching web data for {client.get('Full Name')} at {client.get('Company')}...")
            research_data = self.comprehensive_research.research_company(client)
            return self.format_web_data_for_context(client, research_data)
    
    def scrape_company_website(self, client):
        """Get website data for conversation context (uses stored data)"""
        # Return the stored website data if available
        if self._website_data:
            return self._website_data
        else:
            # Fallback: scrape data if not already available
            try:
                # Use domain from client data if available
                domain = client.get('Domain', '')
                if not domain:
                    print("❌ No domain found in client data")
                    return None
                
                # Construct website URL from domain
                if not domain.startswith(('http://', 'https://')):
                    website_url = f"https://www.{domain}"
                else:
                    website_url = domain
                
                print(f"🌐 Scraping company website: {website_url}")
                
                # Use the improved scraper with caching
                scraped_data = self.website_scraper.scrape_company_website(website_url)
                
                if scraped_data and 'error' not in scraped_data:
                    company_name = scraped_data.get('company_overview', {}).get('company_name', client.get('Company', 'Unknown'))
                    print(f"✅ Successfully scraped website data for {company_name}")
                    
                    # Check if data was from cache
                    if scraped_data.get('cached_at'):
                        print(f"📦 Using cached data (cached: {scraped_data['cached_at']})")
                    else:
                        print(f"🔄 Fresh data scraped")
                    
                    return scraped_data
                else:
                    error_msg = scraped_data.get('error', 'Unknown error') if scraped_data else 'No data returned'
                    print(f"❌ Failed to scrape website: {error_msg}")
                    return None
                    
            except Exception as e:
                print(f"❌ Error scraping company website: {e}")
                return None
    
    def pre_fetch_website_data(self, client):
        """Pre-fetch website data during client setup to avoid delays during conversation"""
        try:
            domain = client.get('Domain', '')
            if not domain:
                return None
            
            # Construct website URL
            if not domain.startswith(('http://', 'https://')):
                website_url = f"https://www.{domain}"
            else:
                website_url = domain
            
            # Pre-fetch data in background
            print(f"🔄 Pre-fetching website data for {domain}...")
            scraped_data = self.website_scraper.scrape_company_website(website_url)
            
            if scraped_data and 'error' not in scraped_data:
                # Store for later use
                self.current_website_data = scraped_data
                return scraped_data
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Pre-fetch failed: {e}")
            return None
    
    def format_website_data_for_context(self, scraped_data):
        """Format scraped website data for conversation context"""
        if not scraped_data:
            return ""
        
        context_parts = []
        
        # Company information
        company_overview = scraped_data.get('company_overview', {})
        if company_overview.get('company_name'):
            context_parts.append(f"Company: {company_overview['company_name']}")
        
        if company_overview.get('tagline'):
            context_parts.append(f"Tagline: {company_overview['tagline']}")
        
        if company_overview.get('description'):
            context_parts.append(f"Description: {company_overview['description']}")
        
        if company_overview.get('industry'):
            context_parts.append(f"Industry: {company_overview['industry']}")
        
        if company_overview.get('business_model'):
            context_parts.append(f"Business Model: {company_overview['business_model']}")
        
        # Contact information
        contact_info = scraped_data.get('contact_information', {})
        if contact_info.get('email_addresses'):
            context_parts.append(f"Contact Email: {', '.join(contact_info['email_addresses'])}")
        if contact_info.get('phone_numbers'):
            context_parts.append(f"Contact Phone: {', '.join(contact_info['phone_numbers'])}")
        
        # Products and services
        products_services = scraped_data.get('products_services', {})
        if products_services.get('primary_products'):
            context_parts.append(f"Primary Products: {', '.join(products_services['primary_products'][:3])}")
        if products_services.get('services_offered'):
            context_parts.append(f"Services: {', '.join(products_services['services_offered'][:3])}")
        
        # Technologies
        tech_stack = scraped_data.get('technology_stack', {})
        if tech_stack.get('erp_systems'):
            context_parts.append(f"ERP Systems: {', '.join(tech_stack['erp_systems'])}")
        if tech_stack.get('automation_tools'):
            context_parts.append(f"Automation Tools: {', '.join(tech_stack['automation_tools'])}")
        
        # Social media
        social_media = scraped_data.get('social_media_presence', {})
        active_platforms = []
        for platform, url in social_media.items():
            if url and platform != 'social_media_activity':
                active_platforms.append(platform.title())
        if active_platforms:
            context_parts.append(f"Social Media: {', '.join(active_platforms)}")
        
        # Business operations
        operations = scraped_data.get('business_operations', {})
        if operations.get('automation_mentions'):
            context_parts.append(f"Automation Focus: {', '.join(operations['automation_mentions'])}")
        if operations.get('digital_transformation'):
            context_parts.append(f"Digital Initiatives: {', '.join(operations['digital_transformation'])}")
        
        return "\n".join(context_parts)
    
    def print_website_scraping_details(self, client):
        """Print detailed website scraping information for a client"""
        print("\n" + "=" * 60)
        print("🌐 WEBSITE SCRAPING DETAILS")
        print("=" * 60)
        
        # Get domain from client
        domain = client.get('Domain', '')
        if not domain:
            print("❌ No domain found in client data")
            return
        
        print(f"🌐 Scraping website: {domain}")
        
        # Scrape website data
        scraped_data = self.scrape_company_website(client)
        
        if scraped_data and 'error' not in scraped_data:
            print("✅ Website scraping completed successfully")
            print()
            
            # Section 1: Company Overview
            company_overview = scraped_data.get('company_overview', {})
            if company_overview:
                print("📊 SECTION 1: COMPANY OVERVIEW")
                print("-" * 40)
                if company_overview.get('company_name'):
                    print(f"   Company Name: {company_overview['company_name']}")
                if company_overview.get('tagline'):
                    print(f"   Tagline: {company_overview['tagline']}")
                if company_overview.get('industry'):
                    print(f"   Industry: {company_overview['industry']}")
                if company_overview.get('sub_industry'):
                    print(f"   Sub-Industry: {company_overview['sub_industry']}")
                if company_overview.get('company_type'):
                    print(f"   Company Type: {company_overview['company_type']}")
                if company_overview.get('headquarters_location'):
                    print(f"   Headquarters: {company_overview['headquarters_location']}")
                if company_overview.get('number_of_employees'):
                    print(f"   Employees: {company_overview['number_of_employees']}")
                if company_overview.get('annual_revenue'):
                    print(f"   Annual Revenue: {company_overview['annual_revenue']}")
                if company_overview.get('year_founded'):
                    print(f"   Founded: {company_overview['year_founded']}")
                if company_overview.get('business_model'):
                    print(f"   Business Model: {company_overview['business_model']}")
                if company_overview.get('revenue_model'):
                    print(f"   Revenue Model: {company_overview['revenue_model']}")
                print()
            
            # Section 2: Contact Information
            contact_info = scraped_data.get('contact_information', {})
            if contact_info:
                print("📞 SECTION 2: CONTACT INFORMATION")
                print("-" * 40)
                if contact_info.get('email_addresses'):
                    print(f"   Email Addresses: {', '.join(contact_info['email_addresses'])}")
                if contact_info.get('phone_numbers'):
                    print(f"   Phone Numbers: {', '.join(contact_info['phone_numbers'])}")
                if contact_info.get('contact_page'):
                    print(f"   Contact Page: {contact_info['contact_page']}")
                if contact_info.get('office_locations'):
                    print(f"   Office Locations: {', '.join(contact_info['office_locations'][:3])}")
                print()
            
            # Section 3: Products & Services
            products_services = scraped_data.get('products_services', {})
            if products_services:
                print("📦 SECTION 3: PRODUCTS & SERVICES")
                print("-" * 40)
                if products_services.get('primary_products'):
                    print("   Primary Products:")
                    for i, product in enumerate(products_services['primary_products'][:3], 1):
                        print(f"     {i}. {product}")
                if products_services.get('services_offered'):
                    print("   Services Offered:")
                    for i, service in enumerate(products_services['services_offered'][:3], 1):
                        print(f"     {i}. {service}")
                if products_services.get('solutions_by_industry'):
                    print(f"   Industry Solutions: {', '.join(products_services['solutions_by_industry'][:3])}")
                print()
            
            # Section 4: Team & Leadership
            team_leadership = scraped_data.get('team_leadership', {})
            if team_leadership:
                print("👥 SECTION 4: TEAM & LEADERSHIP")
                print("-" * 40)
                if team_leadership.get('key_decision_makers'):
                    print("   Key Decision Makers:")
                    for i, member in enumerate(team_leadership['key_decision_makers'][:3], 1):
                        name = member.get('name', 'N/A')
                        title = member.get('title', 'N/A')
                        print(f"     {i}. {name} - {title}")
                if team_leadership.get('finance_ap_team'):
                    print("   Finance/AP Team:")
                    for i, member in enumerate(team_leadership['finance_ap_team'][:3], 1):
                        name = member.get('name', 'N/A')
                        title = member.get('title', 'N/A')
                        print(f"     {i}. {name} - {title}")
                if team_leadership.get('team_size_estimate'):
                    print(f"   Estimated Team Size: {team_leadership['team_size_estimate']}")
                print()
            
            # Section 5: Technology Stack
            tech_stack = scraped_data.get('technology_stack', {})
            if tech_stack:
                print("💻 SECTION 5: TECHNOLOGY STACK")
                print("-" * 40)
                if tech_stack.get('erp_systems'):
                    print(f"   ERP Systems: {', '.join(tech_stack['erp_systems'])}")
                if tech_stack.get('automation_tools'):
                    print(f"   Automation Tools: {', '.join(tech_stack['automation_tools'])}")
                if tech_stack.get('cms_platforms'):
                    print(f"   CMS Platforms: {', '.join(tech_stack['cms_platforms'])}")
                if tech_stack.get('analytics_tools'):
                    print(f"   Analytics Tools: {', '.join(tech_stack['analytics_tools'])}")
                print()
            
            # Section 6: Social Media Presence
            social_media = scraped_data.get('social_media_presence', {})
            if social_media:
                print("🔗 SECTION 6: SOCIAL MEDIA PRESENCE")
                print("-" * 40)
                active_platforms = []
                for platform, url in social_media.items():
                    if url and platform != 'social_media_activity':
                        active_platforms.append(platform.title())
                if active_platforms:
                    print(f"   Active Platforms: {', '.join(active_platforms)}")
                print()
            
            # Section 7: Recent News & Articles
            news_articles = scraped_data.get('recent_news_articles', {})
            if news_articles:
                print("📰 SECTION 7: RECENT NEWS & ARTICLES")
                print("-" * 40)
                if news_articles.get('recent_articles'):
                    print("   Recent Articles:")
                    for i, article in enumerate(news_articles['recent_articles'][:2], 1):
                        title = article.get('title', 'N/A')
                        date = article.get('date', 'N/A')
                        print(f"     {i}. {title} ({date})")
                if news_articles.get('company_updates'):
                    print("   Company Updates:")
                    for i, update in enumerate(news_articles['company_updates'][:2], 1):
                        print(f"     {i}. {update}")
                print()
            
            # Section 8: Financial Indicators
            financial = scraped_data.get('financial_indicators', {})
            if financial:
                print("💰 SECTION 8: FINANCIAL INDICATORS")
                print("-" * 40)
                if financial.get('revenue_mentions'):
                    print(f"   Revenue Mentions: {', '.join(financial['revenue_mentions'][:3])}")
                if financial.get('funding_mentions'):
                    print(f"   Funding Mentions: {', '.join(financial['funding_mentions'][:3])}")
                if financial.get('growth_indicators'):
                    print(f"   Growth Indicators: {', '.join(financial['growth_indicators'][:3])}")
                print()
            
            # Section 9: Business Operations
            operations = scraped_data.get('business_operations', {})
            if operations:
                print("⚙️ SECTION 9: BUSINESS OPERATIONS")
                print("-" * 40)
                if operations.get('automation_mentions'):
                    print(f"   Automation Mentions: {', '.join(operations['automation_mentions'])}")
                if operations.get('digital_transformation'):
                    print(f"   Digital Transformation: {', '.join(operations['digital_transformation'])}")
                if operations.get('process_improvements'):
                    print(f"   Process Improvements: {', '.join(operations['process_improvements'])}")
                print()
            
            # Scraping Method
            if scraped_data.get('method'):
                print(f"🔧 Scraping Method: {scraped_data['method']}")
            
        else:
            print("❌ Website scraping failed")
            if scraped_data and 'error' in scraped_data:
                print(f"   Error: {scraped_data['error']}")
            print("   Note: This is common for websites with anti-bot protection")
        
        print("=" * 60)

    def listen(self):
        """Listen for user input with real-time audio analysis"""
        with sr.Microphone() as source:
            # Adjust microphone for ambient noise with longer duration
            print("🎤 Adjusting microphone for ambient noise...")
            self.listener.adjust_for_ambient_noise(source, duration=SPEECH_RECOGNITION['ambient_noise_duration'])
            
            # Set optimized recognition parameters from config
            self.listener.energy_threshold = SPEECH_RECOGNITION['energy_threshold']
            self.listener.dynamic_energy_threshold = True
            self.listener.pause_threshold = SPEECH_RECOGNITION['pause_threshold']
            
            print("🎤 Listening... (speak clearly and naturally)")
            
            # More patient listening settings
            audio = self.listener.listen(
                source, 
                timeout=SPEECH_RECOGNITION['timeout'],  # Wait up to 15 seconds for speech to start
                phrase_time_limit=SPEECH_RECOGNITION['phrase_time_limit'],  # Allow up to 20 seconds for complete phrases
                snowboy_configuration=None
            )
        
        try:
            # Get audio data for real-time analysis
            audio_data = audio.get_wav_data()
            
            # Perform real-time audio sentiment analysis
            if self._audio_analysis_active:
                audio_sentiment = self.analyze_audio_sentiment(audio_data)
                self._audio_sentiment_history.append(audio_sentiment)
                self._print_live_audio_analysis(audio_sentiment)
            
            # Try Google Speech Recognition first
            text = self.listener.recognize_google(audio)
            print(f"👂 You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("⏰ No speech detected within timeout period")
            return ""
        except sr.UnknownValueError:
            print("❌ Could not understand the audio - please speak more clearly")
            return ""
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return ""
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""

    def speak(self, text):
        """Speak text with interruption capability"""
        print(f"🤖 Assistant: {text}")
        
        # Start speaking
        self.tts.say(text)
        
        # Use a non-blocking approach to allow interruption
        try:
            # Start the speech in a way that can be interrupted
            self.tts.runAndWait()
        except KeyboardInterrupt:
            # Stop speaking if interrupted
            self.tts.stop()
            print("🔇 Speech interrupted")
    
    def speak_interruptible(self, text):
        """Speak text (simplified without interruption)"""
        print(f"🤖 Assistant: {text}")
        
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            print(f"❌ Error speaking: {e}")

    def answer(self, query):
        # First, try to identify client and get enhanced context
        enhanced_query, client = self.client_rag.enhance_conversation_with_client_context(query)
        
        # Update current client if identified
        if client and not self.current_client:
            self.current_client = client
            print(f"👤 New client identified: {client.get('Full Name')} from {client.get('Company')}")
        
        # Get RAG context from documents
        retrieved_docs = self.rag.retrieve(query)
        context_parts = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = doc.get('content', str(doc))
            else:
                content = str(doc)
            context_parts.append(content)
        
        # Get conversation history context
        conversation_context = self.get_conversation_context()
        
        # Combine all context
        rag_context = "\n".join(context_parts)
        
        # Get web data for current client if available (use stored data)
        web_data = ""
        website_data = ""
        if self.current_client:
            # Use stored web data (already fetched during initialization)
            web_data = self._web_context if self._web_context else ""
            
            # Use stored website data (already fetched during initialization)
            website_data = self._website_context if self._website_context else ""
        
        # Create enhanced prompt with client context and web data
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
        
        # Include web data in context if available
        web_context = f"\nChatGPT Web Research Data:\n{web_data}" if web_data else ""
        website_context = f"\nWebsite Scraping Data:\n{website_data}" if website_data else ""
        
        # Combine all web data
        combined_web_data = f"{web_context}{website_context}"
        
        if conversation_context:
            prompt = f"""{self.system_instructions}

{client_info}
Context from documents: {rag_context}{combined_web_data}

Previous conversation:
{conversation_context}

Current question: {enhanced_query}

Please provide a personalized response that takes into account the client's role, company, web research data, website scraping data, and our conversation history."""
        else:
            prompt = f"""{self.system_instructions}

{client_info}
Context: {rag_context}{combined_web_data}

Question: {enhanced_query}

Please provide a personalized response based on the context, web research, website data, and your role as a sales executive."""
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            # Save client interaction if we have a current client
            if self.current_client:
                interaction = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'user_input': query,
                    'enhanced_query': enhanced_query,
                    'response': response['message']['content']
                }
                self.client_rag.save_client_interaction(self.current_client, interaction)
            
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "I'm sorry, I couldn't process your request right now."

    def parse_schedule_request(self, user_input):
        """Parse user input to extract scheduling information"""
        user_input_lower = user_input.lower()
        
        # Default values
        event_title = "Sales Call"
        duration = 30
        start_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        
        # Extract event title
        if "call" in user_input_lower:
            if "demo" in user_input_lower:
                event_title = "Product Demo Call"
            elif "sales" in user_input_lower:
                event_title = "Sales Call"
            elif "meeting" in user_input_lower:
                event_title = "Sales Meeting"
            else:
                event_title = "Call"
        
        # Extract duration
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
        
        return event_title, start_time, duration

    def schedule_event(self, user_input):
        """Create a calendar event based on user input"""
        try:
            event_title, start_time, duration = self.parse_schedule_request(user_input)
            
            # Create the calendar event
            event_link = create_event(
                summary=event_title,
                start_time=start_time,
                duration_minutes=duration,
                description=f"Automatically scheduled based on: {user_input}"
            )
            
            # Format time for speech
            time_str = start_time.strftime("%I:%M %p on %B %d")
            
            return f"I've scheduled a {duration}-minute {event_title} for {time_str}. The event has been added to your calendar."
            
        except Exception as e:
            print(f"Error creating calendar event: {e}")
            return "I'm sorry, I couldn't create the calendar event. Please try again."

    def add_to_calendar(self, info):
        with open(CALENDAR_PATH, "a") as f:
            f.write(info + "\n")
    
    def save_conversation(self):
        """Save conversation history to a JSON file"""
        conversation_file = f"conversations/conversation_{self.session_id}.json"
        os.makedirs("conversations", exist_ok=True)
        
        conversation_data = {
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "conversation": self.conversation_history
        }
        
        with open(conversation_file, "w") as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"💾 Conversation saved to {conversation_file}")
    
    def load_conversation(self, session_id=None):
        """Load conversation history from a JSON file"""
        if session_id is None:
            session_id = self.session_id
        
        conversation_file = f"conversations/conversation_{session_id}.json"
        
        if os.path.exists(conversation_file):
            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)
                self.conversation_history = conversation_data.get("conversation", [])
                print(f"📂 Loaded conversation with {len(self.conversation_history)} exchanges")
                return True
        else:
            print(f"📂 No previous conversation found for session {session_id}")
            return False
    
    def add_to_history(self, user_input, response):
        """Add a conversation exchange to history"""
        exchange = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user": user_input,
            "assistant": response
        }
        self.conversation_history.append(exchange)
        
        # Auto-save every 5 exchanges
        if len(self.conversation_history) % 5 == 0:
            self.save_conversation()
    
    def get_conversation_context(self, max_exchanges=10):
        """Get recent conversation context for the AI"""
        recent_exchanges = self.conversation_history[-max_exchanges:] if self.conversation_history else []
        
        context = []
        for exchange in recent_exchanges:
            context.append(f"User: {exchange['user']}")
            context.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(context)
    

    
    def apply_voice_settings(self):
        """Apply voice settings from config"""
        voices = self.tts.getProperty('voices')
        
        # Find the specified voice from config
        target_voice = None
        for voice in voices:
            if VOICE_SETTINGS['voice_name'].lower() in voice.name.lower():
                target_voice = voice
                break
        
        if target_voice:
            self.tts.setProperty('voice', target_voice.id)
            print(f"🎤 Using {VOICE_SETTINGS['voice_name']} voice: {target_voice.name}")
        else:
            print(f"🎤 {VOICE_SETTINGS['voice_name']} voice not found, using default voice")
        
        # Apply settings from config
        self.tts.setProperty('rate', VOICE_SETTINGS['rate'])
        self.tts.setProperty('volume', VOICE_SETTINGS['volume'])
        
        print(f"🎤 Applied {VOICE_SETTINGS['voice_name']} voice settings:")
        print(f"   Rate: {VOICE_SETTINGS['rate']} WPM")
        print(f"   Volume: {VOICE_SETTINGS['volume']}")
        
        # Update voice settings
        self.voice_settings.update({
            'rate': VOICE_SETTINGS['rate'],
            'volume': VOICE_SETTINGS['volume'],
            'voice_id': target_voice.id if target_voice else self.voice_settings.get('voice_id')
        })

    def setup_client_session(self, client):
        """Setup client session with all data fetching and display - OPTIMIZED"""
        self.current_client = client
        
        # 1. Print client details
        print("\n" + "=" * 60)
        print("👤 CLIENT DETAILS")
        print("=" * 60)
        print(f"Name: {client.get('Full Name', 'N/A')}")
        print(f"Company: {client.get('Company', 'N/A')}")
        print(f"Phone: {client.get('Phone', 'N/A')}")
        print(f"Email: {client.get('Email', 'N/A')}")
        print(f"Title: {client.get('Title', 'N/A')}")
        print(f"Domain: {client.get('Domain', 'N/A')}")
        print(f"Industry: {client.get('Industry', 'N/A')}")
        print(f"Company Size: {client.get('Company Size', 'N/A')}")
        print(f"Revenue: {client.get('Revenue', 'N/A')}")
        print("=" * 60)
        
        # 2. Fetch and store web data (ONCE during initialization)
        print("🌐 Fetching comprehensive research data...")
        self._web_data = self.comprehensive_research.research_company(client)
        self.print_web_search_details_from_data(self._web_data)
        # Store formatted web context
        self._web_context = self.format_web_data_for_context(client, self._web_data)
        
        # 3. Fetch and store website data (ONCE during initialization)
        print("🌐 Pre-fetching website data...")
        self._website_data = self.pre_fetch_website_data(client)
        if self._website_data:
            self.print_website_scraping_details_from_data(self._website_data)
            # Store formatted website context
            self._website_context = self.format_website_data_for_context(self._website_data)
        else:
            # Fallback to original method if pre-fetch fails
            self.print_website_scraping_details(client)
        
        # 4. Print conversation details
        print("\n" + "=" * 60)
        print("💬 CONVERSATION DETAILS")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Voice: {VOICE_SETTINGS['voice_name']}")
        print(f"Speech Rate: {VOICE_SETTINGS['rate']} WPM")
        print(f"Volume: {int(VOICE_SETTINGS['volume'] * 100)}%")
        print(f"AI Model: {self.model_name}")
        print(f"RAG Documents: {len(self.rag.documents)} loaded")
        print(f"Client Database: {len(self.client_rag.clients)} clients loaded")
        print("=" * 60)
        
        welcome_message = f"Hi {client.get('Full Name')}, this is Sunny from Hyprbots. I'm calling because companies like Carter's Inc. are losing thousands of dollars monthly on manual invoice processing. We've helped similar companies cut that cost by 90%. You're probably spending 10+ hours a week on this - am I right?"
        print(f"\n🤖 ASSISTANT: {welcome_message}")
        print("=" * 60)
        self.speak_interruptible(welcome_message)

    def run(self):
        # Initialize RAG systems
        if not os.path.exists(INDEX_PATH):
            print("⚙️ Building RAG index...")
            with open(DOC_PATH) as f:
                docs = json.load(f)
            self.rag.build_index(docs)
        else:
            self.rag.load_index()
        
        # Initialize client RAG system
        print("👥 Initializing client database...")
        if not self.client_rag.load_clients():
            print("❌ Failed to load clients")
            return
        
        if not os.path.exists(self.client_rag.client_index_path):
            print("🔨 Building client search index...")
            self.client_rag.build_client_index()
        else:
            self.client_rag.load_client_index()

        # Handle client identification
        if self.phone_number:
            # Use phone number from command line argument
            print(f"📞 Using provided phone number: {self.phone_number}")
            client = self.client_rag.search_client_by_phone(self.phone_number)
            if client:
                self.setup_client_session(client)
            else:
                self.speak_interruptible("I couldn't find a client with that phone number. Let's continue with general assistance.")
                print("❌ No client found with provided phone number")
        else:
            # Ask for phone number if not provided
            self.speak_interruptible("Please provide the client's phone number to start a personalized conversation.")
            phone_input = self.listen()
            
            if phone_input:
                # Search for client by phone number
                client = self.client_rag.search_client_by_phone(phone_input)
                if client:
                    self.setup_client_session(client)
                else:
                    self.speak_interruptible("I couldn't find a client with that phone number. Let's continue with general assistance.")
                    print("❌ No client found with provided phone number")
            else:
                self.speak_interruptible("No phone number provided. Let's continue with general assistance.")
        
        # Main conversation loop
        while True:
            user_input = self.listen()
            if not user_input:
                continue
            if "exit" in user_input.lower() or "bye" in user_input.lower():
                if self.current_client:
                    self.speak_interruptible(f"Thank you {self.current_client.get('Full Name')}! Have a great day!")
                else:
                    self.speak_interruptible("Thank you for your patience! Have a Good Day!")
                # Save conversation before exiting
                self.save_conversation()
                break
            
            # Print user input
            print(f"\n👤 USER: {user_input}")
            print("-" * 50)
            
            # Perform sentiment analysis and buying signal detection
            sentiment_data = self.analyze_sentiment(user_input)
            buying_signals = self.detect_buying_signals(user_input)
            
            # Update buying probability
            self.update_buying_probability(user_input, sentiment_data, buying_signals)
            
            # Store sentiment and signals
            self._sentiment_history.append(sentiment_data)
            if buying_signals['buying_signals'] > 0:
                self._buying_signals.append({
                    'input': user_input,
                    'signals': buying_signals,
                    'timestamp': datetime.datetime.now().isoformat()
                })
            if buying_signals['objection_signals'] > 0:
                self._objection_signals.append({
                    'input': user_input,
                    'objections': buying_signals,
                    'timestamp': datetime.datetime.now().isoformat()
                })
            
            # Print sentiment analysis (every 3rd interaction)
            if len(self.conversation_history) % 3 == 0:
                self.print_sentiment_analysis()
            
            # Check for scheduling requests first
            if "schedule" in user_input.lower() or "book" in user_input.lower() or "appointment" in user_input.lower():
                response = self.schedule_event(user_input)
                print(f"🤖 ASSISTANT: {response}")
                print("=" * 50)
                self.speak_interruptible(response)
                # Also log to text file as backup
                self.add_to_calendar(f"Calendar event created: {user_input}")
                # Save to conversation history
                self.add_to_history(user_input, response)
            else:
                answer = self.answer(user_input)
                print(f"🤖 ASSISTANT: {answer}")
                print("=" * 50)
                self.speak_interruptible(answer)
                # Save to conversation history
                self.add_to_history(user_input, answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Executive AI Assistant")
    parser.add_argument("--phone", type=str, help="Client's phone number for personalized assistance")
    args = parser.parse_args()

    agent = VoiceAgent(phone_number=args.phone)
    agent.run()
