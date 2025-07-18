"""
Sales Voice Agent with Twilio Voice Synthesis Option
Modified version of main.py that supports both pyttsx3 and Twilio TTS
"""

import os
import json
import speech_recognition as sr
import faiss
import ollama
from sentence_transformers import SentenceTransformer
import re
import datetime
from calendar_helper import create_event
from twilio_voice import create_twilio_voice, TwilioVoiceManager
import pyttsx3
from textblob import TextBlob
import numpy as np
import librosa
import threading
import time
from agent_config import system_instructions

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

# Voice configuration
VOICE_TYPE = os.getenv("VOICE_TYPE", "pyttsx3")  # "pyttsx3" or "twilio"
VOICE_SETTINGS = {
    'rate': 180,  # Words per minute
    'volume': 0.85  # Volume level (0.0 to 1.0)
}

class RAG:
    def __init__(self):
        self.index = None
        self.documents = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def build_index(self, docs):
        """Build FAISS index from documents"""
        self.documents = docs
        embeddings = self.embedder.encode(docs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        print(f"✅ Built FAISS index with {len(docs)} documents")
    
    def load_index(self):
        """Load existing index if available"""
        try:
            if os.path.exists('faiss_index'):
                self.index = faiss.read_index('faiss_index')
                with open('documents.json', 'r') as f:
                    self.documents = json.load(f)
                print(f"✅ Loaded existing FAISS index with {len(self.documents)} documents")
                return True
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
        return False
    
    def retrieve(self, query, top_k=3):
        """Retrieve relevant documents"""
        if self.index is None:
            return []
        
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results

class ClientRAG:
    """RAG system for client data and interactions"""
    
    def __init__(self):
        self.clients = []
        self.index = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_clients()
    
    def load_clients(self):
        """Load clients from JSON file"""
        try:
            with open('clients.json', 'r') as f:
                self.clients = json.load(f)
            print(f"✅ Loaded {len(self.clients)} clients")
            self.build_index()
        except FileNotFoundError:
            print("⚠️  clients.json not found")
            self.clients = []
        except Exception as e:
            print(f"❌ Error loading clients: {e}")
            self.clients = []
    
    def build_index(self):
        """Build FAISS index for client search"""
        if not self.clients:
            return
        
        # Create searchable text for each client
        client_texts = []
        for client in self.clients:
            text = f"{client.get('Full Name', '')} {client.get('Company', '')} {client.get('Title', '')} {client.get('Phone', '')} {client.get('Email', '')}"
            client_texts.append(text)
        
        # Build index
        embeddings = self.embedder.encode(client_texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        print(f"✅ Built client search index with {len(self.clients)} clients")
    
    def search_by_phone(self, phone_number):
        """Search for client by phone number"""
        for client in self.clients:
            if client.get('Phone') == phone_number:
                return client
        return None
    
    def enhance_conversation_with_client_context(self, query):
        """Enhance conversation with client context"""
        if not self.index:
            return query, None
        
        # Search for relevant clients
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), 1)
        
        if indices[0][0] < len(self.clients):
            client = self.clients[indices[0][0]]
            enhanced_query = f"Client: {client.get('Full Name')} from {client.get('Company')}. Query: {query}"
            return enhanced_query, client
        
        return query, None
    
    def save_client_interaction(self, client, interaction):
        """Save client interaction to file"""
        try:
            os.makedirs('client_interactions', exist_ok=True)
            filename = f"client_interactions/{client.get('Full Name', 'unknown').replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing interactions or create new
            interactions = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    interactions = json.load(f)
            
            interactions.append(interaction)
            
            with open(filename, 'w') as f:
                json.dump(interactions, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving client interaction: {e}")

class VoiceAgent:
    def __init__(self, phone_number=None):
        self.model_name = "llama3.2:3b"
        
        # Initialize voice synthesis based on configuration
        if VOICE_TYPE.lower() == "twilio":
            self.voice_manager = create_twilio_voice()
            self.tts = self.voice_manager.tts if self.voice_manager.is_enabled else None
            print("🎤 Using Twilio voice synthesis")
        else:
            self.voice_manager = None
            self.tts = pyttsx3.init()
            print("🎤 Using pyttsx3 voice synthesis")
        
        self.listener = sr.Recognizer()
        
        # Initialize RAG systems
        self.rag = RAG()
        self.client_rag = ClientRAG()
        
        # Load documents and build index
        self.load_documents()
        
        # Conversation management
        self.conversation_history = []
        self.current_client = None
        
        # Sentiment analysis
        self.sentiment_history = []
        self._buying_probability = 0.0
        
        # Audio analysis
        self._audio_analysis_active = False
        self._audio_sentiment_history = []
        
        # Web data storage
        self._web_context = None
        self._website_context = None
        
        # Setup voice
        self.setup_voice()
        
        # Initialize client if phone number provided
        if phone_number:
            self.current_client = self.client_rag.search_by_phone(phone_number)
            if self.current_client:
                print(f"👤 Client identified: {self.current_client.get('Full Name')} from {self.current_client.get('Company')}")
                self.pre_fetch_web_data()
    
    def setup_voice(self):
        """Setup voice synthesis"""
        if VOICE_TYPE.lower() == "twilio" and self.voice_manager and self.voice_manager.is_enabled:
            # Setup Twilio voice
            self.voice_manager.setup_voice("alice", rate=1.0, volume=1.0)
        else:
            # Setup pyttsx3 voice
            voices = self.tts.getProperty('voices')
            if voices:
                self.tts.setProperty('voice', voices[0].id)
            
            self.tts.setProperty('rate', VOICE_SETTINGS['rate'])
            self.tts.setProperty('volume', VOICE_SETTINGS['volume'])
    
    def load_documents(self):
        """Load documents and build RAG index"""
        try:
            with open('documents.json', 'r') as f:
                docs = json.load(f)
            
            if not self.rag.load_index():
                self.rag.build_index(docs)
                faiss.write_index(self.rag.index, 'faiss_index')
                with open('documents.json', 'w') as f:
                    json.dump(docs, f)
            
        except FileNotFoundError:
            print("⚠️  documents.json not found")
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
    
    def analyze_audio_sentiment(self, audio_data, sample_rate=16000):
        """Analyze audio sentiment using librosa"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Extract audio features
            features = self._extract_audio_features(audio_array, sample_rate)
            
            # Analyze voice characteristics
            voice_characteristics = self._analyze_voice_characteristics(features)
            
            # Classify emotion
            emotion = self._classify_voice_emotion(features)
            
            # Calculate confidence and engagement
            confidence = self._calculate_confidence_score(features)
            engagement = self._calculate_engagement_score(features)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'engagement': engagement,
                'voice_characteristics': voice_characteristics,
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
        """Extract audio features using librosa"""
        features = {}
        
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_array, sr=sample_rate)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
            
            # Energy and RMS
            rms = librosa.feature.rms(y=audio_array)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sample_rate)
            
            features = {
                'spectral_centroids': np.mean(spectral_centroids),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'spectral_bandwidth': np.mean(spectral_bandwidth),
                'mfccs': np.mean(mfccs, axis=1),
                'rms': np.mean(rms),
                'zcr': np.mean(zcr),
                'pitch_mean': np.mean(pitches[magnitudes > 0.1]) if np.any(magnitudes > 0.1) else 0,
                'pitch_std': np.std(pitches[magnitudes > 0.1]) if np.any(magnitudes > 0.1) else 0
            }
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
        
        return features
    
    def _analyze_voice_characteristics(self, features):
        """Analyze voice characteristics from features"""
        characteristics = {}
        
        try:
            # Energy level
            if features.get('rms', 0) > 0.1:
                characteristics['energy_level'] = 'high'
            elif features.get('rms', 0) > 0.05:
                characteristics['energy_level'] = 'medium'
            else:
                characteristics['energy_level'] = 'low'
            
            # Speech rate (based on zero crossing rate)
            if features.get('zcr', 0) > 0.1:
                characteristics['speech_rate'] = 'fast'
            elif features.get('zcr', 0) > 0.05:
                characteristics['speech_rate'] = 'normal'
            else:
                characteristics['speech_rate'] = 'slow'
            
            # Voice clarity (based on spectral bandwidth)
            if features.get('spectral_bandwidth', 0) > 2000:
                characteristics['voice_clarity'] = 'clear'
            elif features.get('spectral_bandwidth', 0) > 1000:
                characteristics['voice_clarity'] = 'moderate'
            else:
                characteristics['voice_clarity'] = 'unclear'
                
        except Exception as e:
            print(f"Error analyzing voice characteristics: {e}")
        
        return characteristics
    
    def _classify_voice_emotion(self, features):
        """Classify emotion based on audio features"""
        try:
            # Simple rule-based emotion classification
            pitch_mean = features.get('pitch_mean', 0)
            rms = features.get('rms', 0)
            spectral_centroids = features.get('spectral_centroids', 0)
            
            # High pitch + high energy = excited
            if pitch_mean > 200 and rms > 0.1:
                return 'excited'
            # High pitch + moderate energy = interested
            elif pitch_mean > 150 and rms > 0.05:
                return 'interested'
            # Low energy = concerned/frustrated
            elif rms < 0.03:
                return 'concerned'
            # Normal ranges = neutral
            else:
                return 'neutral'
                
        except Exception as e:
            print(f"Error classifying emotion: {e}")
            return 'neutral'
    
    def _calculate_confidence_score(self, features):
        """Calculate confidence score based on audio quality"""
        try:
            # Higher spectral bandwidth = clearer speech = higher confidence
            spectral_bandwidth = features.get('spectral_bandwidth', 0)
            confidence = min(1.0, spectral_bandwidth / 3000.0)
            return max(0.1, confidence)
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_engagement_score(self, features):
        """Calculate engagement score based on voice characteristics"""
        try:
            # Higher energy + higher pitch variation = more engaged
            rms = features.get('rms', 0)
            pitch_std = features.get('pitch_std', 0)
            
            energy_score = min(1.0, rms / 0.1)
            pitch_score = min(1.0, pitch_std / 100.0)
            
            engagement = (energy_score + pitch_score) / 2
            return max(0.1, engagement)
        except Exception as e:
            print(f"Error calculating engagement: {e}")
            return 0.5
    
    def start_audio_analysis(self):
        """Start real-time audio analysis"""
        self._audio_analysis_active = True
        print("🎤 Real-time audio analysis started")
    
    def stop_audio_analysis(self):
        """Stop real-time audio analysis"""
        self._audio_analysis_active = False
        print("🎤 Real-time audio analysis stopped")
    
    def _audio_analysis_worker(self):
        """Background worker for audio analysis"""
        try:
            while self._audio_analysis_active:
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
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
        if not self.sentiment_history:
            print("📊 No sentiment data available yet")
            return
        
        latest_sentiment = self.sentiment_history[-1]
        avg_sentiment = sum(s['polarity'] for s in self.sentiment_history) / len(self.sentiment_history)
        
        print("\n" + "="*50)
        print("🧠 SENTIMENT ANALYSIS & BUYING SIGNALS")
        print("="*50)
        
        print(f"📊 Latest Sentiment: {latest_sentiment['category']} ({latest_sentiment['polarity']:.2f})")
        print(f"📈 Average Sentiment: {avg_sentiment:.2f}")
        print(f"🎯 Buying Probability: {self._buying_probability:.1f}%")
        
        recommendation = self.get_buying_recommendation()
        print(f"💡 Recommendation: {recommendation['action']} - {recommendation['message']}")
        print("="*50)
    
    def pre_fetch_web_data(self):
        """Pre-fetch web data for current client"""
        if not self.current_client:
            return
        
        print(f"🌐 Fetching web data for {self.current_client.get('Full Name')}...")
        
        # Simulate web data fetching (replace with actual implementation)
        self._web_context = f"Web research data for {self.current_client.get('Company')}"
        self._website_context = f"Website data for {self.current_client.get('Company')}"
        
        print("✅ Web data pre-fetched")
    
    def listen(self):
        """Listen for user input with enhanced parameters"""
        with sr.Microphone() as source:
            # Adjust for ambient noise with longer duration
            print("🎤 Adjusting for ambient noise...")
            self.listener.adjust_for_ambient_noise(source, duration=2)
            
            # Optimized parameters for better recognition
            self.listener.energy_threshold = 300  # Lower threshold for better sensitivity
            self.listener.pause_threshold = 1.5  # Longer pause threshold
            self.listener.phrase_threshold = 0.3  # Shorter phrase threshold
            self.listener.non_speaking_duration = 0.5  # Shorter non-speaking duration
            
            print("🎤 Listening... (speak clearly and naturally)")
            
            try:
                audio = self.listener.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=30
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
                    
            except Exception as e:
                print(f"❌ Error in listen: {e}")
                return ""
    
    def speak(self, text):
        """Speak text with interruption capability"""
        print(f"🤖 Assistant: {text}")
        
        if VOICE_TYPE.lower() == "twilio" and self.voice_manager and self.voice_manager.is_enabled:
            # Use Twilio voice synthesis
            self.voice_manager.speak_interruptible(text)
        else:
            # Use pyttsx3 voice synthesis
            self.tts.say(text)
            try:
                self.tts.runAndWait()
            except KeyboardInterrupt:
                self.tts.stop()
                print("🔇 Speech interrupted")
    
    def speak_interruptible(self, text):
        """Speak text (simplified without interruption)"""
        print(f"🤖 Assistant: {text}")
        
        if VOICE_TYPE.lower() == "twilio" and self.voice_manager and self.voice_manager.is_enabled:
            # Use Twilio voice synthesis
            try:
                self.voice_manager.speak(text)
            except Exception as e:
                print(f"❌ Error speaking: {e}")
        else:
            # Use pyttsx3 voice synthesis
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
            prompt = f"""{system_instructions}

{client_info}
Context from documents: {rag_context}{combined_web_data}

Previous conversation:
{conversation_context}

Current question: {enhanced_query}

Please provide a personalized response that takes into account the client's role, company, web research data, website scraping data, and our conversation history."""
        else:
            prompt = f"""{system_instructions}

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
    
    def save_conversation(self):
        """Save conversation history to file"""
        try:
            os.makedirs('conversations', exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'conversations/conversation_{timestamp}.json'
            
            conversation_data = {
                'timestamp': timestamp,
                'client': self.current_client.get('Full Name') if self.current_client else 'Unknown',
                'conversation': self.conversation_history,
                'sentiment_history': self.sentiment_history,
                'buying_probability': self._buying_probability
            }
            
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            print(f"💾 Conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
    
    def load_conversation(self, session_id=None):
        """Load conversation history from file"""
        try:
            if session_id:
                filename = f'conversations/conversation_{session_id}.json'
            else:
                # Load most recent conversation
                conversations_dir = 'conversations'
                if not os.path.exists(conversations_dir):
                    return
                
                files = [f for f in os.listdir(conversations_dir) if f.startswith('conversation_')]
                if not files:
                    return
                
                files.sort(reverse=True)
                filename = os.path.join(conversations_dir, files[0])
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.conversation_history = data.get('conversation', [])
            self.sentiment_history = data.get('sentiment_history', [])
            self._buying_probability = data.get('buying_probability', 0.0)
            
            print(f"📂 Loaded conversation from {filename}")
            
        except Exception as e:
            print(f"❌ Error loading conversation: {e}")
    
    def add_to_history(self, user_input, response):
        """Add exchange to conversation history"""
        exchange = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user': user_input,
            'assistant': response
        }
        self.conversation_history.append(exchange)
        
        # Analyze sentiment
        sentiment_data = self.analyze_sentiment(user_input)
        self.sentiment_history.append(sentiment_data)
        
        # Detect buying signals
        buying_signals = self.detect_buying_signals(user_input)
        
        # Update buying probability
        self.update_buying_probability(user_input, sentiment_data, buying_signals)
    
    def get_conversation_context(self, max_exchanges=10):
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        recent_exchanges = self.conversation_history[-max_exchanges:]
        context_parts = []
        
        for exchange in recent_exchanges:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(context_parts)
    
    def apply_voice_settings(self):
        """Apply voice settings based on configuration"""
        if VOICE_TYPE.lower() == "twilio" and self.voice_manager and self.voice_manager.is_enabled:
            # Apply Twilio voice settings
            self.voice_manager.setup_voice("alice", rate=1.0, volume=1.0)
        else:
            # Apply pyttsx3 voice settings
            voices = self.tts.getProperty('voices')
            if voices:
                # Find Karen voice (American female)
                target_voice = None
                for voice in voices:
                    if 'karen' in voice.name.lower():
                        target_voice = voice
                        break
                
                if target_voice:
                    self.tts.setProperty('voice', target_voice.id)
                    print(f"🎤 Voice set to: {target_voice.name}")
                else:
                    self.tts.setProperty('voice', voices[0].id)
                    print(f"🎤 Voice set to: {voices[0].name}")
            
            self.tts.setProperty('rate', VOICE_SETTINGS['rate'])
            self.tts.setProperty('volume', VOICE_SETTINGS['volume'])
    
    def setup_client_session(self, client):
        """Setup session for a specific client"""
        self.current_client = client
        print(f"👤 Client session setup: {client.get('Full Name')} from {client.get('Company')}")
        
        # Pre-fetch web data
        self.pre_fetch_web_data()
        
        # Create welcome message
        welcome_message = f"Hello {client.get('Full Name')}! I'm your AI sales assistant. I'm here to help you with information about Hyprbots' innovative automation solutions, including our AI-powered chatbots, workflow automation tools, and custom software development services. How can I assist you today?"
        
        return welcome_message
    
    def run(self):
        """Main conversation loop"""
        print("🚀 Starting Sales Voice Agent with Twilio Voice Option")
        print(f"🎤 Voice Type: {VOICE_TYPE}")
        print("="*60)
        
        # Initialize RAG systems
        print("📚 Initializing RAG systems...")
        
        # Start audio analysis
        self.start_audio_analysis()
        
        # Apply voice settings
        self.apply_voice_settings()
        
        # Welcome message
        if self.current_client:
            welcome_message = self.setup_client_session(self.current_client)
            self.speak_interruptible(welcome_message)
        else:
            # Ask for phone number if no client identified
            self.speak_interruptible("Please provide the client's phone number to start a personalized conversation.")
            phone_input = input("📱 Enter phone number (or press Enter to continue without client): ").strip()
            
            if phone_input:
                client = self.client_rag.search_by_phone(phone_input)
                if client:
                    self.current_client = client
                    welcome_message = self.setup_client_session(client)
                    self.speak_interruptible(welcome_message)
                else:
                    self.speak_interruptible("I couldn't find a client with that phone number. Let's continue with general assistance.")
            else:
                self.speak_interruptible("No phone number provided. Let's continue with general assistance.")
        
        print("\n💬 Conversation started! (Type 'quit' to exit)")
        print("="*60)
        
        try:
            while True:
                # Listen for user input
                user_input = self.listen()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    if self.current_client:
                        self.speak_interruptible(f"Thank you {self.current_client.get('Full Name')}! Have a great day!")
                    else:
                        self.speak_interruptible("Thank you for your patience! Have a Good Day!")
                    break
                
                # Get AI response
                response = self.answer(user_input)
                
                # Add to conversation history
                self.add_to_history(user_input, response)
                
                # Speak response
                self.speak_interruptible(response)
                
                # Print sentiment analysis (every 3rd interaction)
                if len(self.conversation_history) % 3 == 0:
                    self.print_sentiment_analysis()
                
                # Save conversation periodically
                if len(self.conversation_history) % 5 == 0:
                    self.save_conversation()
                
        except KeyboardInterrupt:
            print("\n\n🛑 Conversation interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in conversation loop: {e}")
        finally:
            # Stop audio analysis
            self.stop_audio_analysis()
            
            # Save final conversation
            self.save_conversation()
            
            print("👋 Sales Voice Agent stopped")


if __name__ == "__main__":
    import sys
    
    # Get phone number from command line argument
    phone_number = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Create and run voice agent
    agent = VoiceAgent(phone_number=phone_number)
    agent.run() 