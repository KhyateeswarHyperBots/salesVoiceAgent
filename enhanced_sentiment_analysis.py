#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis using Open Source Models
Multiple models for better accuracy and different use cases
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class EnhancedSentimentAnalysis:
    """Enhanced sentiment analysis using multiple open source models"""
    
    def __init__(self, model_type="ensemble"):
        """
        Initialize sentiment analysis with specified model type
        
        Args:
            model_type: "textblob", "vader", "transformers", "spacy", or "ensemble"
        """
        self.model_type = model_type
        self.models = {}
        self.initialize_models()
        
        # Sales-specific keywords for enhanced analysis
        self.sales_keywords = {
            'positive': [
                'interested', 'great', 'excellent', 'amazing', 'perfect', 'love', 'like',
                'good', 'beneficial', 'valuable', 'helpful', 'useful', 'effective',
                'efficient', 'fast', 'quick', 'easy', 'simple', 'convenient',
                'affordable', 'reasonable', 'worth', 'investment', 'solution',
                'improve', 'better', 'upgrade', 'modern', 'advanced', 'innovative'
            ],
            'negative': [
                'expensive', 'costly', 'expensive', 'overpriced', 'unaffordable',
                'difficult', 'complex', 'complicated', 'hard', 'challenging',
                'slow', 'inefficient', 'waste', 'useless', 'pointless',
                'worried', 'concerned', 'scared', 'afraid', 'risky', 'dangerous',
                'not sure', 'doubt', 'skeptical', 'suspicious', 'uncomfortable'
            ],
            'buying_signals': [
                'pricing', 'cost', 'price', 'quote', 'proposal', 'demo', 'trial',
                'purchase', 'buy', 'implement', 'deploy', 'solution', 'benefits',
                'roi', 'investment', 'budget', 'timeline', 'schedule', 'next steps',
                'decision', 'approval', 'contract', 'agreement', 'start', 'begin'
            ],
            'objections': [
                'expensive', 'costly', 'budget', 'not sure', 'concerned', 'worried',
                'risk', 'security', 'integration', 'complex', 'difficult', 'time',
                'resources', 'staff', 'training', 'support', 'maintenance',
                'replacement', 'migration', 'change', 'disruption', 'learning curve'
            ]
        }
    
    def initialize_models(self):
        """Initialize the requested sentiment analysis models"""
        try:
            if self.model_type in ["textblob", "ensemble"]:
                from textblob import TextBlob
                self.models['textblob'] = TextBlob
            
            if self.model_type in ["vader", "ensemble"]:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.models['vader'] = SentimentIntensityAnalyzer()
            
            if self.model_type in ["transformers", "ensemble"]:
                try:
                    from transformers import pipeline
                    self.models['transformers'] = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        return_all_scores=True
                    )
                except Exception as e:
                    print(f"⚠️ Transformers model not available: {e}")
            
            if self.model_type in ["spacy", "ensemble"]:
                try:
                    import spacy
                    # Try to load English model, download if not available
                    try:
                        self.models['spacy'] = spacy.load("en_core_web_sm")
                    except OSError:
                        print("📥 Downloading spaCy English model...")
                        os.system("python -m spacy download en_core_web_sm")
                        self.models['spacy'] = spacy.load("en_core_web_sm")
                except Exception as e:
                    print(f"⚠️ spaCy model not available: {e}")
            
            print(f"✅ Initialized sentiment analysis with: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            # Fallback to TextBlob
            from textblob import TextBlob
            self.models['textblob'] = TextBlob
    
    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            blob = self.models['textblob'](text)
            sentiment_score = blob.sentiment.polarity
            subjectivity_score = blob.sentiment.subjectivity
            
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                'model': 'textblob',
                'sentiment': sentiment,
                'polarity': sentiment_score,
                'subjectivity': subjectivity_score,
                'confidence': abs(sentiment_score)
            }
        except Exception as e:
            print(f"Error in TextBlob analysis: {e}")
            return self._get_default_result('textblob')
    
    def analyze_sentiment_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        try:
            scores = self.models['vader'].polarity_scores(text)
            
            # Determine sentiment category
            if scores['compound'] >= 0.05:
                sentiment = "positive"
            elif scores['compound'] <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                'model': 'vader',
                'sentiment': sentiment,
                'polarity': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'confidence': abs(scores['compound'])
            }
        except Exception as e:
            print(f"Error in VADER analysis: {e}")
            return self._get_default_result('vader')
    
    def analyze_sentiment_transformers(self, text: str) -> Dict:
        """Analyze sentiment using Transformers"""
        try:
            if 'transformers' not in self.models:
                return self._get_default_result('transformers')
            
            results = self.models['transformers'](text)
            
            # Find the highest scoring sentiment
            best_result = max(results[0], key=lambda x: x['score'])
            
            # Map labels to our format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            sentiment = label_mapping.get(best_result['label'], 'neutral')
            
            return {
                'model': 'transformers',
                'sentiment': sentiment,
                'confidence': best_result['score'],
                'all_scores': results[0]
            }
        except Exception as e:
            print(f"Error in Transformers analysis: {e}")
            return self._get_default_result('transformers')
    
    def analyze_sentiment_spacy(self, text: str) -> Dict:
        """Analyze sentiment using spaCy"""
        try:
            if 'spacy' not in self.models:
                return self._get_default_result('spacy')
            
            doc = self.models['spacy'](text)
            
            # spaCy doesn't have built-in sentiment, but we can analyze entities and patterns
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Simple sentiment based on positive/negative words
            positive_words = sum(1 for token in doc if token.text.lower() in self.sales_keywords['positive'])
            negative_words = sum(1 for token in doc if token.text.lower() in self.sales_keywords['negative'])
            
            if positive_words > negative_words:
                sentiment = "positive"
                confidence = min(0.9, positive_words / (positive_words + negative_words + 1))
            elif negative_words > positive_words:
                sentiment = "negative"
                confidence = min(0.9, negative_words / (positive_words + negative_words + 1))
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            return {
                'model': 'spacy',
                'sentiment': sentiment,
                'confidence': confidence,
                'entities': entities,
                'positive_words': positive_words,
                'negative_words': negative_words
            }
        except Exception as e:
            print(f"Error in spaCy analysis: {e}")
            return self._get_default_result('spacy')
    
    def analyze_sentiment_ensemble(self, text: str) -> Dict:
        """Analyze sentiment using ensemble of all available models"""
        results = {}
        
        # Run all available models
        if 'textblob' in self.models:
            results['textblob'] = self.analyze_sentiment_textblob(text)
        
        if 'vader' in self.models:
            results['vader'] = self.analyze_sentiment_vader(text)
        
        if 'transformers' in self.models:
            results['transformers'] = self.analyze_sentiment_transformers(text)
        
        if 'spacy' in self.models:
            results['spacy'] = self.analyze_sentiment_spacy(text)
        
        # Combine results using weighted voting
        sentiment_votes = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = 0
        
        for model_name, result in results.items():
            sentiment = result.get('sentiment', 'neutral')
            confidence = result.get('confidence', 0.5)
            
            sentiment_votes[sentiment] += confidence
            total_confidence += confidence
        
        # Determine final sentiment
        if total_confidence > 0:
            final_sentiment = max(sentiment_votes, key=sentiment_votes.get)
            final_confidence = sentiment_votes[final_sentiment] / total_confidence
        else:
            final_sentiment = 'neutral'
            final_confidence = 0.5
        
        return {
            'model': 'ensemble',
            'sentiment': final_sentiment,
            'confidence': final_confidence,
            'individual_results': results,
            'sentiment_votes': sentiment_votes
        }
    
    def detect_sales_signals(self, text: str) -> Dict:
        """Detect sales-specific signals and buying intent"""
        text_lower = text.lower()
        
        # Count keyword occurrences
        buying_signals = [word for word in self.sales_keywords['buying_signals'] if word in text_lower]
        objections = [word for word in self.sales_keywords['objections'] if word in text_lower]
        positive_words = [word for word in self.sales_keywords['positive'] if word in text_lower]
        negative_words = [word for word in self.sales_keywords['negative'] if word in text_lower]
        
        # Calculate scores
        buying_score = len(buying_signals)
        objection_score = len(objections)
        positive_score = len(positive_words)
        negative_score = len(negative_words)
        
        # Determine buying intent
        if buying_score > 0 and objection_score == 0:
            intent = "strong_buying"
        elif buying_score > objection_score:
            intent = "moderate_buying"
        elif objection_score > buying_score:
            intent = "objections"
        else:
            intent = "neutral"
        
        return {
            'buying_signals': buying_signals,
            'objections': objections,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'buying_score': buying_score,
            'objection_score': objection_score,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'intent': intent,
            'confidence': min(1.0, (buying_score + positive_score) / (buying_score + objection_score + positive_score + negative_score + 1))
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Main sentiment analysis method"""
        if not text or not text.strip():
            return self._get_default_result(self.model_type)
        
        # Analyze sentiment based on model type
        if self.model_type == "textblob":
            result = self.analyze_sentiment_textblob(text)
        elif self.model_type == "vader":
            result = self.analyze_sentiment_vader(text)
        elif self.model_type == "transformers":
            result = self.analyze_sentiment_transformers(text)
        elif self.model_type == "spacy":
            result = self.analyze_sentiment_spacy(text)
        elif self.model_type == "ensemble":
            result = self.analyze_sentiment_ensemble(text)
        else:
            result = self.analyze_sentiment_textblob(text)
        
        # Add sales signal analysis
        sales_signals = self.detect_sales_signals(text)
        result['sales_signals'] = sales_signals
        
        # Add timestamp
        result['timestamp'] = time.time()
        
        return result
    
    def _get_default_result(self, model_name: str) -> Dict:
        """Get default result when analysis fails"""
        return {
            'model': model_name,
            'sentiment': 'neutral',
            'polarity': 0.0,
            'confidence': 0.5,
            'sales_signals': {
                'buying_signals': [],
                'objections': [],
                'buying_score': 0,
                'objection_score': 0,
                'intent': 'neutral',
                'confidence': 0.5
            },
            'timestamp': time.time()
        }
    
    def get_buying_probability(self, sentiment_data: Dict) -> float:
        """Calculate buying probability from sentiment data"""
        base_probability = 50.0  # Start at 50%
        
        # Sentiment impact
        sentiment = sentiment_data.get('sentiment', 'neutral')
        confidence = sentiment_data.get('confidence', 0.5)
        
        if sentiment == 'positive':
            base_probability += 20 * confidence
        elif sentiment == 'negative':
            base_probability -= 20 * confidence
        
        # Sales signals impact
        sales_signals = sentiment_data.get('sales_signals', {})
        buying_score = sales_signals.get('buying_score', 0)
        objection_score = sales_signals.get('objection_score', 0)
        
        base_probability += buying_score * 10
        base_probability -= objection_score * 8
        
        # Clamp between 0 and 100
        return max(0.0, min(100.0, base_probability))
    
    def print_analysis(self, text: str, sentiment_data: Dict):
        """Print detailed sentiment analysis"""
        print(f"\n📊 SENTIMENT ANALYSIS: '{text[:50]}...'")
        print(f"   Model: {sentiment_data.get('model', 'unknown')}")
        print(f"   Sentiment: {sentiment_data.get('sentiment', 'unknown')}")
        print(f"   Confidence: {sentiment_data.get('confidence', 0):.2f}")
        
        sales_signals = sentiment_data.get('sales_signals', {})
        if sales_signals.get('buying_signals'):
            print(f"   🎯 Buying Signals: {', '.join(sales_signals['buying_signals'])}")
        if sales_signals.get('objections'):
            print(f"   ⚠️ Objections: {', '.join(sales_signals['objections'])}")
        
        buying_prob = self.get_buying_probability(sentiment_data)
        print(f"   💰 Buying Probability: {buying_prob:.1f}%")


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced sentiment analysis
    analyzer = EnhancedSentimentAnalysis(model_type="ensemble")
    
    test_texts = [
        "I'm very interested in your pricing and would like to see a demo",
        "This seems too expensive and I'm not sure about the integration",
        "The solution looks good but I need to think about it",
        "I love the benefits and want to implement this right away"
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        analyzer.print_analysis(text, result)
        print("-" * 50) 