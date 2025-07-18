#!/usr/bin/env python3
"""
Test script for sentiment analysis and buying signal detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import VoiceAgent

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("🧠 Testing Sentiment Analysis & Buying Signal Detection")
    print("=" * 60)
    
    # Create agent instance
    agent = VoiceAgent()
    
    # Test cases with different sentiment and buying signals
    test_cases = [
        {
            'input': "I'm very interested in your AI solutions. What's the cost?",
            'expected_sentiment': 'Positive',
            'expected_signals': 'High buying signals'
        },
        {
            'input': "This sounds expensive. I'm not sure we can afford it.",
            'expected_sentiment': 'Negative',
            'expected_signals': 'Objections detected'
        },
        {
            'input': "Tell me more about the features and benefits.",
            'expected_sentiment': 'Neutral',
            'expected_signals': 'Moderate buying signals'
        },
        {
            'input': "We need to automate our invoice processing. When can we start?",
            'expected_sentiment': 'Positive',
            'expected_signals': 'Strong buying signals'
        },
        {
            'input': "I'll think about it and get back to you later.",
            'expected_sentiment': 'Neutral',
            'expected_signals': 'Objections detected'
        }
    ]
    
    print("📊 Running sentiment analysis tests...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['input']}")
        print("-" * 40)
        
        # Analyze sentiment
        sentiment = agent.analyze_sentiment(test_case['input'])
        print(f"Sentiment: {sentiment['category']} (Score: {sentiment['polarity']:.2f})")
        
        # Detect buying signals
        signals = agent.detect_buying_signals(test_case['input'])
        print(f"Buying Signals: {signals['buying_signals']}")
        print(f"Objection Signals: {signals['objection_signals']}")
        print(f"Net Signal Strength: {signals['net_signal']:.1f}%")
        
        # Update probability
        probability = agent.update_buying_probability(test_case['input'], sentiment, signals)
        print(f"Buying Probability: {probability:.1f}%")
        
        # Get recommendation
        recommendation = agent.get_buying_recommendation()
        print(f"Recommendation: {recommendation['action']} - {recommendation['message']}")
        
        print()
    
    # Print final analysis
    print("🎯 FINAL ANALYSIS")
    print("=" * 60)
    agent.print_sentiment_analysis()
    
    print("\n✅ Sentiment analysis test completed!")

def test_buying_signal_keywords():
    """Test buying signal keyword detection"""
    print("\n🔍 Testing Buying Signal Keywords")
    print("=" * 60)
    
    agent = VoiceAgent()
    
    # Test buying keywords
    buying_tests = [
        "I'm interested in your solution",
        "What's the cost and timeline?",
        "We need to implement this soon",
        "Can you schedule a demo?",
        "Tell me about the ROI and benefits"
    ]
    
    print("✅ Testing Positive Buying Signals:")
    for test in buying_tests:
        signals = agent.detect_buying_signals(test)
        print(f"  '{test}' -> {signals['buying_signals']} buying signals")
    
    # Test objection keywords
    objection_tests = [
        "This seems expensive",
        "We need to think about it",
        "I'm not sure about the budget",
        "Maybe we'll look at alternatives",
        "We're too busy right now"
    ]
    
    print("\n⚠️ Testing Objection Signals:")
    for test in objection_tests:
        signals = agent.detect_buying_signals(test)
        print(f"  '{test}' -> {signals['objection_signals']} objection signals")

if __name__ == "__main__":
    test_sentiment_analysis()
    test_buying_signal_keywords() 