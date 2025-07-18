#!/usr/bin/env python3
"""
Configuration file for the Sales Voice Agent
"""

import os

# OpenAI API Configuration
OPENAI_API_KEY = "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"

# Set environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Voice Agent Configuration
VOICE_SETTINGS = {
    'rate': 155,  # Words per minute
    'volume': 0.85,  # Volume level (0.0 to 1.0)
    'voice_name': 'Karen'  # Voice to use
}

# AI Model Configuration
AI_MODEL = "llama3.2:3b"

# Speech Recognition Configuration
SPEECH_RECOGNITION = {
    'energy_threshold': 300,
    'pause_threshold': 1.5,
    'ambient_noise_duration': 1.0,
    'timeout': 15,
    'phrase_time_limit': 20
}

# Non-interruption period (seconds)
NON_INTERRUPTION_PERIOD = 5

# File paths
DOC_PATH = "documents.json"
INDEX_PATH = "vector.index"
EMBEDDINGS_PATH = "embeddings.pkl"
CALENDAR_PATH = "calendar.txt"
CLIENTS_PATH = "clients.json" 