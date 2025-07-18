#!/usr/bin/env python3
"""
Sales Voice Agent Launcher
Simple script to load configuration and run the voice agent
"""

import os
import sys
import subprocess

def main():
    """Main launcher function"""
    print("🚀 Sales Voice Agent Launcher")
    print("=" * 40)
    
    # Load configuration
    try:
        from load_config import load_config
        load_config()
        print("✅ Configuration loaded")
    except ImportError:
        print("⚠️  Configuration loader not found, using environment variables")
    
    # Get phone number from command line
    phone_number = None
    if len(sys.argv) > 1:
        phone_number = sys.argv[1]
    
    # Check if Ollama is running
    try:
        import ollama
        ollama.list()
        print("✅ Ollama is running")
    except Exception as e:
        print("❌ Ollama is not running")
        print("   Please start Ollama: ollama serve")
        return
    
    # Check voice type
    voice_type = os.getenv("VOICE_TYPE", "pyttsx3")
    print(f"🎤 Voice Type: {voice_type}")
    
    # Show client info if phone number provided
    if phone_number:
        try:
            import json
            with open('clients.json', 'r') as f:
                clients = json.load(f)
            
            client = None
            for c in clients:
                if c.get('Phone') == phone_number:
                    client = c
                    break
            
            if client:
                print(f"👤 Client: {client.get('Full Name')} from {client.get('Company')}")
            else:
                print(f"⚠️  No client found for phone number: {phone_number}")
        except Exception as e:
            print(f"⚠️  Could not load client data: {e}")
    
    print("\n🎯 Starting Sales Voice Agent...")
    print("=" * 40)
    
    # Run the appropriate agent
    if voice_type.lower() == "twilio":
        print("🎤 Using Twilio Voice Synthesis")
        subprocess.run([sys.executable, "main_twilio.py"] + (sys.argv[1:] if len(sys.argv) > 1 else []))
    else:
        print("🎤 Using pyttsx3 Voice Synthesis")
        subprocess.run([sys.executable, "main.py"] + (sys.argv[1:] if len(sys.argv) > 1 else []))

if __name__ == "__main__":
    main() 