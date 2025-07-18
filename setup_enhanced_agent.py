#!/usr/bin/env python3
"""
Setup Enhanced Interactive Voice Agent
Helps configure and run the enhanced interactive voice agent with all features
"""

import os
import subprocess
import time
import requests
import json

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking Enhanced Agent Requirements...")
    print("=" * 50)
    
    # Check Twilio credentials
    twilio_sid = os.getenv("TWILIO_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    
    print(f"   Twilio SID: {'✅' if twilio_sid else '❌'}")
    print(f"   Twilio Token: {'✅' if twilio_token else '❌'}")
    print(f"   From Number: {'✅' if from_number else '❌'}")
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        print(f"   Ollama: ✅ ({len(models)} models)")
    except:
        print("   Ollama: ❌ (not running)")
    
    # Check Flask
    try:
        import flask
        print("   Flask: ✅")
    except:
        print("   Flask: ❌ (not installed)")
    
    # Check Twilio
    try:
        import twilio
        print("   Twilio: ✅")
    except:
        print("   Twilio: ❌ (not installed)")
    
    # Check RAG dependencies
    try:
        import faiss
        print("   FAISS: ✅")
    except:
        print("   FAISS: ❌ (not installed)")
    
    try:
        import sentence_transformers
        print("   Sentence Transformers: ✅")
    except:
        print("   Sentence Transformers: ❌ (not installed)")
    
    try:
        import textblob
        print("   TextBlob: ✅")
    except:
        print("   TextBlob: ❌ (not installed)")
    
    try:
        import numpy
        print("   NumPy: ✅")
    except:
        print("   NumPy: ❌ (not installed)")
    
    try:
        import librosa
        print("   Librosa: ✅")
    except:
        print("   Librosa: ❌ (not installed)")
    
    return all([twilio_sid, twilio_token, from_number])

def install_missing_dependencies():
    """Install missing dependencies"""
    print("\n📦 Installing missing dependencies...")
    
    dependencies = [
        'flask',
        'twilio',
        'faiss-cpu',
        'sentence-transformers',
        'textblob',
        'numpy',
        'librosa',
        'soundfile',
        'scipy'
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True)
            print(f"   ✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {dep}")

def get_ngrok_url():
    """Get ngrok URL if running"""
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
        if response.status_code == 200:
            tunnels = response.json()
            for tunnel in tunnels['tunnels']:
                if tunnel['proto'] == 'https':
                    return tunnel['public_url']
    except:
        pass
    return None

def start_ngrok():
    """Start ngrok tunnel"""
    print("\n🌐 Starting ngrok tunnel...")
    
    # Check if ngrok is already running
    ngrok_url = get_ngrok_url()
    if ngrok_url:
        print(f"   ✅ ngrok already running: {ngrok_url}")
        return ngrok_url
    
    try:
        # Start ngrok in background
        process = subprocess.Popen(
            ["ngrok", "http", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ngrok to start
        print("   ⏳ Waiting for ngrok to start...")
        for i in range(10):
            time.sleep(1)
            ngrok_url = get_ngrok_url()
            if ngrok_url:
                print(f"   ✅ ngrok started: {ngrok_url}")
                return ngrok_url
        
        print("   ❌ ngrok failed to start")
        return None
        
    except Exception as e:
        print(f"   ❌ Error starting ngrok: {e}")
        return None

def update_webhook_url(ngrok_url):
    """Update webhook URL in the enhanced agent"""
    try:
        with open('enhanced_interactive_agent.py', 'r') as f:
            content = f.read()
        
        # Replace the placeholder URL
        updated_content = content.replace(
            'return "https://your-ngrok-url.ngrok.io"',
            f'return "{ngrok_url}"'
        )
        
        with open('enhanced_interactive_agent.py', 'w') as f:
            f.write(updated_content)
        
        print(f"   ✅ Updated webhook URL to: {ngrok_url}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error updating webhook URL: {e}")
        return False

def configure_twilio_webhook(ngrok_url):
    """Configure Twilio webhook URL"""
    print(f"\n📞 Configuring Twilio webhook...")
    print("=" * 40)
    
    print("1. Go to Twilio Console: https://console.twilio.com/")
    print("2. Navigate to Phone Numbers > Manage > Active numbers")
    print("3. Click on your phone number")
    print("4. Under 'Voice & Fax > A CALL COMES IN', set:")
    print(f"   Webhook URL: {ngrok_url}/call")
    print("   HTTP Method: POST")
    print("5. Save the configuration")
    print()
    print("   ⏳ Press Enter when you've configured the webhook...")
    input()

def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Checking data files...")
    
    files_to_check = [
        ('clients.json', 'Client data'),
        ('documents.json', 'Knowledge base documents'),
        ('agent_config.py', 'Agent configuration')
    ]
    
    missing_files = []
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"   ✅ {description}: {filename}")
        else:
            print(f"   ❌ {description}: {filename} (missing)")
            missing_files.append(filename)
    
    return missing_files

def start_enhanced_agent():
    """Start the enhanced interactive voice agent"""
    print("\n🎤 Starting Enhanced Interactive Voice Agent...")
    print("=" * 50)
    
    try:
        # Import and start the agent
        from enhanced_interactive_agent import EnhancedInteractiveVoiceAgent
        
        agent = EnhancedInteractiveVoiceAgent()
        print("   ✅ Enhanced agent initialized")
        print("   🧠 Features: RAG, Sentiment Analysis, Client Data, Buying Signals")
        
        # Start the server
        print("   🌐 Starting webhook server...")
        agent.start_server()
        
    except Exception as e:
        print(f"   ❌ Error starting enhanced agent: {e}")

def main():
    """Main setup function"""
    print("🎤 Enhanced Interactive Voice Agent Setup")
    print("=" * 60)
    print("Features: RAG + Sentiment Analysis + Client Data + Buying Signals")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Missing requirements.")
        install_choice = input("Would you like to install missing dependencies? (y/n): ").strip().lower()
        if install_choice == 'y':
            import sys
            install_missing_dependencies()
        else:
            print("Please install missing dependencies manually and try again.")
            return
    
    print("\n✅ All requirements met!")
    
    # Check data files
    missing_files = check_data_files()
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        print("   Some features may not work properly without these files.")
    
    # Start ngrok
    ngrok_url = start_ngrok()
    if not ngrok_url:
        print("\n❌ Failed to start ngrok. Please start it manually:")
        print("   ngrok http 5000")
        return
    
    # Update webhook URL
    if not update_webhook_url(ngrok_url):
        print("\n❌ Failed to update webhook URL")
        return
    
    # Configure Twilio webhook
    configure_twilio_webhook(ngrok_url)
    
    # Start the enhanced agent
    start_enhanced_agent()

if __name__ == "__main__":
    main() 