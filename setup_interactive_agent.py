#!/usr/bin/env python3
"""
Setup Interactive Voice Agent
Helps configure and run the interactive voice agent with ngrok
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
    print("🔍 Checking requirements...")
    print("=" * 40)
    
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
    
    return all([twilio_sid, twilio_token, from_number])

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
    """Update webhook URL in the interactive agent"""
    try:
        with open('interactive_voice_agent.py', 'r') as f:
            content = f.read()
        
        # Replace the placeholder URL
        updated_content = content.replace(
            'return "https://your-ngrok-url.ngrok.io"',
            f'return "{ngrok_url}"'
        )
        
        with open('interactive_voice_agent.py', 'w') as f:
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

def start_interactive_agent():
    """Start the interactive voice agent"""
    print("\n🎤 Starting Interactive Voice Agent...")
    print("=" * 40)
    
    try:
        # Import and start the agent
        from interactive_voice_agent import InteractiveVoiceAgent
        
        agent = InteractiveVoiceAgent()
        print("   ✅ Agent initialized")
        
        # Start the server
        print("   🌐 Starting webhook server...")
        agent.start_server()
        
    except Exception as e:
        print(f"   ❌ Error starting agent: {e}")

def main():
    """Main setup function"""
    print("🎤 Interactive Voice Agent Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Missing requirements. Please fix the issues above.")
        return
    
    print("\n✅ All requirements met!")
    
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
    
    # Start the agent
    start_interactive_agent()

if __name__ == "__main__":
    main() 