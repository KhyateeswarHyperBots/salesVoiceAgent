#!/usr/bin/env python3
"""
Test Phone Call Integration
Demonstrates the phone call functionality
"""

import os
import sys

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

def test_phone_call_setup():
    """Test phone call setup"""
    print("📞 Testing Phone Call Integration")
    print("=" * 50)
    
    # Check credentials
    twilio_sid = os.getenv("TWILIO_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("TO_PHONE_NUMBER")
    
    print("🔑 Credentials Check:")
    print(f"   TWILIO_SID: {'✅ Set' if twilio_sid else '❌ Not set'}")
    print(f"   TWILIO_AUTH_TOKEN: {'✅ Set' if twilio_token else '❌ Not set'}")
    print(f"   FROM_NUMBER: {from_number or '❌ Not set'}")
    print(f"   TO_NUMBER: {to_number or '❌ Not set'}")
    
    if not all([twilio_sid, twilio_token, from_number, to_number]):
        print("\n❌ Missing credentials")
        print("   Please set all required environment variables in config.env")
        return False
    
    print("\n✅ All credentials are set!")
    return True

def test_twilio_client():
    """Test Twilio client connection"""
    print("\n🔌 Testing Twilio Client Connection:")
    print("-" * 40)
    
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv("TWILIO_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        client = Client(account_sid, auth_token)
        
        # Test API connection by listing calls
        calls = client.calls.list(limit=1)
        print("✅ Twilio client connection successful")
        print(f"   Account: {client.account_sid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Twilio client connection failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\n🤖 Testing Ollama Connection:")
    print("-" * 40)
    
    try:
        import ollama
        
        # Test Ollama connection
        models = ollama.list()
        print("✅ Ollama connection successful")
        print(f"   Available models: {len(models)}")
        
        for model in models:
            print(f"   - {model['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False

def show_setup_instructions():
    """Show setup instructions"""
    print("\n📖 Phone Call Setup Instructions:")
    print("=" * 50)
    
    print("1. Install dependencies:")
    print("   pip install flask twilio")
    print()
    
    print("2. Set up ngrok (for webhook exposure):")
    print("   # Install ngrok")
    print("   brew install ngrok  # macOS")
    print("   # or download from https://ngrok.com/")
    print()
    print("   # Expose your local server")
    print("   ngrok http 5000")
    print()
    
    print("3. Update webhook URL in phone_call_agent.py:")
    print("   Replace 'https://your-ngrok-url.ngrok.io' with your actual ngrok URL")
    print()
    
    print("4. Run the phone call agent:")
    print("   python phone_call_agent.py")
    print()
    
    print("5. Make a call:")
    print("   - Choose option 1 from the menu")
    print("   - Enter the phone number to call")
    print()
    
    print("6. Receive calls:")
    print("   - Start the webhook server (option 2)")
    print("   - Configure your Twilio number webhook URL")
    print("   - Point it to: https://your-ngrok-url.ngrok.io/call")

def show_webhook_setup():
    """Show webhook setup instructions"""
    print("\n🌐 Webhook Setup for Incoming Calls:")
    print("=" * 50)
    
    print("1. Start ngrok:")
    print("   ngrok http 5000")
    print()
    
    print("2. Copy the HTTPS URL (e.g., https://abc123.ngrok.io)")
    print()
    
    print("3. Go to Twilio Console:")
    print("   https://console.twilio.com/")
    print()
    
    print("4. Navigate to Phone Numbers > Manage > Active numbers")
    print()
    
    print("5. Click on your phone number")
    print()
    
    print("6. Set the webhook URL:")
    print("   Voice Configuration:")
    print("   - Webhook URL: https://your-ngrok-url.ngrok.io/call")
    print("   - HTTP Method: POST")
    print()
    
    print("7. Save the configuration")
    print()
    
    print("8. Now incoming calls will be handled by your agent!")

def main():
    """Main test function"""
    print("📞 Phone Call Integration Test Suite")
    print("=" * 60)
    
    # Run tests
    creds_ok = test_phone_call_setup()
    twilio_ok = test_twilio_client() if creds_ok else False
    ollama_ok = test_ollama_connection()
    
    print("\n📊 Test Results:")
    print("=" * 30)
    print(f"   Credentials: {'✅' if creds_ok else '❌'}")
    print(f"   Twilio Client: {'✅' if twilio_ok else '❌'}")
    print(f"   Ollama: {'✅' if ollama_ok else '❌'}")
    
    if creds_ok and twilio_ok and ollama_ok:
        print("\n🎉 All tests passed! You're ready to make phone calls.")
        print("\n🚀 Next steps:")
        print("   1. Run: python phone_call_agent.py")
        print("   2. Choose option 1 to make a call")
        print("   3. Or choose option 2 to start webhook server")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    # Show instructions
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        show_setup_instructions()
    elif len(sys.argv) > 1 and sys.argv[1] == "webhook":
        show_webhook_setup()
    else:
        print("\n💡 For setup instructions, run:")
        print("   python test_phone_call.py setup")
        print("\n💡 For webhook setup, run:")
        print("   python test_phone_call.py webhook")

if __name__ == "__main__":
    main() 