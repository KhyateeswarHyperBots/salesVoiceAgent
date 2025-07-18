#!/usr/bin/env python3
"""
Test script for Twilio Voice Synthesis Option
Demonstrates the alternative voice synthesis using Twilio
"""

import os
import sys

# Load configuration
try:
    from load_config import load_config
    load_config()
except ImportError:
    pass

from twilio_voice import create_twilio_voice, TwilioVoiceManager

def test_twilio_voice():
    """Test Twilio voice synthesis functionality"""
    print("🧪 Testing Twilio Voice Synthesis")
    print("=" * 50)
    
    # Create Twilio voice manager
    voice_manager = create_twilio_voice()
    
    if not voice_manager.is_enabled:
        print("❌ Twilio voice synthesis is not available")
        print("   Please set the following environment variables:")
        print("   - TWILIO_SID: Your Twilio Account SID")
        print("   - TWILIO_AUTH_TOKEN: Your Twilio Auth Token")
        print()
        print("   You can get these from your Twilio Console:")
        print("   https://console.twilio.com/")
        return False
    
    print("✅ Twilio voice synthesis is available")
    print()
    
    # List available voices
    voice_manager.list_voices()
    print()
    
    # Test different voices
    test_voices = ["alice", "bob", "charlie", "diana", "eva"]
    test_text = "Hello! This is a test of the Twilio voice synthesis system. How does this sound?"
    
    print("🎤 Testing different voices:")
    print("-" * 30)
    
    for i, voice in enumerate(test_voices, 1):
        print(f"\n{i}. Testing voice: {voice}")
        voice_manager.setup_voice(voice, rate=1.0, volume=1.0)
        voice_manager.speak(test_text)
        print(f"   ✅ Voice {voice} test completed")
    
    print("\n🎤 Testing voice settings:")
    print("-" * 30)
    
    # Test different rates
    rates = [0.8, 1.0, 1.2, 1.5]
    for rate in rates:
        print(f"\nTesting speech rate: {rate}")
        voice_manager.setup_voice("alice", rate=rate, volume=1.0)
        voice_manager.speak(f"This is a test at {rate}x speed.")
    
    # Test different volumes
    volumes = [0.5, 0.8, 1.0]
    for volume in volumes:
        print(f"\nTesting volume: {volume}")
        voice_manager.setup_voice("alice", rate=1.0, volume=volume)
        voice_manager.speak(f"This is a test at {volume} volume.")
    
    print("\n✅ All Twilio voice tests completed!")
    return True

def test_voice_comparison():
    """Compare Twilio vs pyttsx3 voice synthesis"""
    print("\n🔄 Voice Synthesis Comparison")
    print("=" * 50)
    
    # Test Twilio
    print("🎤 Testing Twilio Voice Synthesis:")
    twilio_manager = create_twilio_voice()
    if twilio_manager.is_enabled:
        twilio_manager.setup_voice("alice", rate=1.0, volume=1.0)
        twilio_manager.speak("This is Twilio voice synthesis.")
        print("   ✅ Twilio voice working")
    else:
        print("   ❌ Twilio voice not available")
    
    # Test pyttsx3
    print("\n🎤 Testing pyttsx3 Voice Synthesis:")
    try:
        import pyttsx3
        tts = pyttsx3.init()
        tts.say("This is pyttsx3 voice synthesis.")
        tts.runAndWait()
        print("   ✅ pyttsx3 voice working")
    except Exception as e:
        print(f"   ❌ pyttsx3 voice error: {e}")
    
    print("\n📊 Comparison Summary:")
    print("   Twilio: High-quality, cloud-based, requires credentials")
    print("   pyttsx3: Local, system-dependent, no credentials needed")

def show_usage_instructions():
    """Show how to use the Twilio voice option"""
    print("\n📖 Usage Instructions")
    print("=" * 50)
    
    print("1. Set up Twilio credentials:")
    print("   export TWILIO_SID='your_account_sid'")
    print("   export TWILIO_AUTH_TOKEN='your_auth_token'")
    print()
    
    print("2. Set voice type environment variable:")
    print("   export VOICE_TYPE='twilio'  # Use Twilio voice synthesis")
    print("   export VOICE_TYPE='pyttsx3' # Use pyttsx3 voice synthesis (default)")
    print()
    
    print("3. Run the voice agent:")
    print("   python main_twilio.py [phone_number]")
    print()
    
    print("4. Available Twilio voices:")
    voice_manager = create_twilio_voice()
    if voice_manager.is_enabled:
        voices = voice_manager.tts.get_available_voices()
        for voice_id, voice_info in list(voices.items())[:10]:  # Show first 10
            print(f"   - {voice_id}: {voice_info['name']} ({voice_info['gender']})")
        print("   ... and more")
    print()
    
    print("5. Voice settings:")
    print("   - Rate: 0.5 to 2.0 (speech speed)")
    print("   - Volume: 0.0 to 1.0 (volume level)")
    print("   - Language: en-US, en-GB, etc.")

def main():
    """Main test function"""
    print("🎤 Twilio Voice Synthesis Test Suite")
    print("=" * 60)
    
    # Check environment variables
    twilio_sid = os.getenv("TWILIO_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    print(f"🔑 Environment Check:")
    print(f"   TWILIO_SID: {'✅ Set' if twilio_sid else '❌ Not set'}")
    print(f"   TWILIO_AUTH_TOKEN: {'✅ Set' if twilio_token else '❌ Not set'}")
    print()
    
    # Run tests
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        test_voice_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "help":
        show_usage_instructions()
    else:
        test_twilio_voice()
        show_usage_instructions()
    
    print("\n🎯 Next Steps:")
    print("1. Set your Twilio credentials")
    print("2. Run: python test_twilio_voice.py")
    print("3. Run: python main_twilio.py [phone_number]")
    print("4. Enjoy high-quality voice synthesis!")

if __name__ == "__main__":
    main() 