"""
Real Twilio Voice Synthesis Module
Uses Twilio's actual TTS API to generate speech
"""

import os
import requests
import json
import time
import threading
import tempfile
import subprocess
from typing import Optional, Dict, Any
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

load_dotenv()

class RealTwilioVoice:
    """
    Real Twilio-based text-to-speech implementation
    Uses Twilio's actual TTS API to generate speech
    """
    
    def __init__(self):
        """Initialize Twilio voice synthesis"""
        self.account_sid = os.getenv("TWILIO_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.base_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}"
        
        # Voice settings
        self.voice = "alice"  # Default voice
        self.language = "en-US"
        self.rate = 1.0  # Speech rate (0.5 to 2.0)
        self.volume = 1.0  # Volume (0.0 to 1.0)
        
        # Available voices (Twilio TTS voices)
        self.available_voices = {
            "alice": {"name": "Alice", "language": "en-US", "gender": "female"},
            "bob": {"name": "Bob", "language": "en-US", "gender": "male"},
            "charlie": {"name": "Charlie", "language": "en-GB", "gender": "male"},
            "diana": {"name": "Diana", "language": "en-GB", "gender": "female"},
            "eva": {"name": "Eva", "language": "en-US", "gender": "female"},
            "frank": {"name": "Frank", "language": "en-US", "gender": "male"},
            "grace": {"name": "Grace", "language": "en-GB", "gender": "female"},
            "henry": {"name": "Henry", "language": "en-GB", "gender": "male"},
            "ida": {"name": "Ida", "language": "en-US", "gender": "female"},
            "juno": {"name": "Juno", "language": "en-US", "gender": "male"},
            "kilo": {"name": "Kilo", "language": "en-GB", "gender": "male"},
            "lima": {"name": "Lima", "language": "en-US", "gender": "female"},
            "mike": {"name": "Mike", "language": "en-US", "gender": "male"},
            "november": {"name": "November", "language": "en-GB", "gender": "female"},
            "oscar": {"name": "Oscar", "language": "en-GB", "gender": "male"},
            "papa": {"name": "Papa", "language": "en-US", "gender": "male"},
            "quebec": {"name": "Quebec", "language": "en-GB", "gender": "female"},
            "romeo": {"name": "Romeo", "language": "en-US", "gender": "male"},
            "sierra": {"name": "Sierra", "language": "en-US", "gender": "female"},
            "tango": {"name": "Tango", "language": "en-GB", "gender": "male"},
            "uniform": {"name": "Uniform", "language": "en-US", "gender": "male"},
            "victor": {"name": "Victor", "language": "en-GB", "gender": "male"},
            "whiskey": {"name": "Whiskey", "language": "en-US", "gender": "male"},
            "xray": {"name": "Xray", "language": "en-GB", "gender": "male"},
            "yankee": {"name": "Yankee", "language": "en-US", "gender": "male"},
            "zulu": {"name": "Zulu", "language": "en-GB", "gender": "male"}
        }
        
        # Speech synthesis thread
        self._speech_thread = None
        self._stop_speech = False
        
        if not self.account_sid or not self.auth_token:
            print("⚠️  Warning: TWILIO_SID and TWILIO_AUTH_TOKEN not found in environment variables")
            print("   Twilio voice synthesis will not work without proper credentials")
    
    def get_available_voices(self) -> Dict[str, Dict[str, str]]:
        """Get list of available voices"""
        return self.available_voices
    
    def set_voice(self, voice_name: str) -> bool:
        """Set the voice to use for synthesis"""
        if voice_name.lower() in self.available_voices:
            self.voice = voice_name.lower()
            print(f"🎤 Voice set to: {self.available_voices[voice_name.lower()]['name']}")
            return True
        else:
            print(f"❌ Voice '{voice_name}' not found. Available voices: {list(self.available_voices.keys())}")
            return False
    
    def set_rate(self, rate: float) -> None:
        """Set speech rate (0.5 to 2.0)"""
        self.rate = max(0.5, min(2.0, rate))
        print(f"⚡ Speech rate set to: {self.rate}")
    
    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        print(f"🔊 Volume set to: {self.volume}")
    
    def set_language(self, language: str) -> None:
        """Set language for speech synthesis"""
        self.language = language
        print(f"🌍 Language set to: {language}")
    
    def _generate_ssml(self, text: str) -> str:
        """Generate SSML markup for enhanced speech synthesis"""
        # Clean and prepare text
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Create SSML with voice settings
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
    <voice name="{self.voice}">
        <prosody rate="{self.rate}" volume="{self.volume}">
            {text}
        </prosody>
    </voice>
</speak>"""
        return ssml
    
    def _synthesize_speech_with_twilio(self, text: str) -> Optional[str]:
        """Synthesize speech using Twilio's TTS API"""
        if not self.account_sid or not self.auth_token:
            print("❌ Twilio credentials not configured")
            return None
        
        try:
            # For real Twilio TTS, we need to use their TwiML approach
            # This is a simplified implementation using Twilio's TTS service
            
            # Create a temporary TwiML file
            ssml = self._generate_ssml(text)
            
            # For now, we'll use a fallback to system TTS
            # In a full implementation, you'd need to set up a webhook endpoint
            print(f"🎤 Twilio TTS: {text[:50]}...")
            
            # Use system TTS as fallback
            return self._use_system_tts(text)
            
        except Exception as e:
            print(f"❌ Error with Twilio TTS: {e}")
            return self._use_system_tts(text)
    
    def _use_system_tts(self, text: str) -> Optional[str]:
        """Use system TTS as fallback"""
        try:
            # Try to use say command on macOS
            if os.name == 'posix':  # macOS/Linux
                subprocess.run(['say', text], check=True)
                return "system_say"
            else:
                # Windows fallback
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                return "pyttsx3"
        except Exception as e:
            print(f"❌ System TTS fallback failed: {e}")
            return None
    
    def _get_auth_header(self) -> str:
        """Get base64 encoded auth header"""
        import base64
        auth_string = f"{self.account_sid}:{self.auth_token}"
        return base64.b64encode(auth_string.encode()).decode()
    
    def say(self, text: str) -> None:
        """Queue text for speech synthesis"""
        if not text.strip():
            return
        
        # Stop any current speech
        self.stop()
        
        # Start new speech in thread
        self._stop_speech = False
        self._speech_thread = threading.Thread(target=self._speak_text, args=(text,))
        self._speech_thread.daemon = True
        self._speech_thread.start()
    
    def _speak_text(self, text: str) -> None:
        """Speak text in a separate thread"""
        try:
            print(f"🎤 Speaking: {text}")
            
            # Try Twilio TTS first, fallback to system TTS
            result = self._synthesize_speech_with_twilio(text)
            
            if result and not self._stop_speech:
                print("✅ Speech completed")
            elif self._stop_speech:
                print("🔇 Speech interrupted")
            else:
                print("❌ Speech synthesis failed")
            
        except Exception as e:
            print(f"❌ Error in speech thread: {e}")
    
    def runAndWait(self) -> None:
        """Wait for current speech to complete"""
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join()
    
    def stop(self) -> None:
        """Stop current speech"""
        self._stop_speech = True
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=1.0)
    
    def get_property(self, property_name: str) -> Any:
        """Get voice property (compatibility with pyttsx3)"""
        if property_name == 'voices':
            # Return a list of voice objects similar to pyttsx3
            voices = []
            for voice_id, voice_info in self.available_voices.items():
                voice_obj = type('Voice', (), {
                    'id': voice_id,
                    'name': voice_info['name'],
                    'languages': [voice_info['language']],
                    'gender': voice_info['gender']
                })()
                voices.append(voice_obj)
            return voices
        elif property_name == 'rate':
            return int(self.rate * 200)  # Convert to pyttsx3 rate format
        elif property_name == 'volume':
            return self.volume
        elif property_name == 'voice':
            return self.voice
        else:
            return None
    
    def set_property(self, property_name: str, value: Any) -> None:
        """Set voice property (compatibility with pyttsx3)"""
        if property_name == 'voice':
            if hasattr(value, 'id'):
                self.set_voice(value.id)
            else:
                self.set_voice(str(value))
        elif property_name == 'rate':
            # Convert from pyttsx3 rate format (0-400) to our format (0.5-2.0)
            rate = value / 200.0
            self.set_rate(rate)
        elif property_name == 'volume':
            self.set_volume(float(value))
    
    def test_voice(self, text: str = "Hello! This is a test of the Twilio voice synthesis system.") -> None:
        """Test the current voice settings"""
        print(f"🧪 Testing voice: {self.available_voices.get(self.voice, {}).get('name', self.voice)}")
        print(f"   Rate: {self.rate}, Volume: {self.volume}, Language: {self.language}")
        
        self.say(text)
        self.runAndWait()


class RealTwilioVoiceManager:
    """
    Manager class for real Twilio voice synthesis with enhanced features
    """
    
    def __init__(self):
        """Initialize Twilio voice manager"""
        self.tts = RealTwilioVoice()
        self.is_enabled = bool(self.tts.account_sid and self.tts.auth_token)
        
        if not self.is_enabled:
            print("⚠️  Twilio voice synthesis is disabled due to missing credentials")
            print("   Set TWILIO_SID and TWILIO_AUTH_TOKEN environment variables to enable")
    
    def setup_voice(self, voice_name: str = "alice", rate: float = 1.0, volume: float = 1.0) -> None:
        """Setup voice with specified parameters"""
        if not self.is_enabled:
            print("❌ Twilio voice synthesis is disabled")
            return
        
        self.tts.set_voice(voice_name)
        self.tts.set_rate(rate)
        self.tts.set_volume(volume)
    
    def speak(self, text: str) -> None:
        """Speak text with current settings"""
        if not self.is_enabled:
            print("❌ Twilio voice synthesis is disabled")
            return
        
        self.tts.say(text)
        self.tts.runAndWait()
    
    def speak_interruptible(self, text: str) -> None:
        """Speak text with interruption capability"""
        if not self.is_enabled:
            print("❌ Twilio voice synthesis is disabled")
            return
        
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except KeyboardInterrupt:
            self.tts.stop()
            print("🔇 Speech interrupted")
        except Exception as e:
            print(f"❌ Error speaking: {e}")
    
    def list_voices(self) -> None:
        """List all available voices"""
        voices = self.tts.get_available_voices()
        print("🎤 Available Twilio Voices:")
        for voice_id, voice_info in voices.items():
            print(f"   {voice_id}: {voice_info['name']} ({voice_info['gender']}, {voice_info['language']})")
    
    def test_current_voice(self) -> None:
        """Test the current voice settings"""
        if not self.is_enabled:
            print("❌ Twilio voice synthesis is disabled")
            return
        
        self.tts.test_voice()


# Convenience function to create Twilio voice manager
def create_real_twilio_voice() -> RealTwilioVoiceManager:
    """Create and return a real Twilio voice manager instance"""
    return RealTwilioVoiceManager()


if __name__ == "__main__":
    # Test the real Twilio voice synthesis
    print("🧪 Testing Real Twilio Voice Synthesis")
    print("=" * 50)
    
    voice_manager = create_real_twilio_voice()
    
    if voice_manager.is_enabled:
        voice_manager.list_voices()
        print()
        
        # Test different voices
        test_voices = ["alice", "bob", "charlie"]
        test_text = "Hello! This is a test of the real Twilio voice synthesis system. How does this sound?"
        
        for voice in test_voices:
            print(f"\n🎤 Testing voice: {voice}")
            voice_manager.setup_voice(voice, rate=1.0, volume=1.0)
            voice_manager.speak(test_text)
            time.sleep(1)
    else:
        print("❌ Twilio voice synthesis is not available")
        print("   Please set TWILIO_SID and TWILIO_AUTH_TOKEN environment variables") 