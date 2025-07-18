#!/usr/bin/env python3
"""
Configuration Loader
Loads environment variables from config.env file
"""

import os
import sys

def load_config():
    """Load configuration from config.env file"""
    config_file = "config.env"
    
    if not os.path.exists(config_file):
        print(f"⚠️  Config file {config_file} not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Set environment variable
                    os.environ[key] = value
                    print(f"✅ Loaded: {key}")
        
        print(f"🎯 Configuration loaded from {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def show_config():
    """Show current configuration"""
    print("\n📋 Current Configuration:")
    print("=" * 50)
    
    config_vars = [
        'TWILIO_SID',
        'TWILIO_AUTH_TOKEN', 
        'TWILIO_FROM_NUMBER',
        'TO_PHONE_NUMBER',
        'VOICE_TYPE',
        'VOICE_RATE',
        'VOICE_VOLUME',
        'VOICE_LANGUAGE',
        'OLLAMA_MODEL',
        'OPENAI_API_KEY'
    ]
    
    for var in config_vars:
        value = os.getenv(var, 'Not set')
        if var in ['TWILIO_AUTH_TOKEN', 'OPENAI_API_KEY'] and value != 'Not set':
            # Mask sensitive values
            value = value[:8] + '...' if len(value) > 8 else '***'
        print(f"   {var}: {value}")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_config()
        return
    
    print("🔧 Loading configuration...")
    if load_config():
        show_config()
        print("\n✅ Configuration loaded successfully!")
        print("   You can now run: python main_twilio.py [phone_number]")
    else:
        print("❌ Failed to load configuration")

if __name__ == "__main__":
    main() 