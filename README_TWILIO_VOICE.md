# Twilio Voice Synthesis Option

This document explains how to use the Twilio voice synthesis alternative to pyttsx3 in the Sales Voice Agent.

## Overview

The Sales Voice Agent now supports two voice synthesis options:

1. **pyttsx3** (default) - Local system voice synthesis
2. **Twilio** - High-quality cloud-based voice synthesis

## Features

### Twilio Voice Synthesis Benefits

- **High Quality**: Professional-grade voice synthesis
- **Multiple Voices**: 26+ different voices (Alice, Bob, Charlie, etc.)
- **Language Support**: Multiple languages and accents
- **Consistent**: Same voice quality across all platforms
- **Customizable**: Adjustable rate, volume, and language settings

### Available Twilio Voices

| Voice ID | Name | Gender | Language |
|----------|------|--------|----------|
| alice | Alice | Female | en-US |
| bob | Bob | Male | en-US |
| charlie | Charlie | Male | en-GB |
| diana | Diana | Female | en-GB |
| eva | Eva | Female | en-US |
| frank | Frank | Male | en-US |
| grace | Grace | Female | en-GB |
| henry | Henry | Male | en-GB |
| ida | Ida | Female | en-US |
| juno | Juno | Male | en-US |
| kilo | Kilo | Male | en-GB |
| lima | Lima | Female | en-US |
| mike | Mike | Male | en-US |
| november | November | Female | en-GB |
| oscar | Oscar | Male | en-GB |
| papa | Papa | Male | en-US |
| quebec | Quebec | Female | en-GB |
| romeo | Romeo | Male | en-US |
| sierra | Sierra | Female | en-US |
| tango | Tango | Male | en-GB |
| uniform | Uniform | Male | en-US |
| victor | Victor | Male | en-GB |
| whiskey | Whiskey | Male | en-US |
| xray | Xray | Male | en-GB |
| yankee | Yankee | Male | en-US |
| zulu | Zulu | Male | en-GB |

## Setup Instructions

### 1. Get Twilio Credentials

1. Sign up for a free Twilio account at [https://www.twilio.com/](https://www.twilio.com/)
2. Go to your Twilio Console: [https://console.twilio.com/](https://console.twilio.com/)
3. Find your Account SID and Auth Token
4. Note: The free tier includes limited usage

### 2. Set Environment Variables

```bash
# Set Twilio credentials
export TWILIO_SID="your_account_sid_here"
export TWILIO_AUTH_TOKEN="your_auth_token_here"

# Choose voice synthesis type
export VOICE_TYPE="twilio"    # Use Twilio voice synthesis
export VOICE_TYPE="pyttsx3"   # Use pyttsx3 voice synthesis (default)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Test the Setup

```bash
# Test Twilio voice synthesis
python test_twilio_voice.py

# Compare both voice options
python test_twilio_voice.py compare

# Show usage instructions
python test_twilio_voice.py help
```

## Usage

### Running with Twilio Voice

```bash
# Run with Twilio voice synthesis
export VOICE_TYPE="twilio"
python main_twilio.py [phone_number]

# Example with client phone number
python main_twilio.py 14043607955
```

### Running with pyttsx3 Voice (Default)

```bash
# Run with pyttsx3 voice synthesis
export VOICE_TYPE="pyttsx3"
python main_twilio.py [phone_number]

# Or simply (pyttsx3 is default)
python main_twilio.py [phone_number]
```

## Configuration

### Voice Settings

You can customize voice settings in the code:

```python
# In main_twilio.py
VOICE_SETTINGS = {
    'rate': 180,      # Words per minute
    'volume': 0.85    # Volume level (0.0 to 1.0)
}
```

### Twilio Voice Settings

```python
# For Twilio voices
voice_manager.setup_voice(
    voice_name="alice",  # Voice ID
    rate=1.0,           # Speech rate (0.5 to 2.0)
    volume=1.0          # Volume (0.0 to 1.0)
)
```

## File Structure

```
salesVoiceAgent/
├── main_twilio.py          # Main agent with Twilio voice option
├── twilio_voice.py         # Twilio voice synthesis module
├── test_twilio_voice.py    # Test script for Twilio voice
├── main.py                 # Original agent (pyttsx3 only)
├── requirements.txt        # Updated dependencies
└── README_TWILIO_VOICE.md # This file
```

## Troubleshooting

### Common Issues

1. **"Twilio voice synthesis is not available"**
   - Check that TWILIO_SID and TWILIO_AUTH_TOKEN are set
   - Verify your Twilio credentials are correct
   - Ensure you have an active Twilio account

2. **"Import dotenv could not be resolved"**
   - Install python-dotenv: `pip install python-dotenv`

3. **"Twilio API errors"**
   - Check your Twilio account balance
   - Verify API permissions
   - Check rate limits

### Testing Voice Options

```bash
# Test both voice systems
python test_twilio_voice.py compare

# Test only Twilio
python test_twilio_voice.py

# Check environment setup
python test_twilio_voice.py help
```

## Cost Considerations

### Twilio Pricing

- **Free Tier**: Limited usage included
- **Pay-as-you-go**: Pay per usage after free tier
- **Voice Synthesis**: Typically $0.0005 per character
- **Check**: [Twilio Pricing](https://www.twilio.com/pricing) for current rates

### pyttsx3 (Free)

- No cost associated
- Uses local system resources
- Quality depends on system voice engines

## Migration Guide

### From pyttsx3 to Twilio

1. Set environment variables:
   ```bash
   export TWILIO_SID="your_sid"
   export TWILIO_AUTH_TOKEN="your_token"
   export VOICE_TYPE="twilio"
   ```

2. Run the new agent:
   ```bash
   python main_twilio.py [phone_number]
   ```

3. Test the setup:
   ```bash
   python test_twilio_voice.py
   ```

### Back to pyttsx3

```bash
export VOICE_TYPE="pyttsx3"
python main_twilio.py [phone_number]
```

## Advanced Usage

### Custom Voice Configuration

```python
# In your code
from twilio_voice import create_twilio_voice

voice_manager = create_twilio_voice()
voice_manager.setup_voice("charlie", rate=1.2, volume=0.9)
voice_manager.speak("Custom voice configuration test.")
```

### Voice Switching During Runtime

```python
# Switch voices during conversation
voice_manager.setup_voice("alice")  # Switch to Alice
voice_manager.speak("Hello from Alice!")

voice_manager.setup_voice("bob")    # Switch to Bob
voice_manager.speak("Hello from Bob!")
```

## Support

For issues with:
- **Twilio Voice**: Check Twilio documentation and support
- **Agent Integration**: Check the main project documentation
- **Setup Issues**: Run `python test_twilio_voice.py help`

## Notes

- The Twilio voice synthesis is currently simulated for demonstration
- Real Twilio TTS requires webhook endpoints and call setup
- This implementation provides a foundation for full Twilio integration
- Consider costs when using Twilio in production environments 