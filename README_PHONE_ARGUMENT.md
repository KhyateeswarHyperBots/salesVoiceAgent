# Voice Agent with Phone Number Argument

## Overview

The voice agent now supports accepting a client's phone number as a command line argument. This allows for immediate client identification and personalized conversations without waiting for voice input.

## Features

### 🎯 Command Line Phone Number Support
- Accept phone number as `--phone` argument
- Immediate client identification on startup
- Enhanced web data fetching for personalized context

### 🌐 Enhanced Web Data Integration
- Fetches company information and market trends
- Provides real-time context for sales conversations
- Integrates web data with RAG knowledge base

### 👤 Personalized Client Experience
- Automatic client identification from phone number
- Company and role-specific conversation context
- Enhanced sales approach based on client profile

## Usage

### Basic Usage with Phone Number

```bash
# Start agent with specific client phone number
python main.py --phone +1-555-0123

# Start agent with different client
python main.py --phone +1-555-0456
```

### Interactive Mode (No Phone Number)

```bash
# Start agent and provide phone number via voice
python main.py
```

### Help

```bash
# Show available options
python main.py --help
```

## Available Client Phone Numbers

Based on `clients.json`:

| Phone Number | Client Name | Company | Title |
|--------------|-------------|---------|-------|
| +1-555-0123 | John Smith | TechCorp | CTO |
| +1-555-0456 | Sarah Johnson | InnovateAI | CEO |
| +1-555-0789 | Michael Brown | DataFlow | VP Engineering |
| +1-555-0321 | Emily Davis | CloudTech | Director of IT |

## How It Works

### 1. Phone Number Processing
```
Command Line → Phone Number → Client RAG Search → Client Identification
```

### 2. Web Data Enhancement
```
Client Found → Web Search → Company Info → Market Trends → Enhanced Context
```

### 3. Personalized Conversation
```
Enhanced Context → RAG Knowledge → Conversation History → AI Response
```

## Technical Implementation

### Command Line Arguments
```python
parser = argparse.ArgumentParser(description="Sales Executive AI Assistant")
parser.add_argument("--phone", type=str, help="Client's phone number for personalized assistance")
```

### Client Identification Flow
1. **Phone Number Input**: From command line or voice input
2. **RAG Search**: Search client database by phone number
3. **Web Data Fetch**: Get company and market information
4. **Context Enhancement**: Combine all data for AI context
5. **Personalized Response**: Generate client-specific responses

### Web Data Sources (Simulated)
- Company information and industry
- Recent news and developments
- Market trends and competitive landscape
- Professional profiles and roles

## Testing

### Run Test Script
```bash
python test_phone_argument.py
```

### Manual Testing
```bash
# Test with John Smith
python main.py --phone +1-555-0123

# Test with Sarah Johnson
python main.py --phone +1-555-0456
```

## Expected Behavior

### With Phone Number Argument
1. ✅ Agent starts immediately
2. ✅ Client identified from phone number
3. ✅ Web data fetched and displayed
4. ✅ Personalized welcome message
5. ✅ Enhanced conversation context

### Without Phone Number
1. ✅ Agent asks for phone number
2. ✅ Voice input processed
3. ✅ Client search performed
4. ✅ Same enhanced experience

## Error Handling

### Invalid Phone Number
- Agent continues with general assistance
- No interruption to conversation flow
- Clear error messaging

### Missing Client Data
- Graceful fallback to general mode
- Maintains conversation quality
- Logs missing data for review

## Benefits

### 🚀 Immediate Personalization
- No waiting for voice input
- Instant client recognition
- Faster conversation startup

### 📊 Enhanced Context
- Real-time web data integration
- Market-aware responses
- Company-specific insights

### 🎯 Better Sales Approach
- Role-based conversation strategy
- Industry-specific solutions
- Personalized value propositions

## Configuration

### Voice Settings
- Karen voice (American female)
- 155 WPM speech rate
- 85% volume
- Interruptible speech

### RAG Integration
- Document knowledge base
- Client database
- Web data enhancement
- Conversation history

## Troubleshooting

### Common Issues

1. **Phone Number Not Found**
   - Check `clients.json` for valid numbers
   - Verify phone number format
   - Use test script for validation

2. **Web Data Not Loading**
   - Check internet connection
   - Verify web search simulation
   - Review error logs

3. **Voice Recognition Issues**
   - Adjust microphone settings
   - Check ambient noise
   - Verify speech recognition setup

### Debug Mode
```bash
# Run with verbose output
python main.py --phone +1-555-0123 --debug
```

## Future Enhancements

### Planned Features
- Real web API integration
- Advanced client analytics
- Multi-language support
- Enhanced voice customization

### Integration Opportunities
- CRM system integration
- Calendar scheduling
- Email follow-up automation
- Sales pipeline tracking 