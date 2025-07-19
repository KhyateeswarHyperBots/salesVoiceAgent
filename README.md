# Sales Voice Agent

A Python-based sales voice agent that can make real-time phone calls with advanced AI features including RAG, sentiment analysis, web scraping, and client data integration.

## Features

- **Real-time Phone Calls**: Make outbound calls using Twilio
- **AI-Powered Conversations**: Uses Ollama (llama3.2:3b) for intelligent responses
- **RAG (Retrieval Augmented Generation)**: Context-aware responses using knowledge base
- **Sentiment Analysis**: Real-time sentiment tracking and buying signal detection
- **Web Scraping**: Company research and website data extraction
- **Client Data Integration**: Personalized conversations using client information
- **Buying Probability Tracking**: Dynamic scoring based on conversation signals

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   - Copy `config.env.example` to `config.env`
   - Add your Twilio credentials and other settings

3. **Start Ollama**:
   ```bash
   ollama serve
   ```

4. **Start ngrok** (in a separate terminal):
   ```bash
   ngrok http 5002
   ```

5. **Run the agent**:
   ```bash
   python make_call.py
   ```

## Configuration

The agent uses the following configuration in `config.env`:

- `TWILIO_SID`: Your Twilio Account SID
- `TWILIO_AUTH_TOKEN`: Your Twilio Auth Token
- `TWILIO_FROM_NUMBER`: Your Twilio phone number
- `OLLAMA_MODEL`: AI model to use (default: llama3.2:3b)
- `OPENAI_API_KEY`: For web search functionality

## Usage

1. Run `python make_call.py`
2. Enter a client phone number for data lookup when prompted
3. The agent will make a call to the test number (+918688164030)
4. Use the client data from your input number for personalized conversation

## Architecture

- **`make_call.py`**: Main phone agent with full architecture integration
- **`main.py`**: Original voice agent architecture (reference)
- **`client_rag_system.py`**: Client data management and RAG
- **`company_website_scraper.py`**: Web scraping functionality
- **`comprehensive_company_research.py`**: Company research
- **`chatgpt_web_search.py`**: Web search integration
- **`agent_config.py`**: Agent configuration and system instructions

## Data Files

- **`clients.json`**: Client database with contact information
- **`documents.json`**: Knowledge base for RAG
- **`embeddings.pkl` & `vector.index`**: RAG index files
- **`client_embeddings.pkl` & `client_vector.index`**: Client RAG index

## Security & Setup

### 🔒 Important Security Notes

**NEVER commit sensitive files to git!** The following files are automatically ignored:
- `config.env` - Contains your API keys and tokens
- `credentials.json` - Google Calendar credentials
- `models/` - Large AI model files
- `conversations/` - Call recordings and logs
- `*.pkl`, `*.index` - Embeddings and vector files

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd salesVoiceAgent
   ```

2. **Create your configuration**:
   ```bash
   cp config.env.example config.env
   # Edit config.env with your real API keys and tokens
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama**:
   ```bash
   ollama pull llama3.2:3b
   ollama serve
   ```

5. **Set up ngrok** (for webhook tunneling):
   ```bash
   ngrok http 5002
   ```

## Requirements

- Python 3.8+
- Twilio account
- Ollama with llama3.2:3b model
- ngrok for webhook tunneling
- Internet connection for web scraping and search 