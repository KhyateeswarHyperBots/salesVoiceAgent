# Combined Web Data System for Sales Voice Agent

## Overview

The sales voice agent now integrates **two powerful web data sources** to provide comprehensive, real-time intelligence during sales conversations:

1. **ChatGPT Web Search** - Real-time market intelligence and company research
2. **Website Scraping** - Direct extraction of current company information from their websites

## System Components

### 1. ChatGPT Web Search (`comprehensive_company_research.py`)
- **Real-time market intelligence** using OpenAI's GPT-4
- **Comprehensive company research** including:
  - Company overview and financial data
  - Industry analysis and market position
  - Recent news and trigger events
  - Key decision makers and their backgrounds
  - Sales messaging angles and pain points
  - Intent scoring and opportunity assessment

### 2. Website Scraper (`company_website_scraper.py`)
- **Direct website data extraction** using BeautifulSoup
- **Comprehensive data collection** including:
  - Company name, tagline, and description
  - Contact information (email, phone, address)
  - Products and services offered
  - Team member information
  - Social media presence
  - Technology stack used
  - Recent news and blog articles
  - Meta data and structured data

### 3. Client RAG System (`client_rag_system.py`)
- **Client identification** by phone number or conversation context
- **Domain extraction** from client database
- **Enhanced conversation context** with client-specific information

## How It Works

### 1. Client Identification
```python
# Client is identified by phone number or conversation
client = client_rag.search_client_by_phone("14043607955")
# Returns: Tom Hanzlick from Answerthink Consulting Group
# Domain: answerthink.com
```

### 2. Dual Data Collection
```python
# ChatGPT provides market intelligence
research_data = comprehensive_research.research_company(client)
# Returns: Industry, revenue, employees, recent news, decision makers

# Website scraper provides current company info
scraped_data = website_scraper.scrape_company_website(client['Domain'])
# Returns: Tagline, products, technologies, contact info
```

### 3. Combined Context Generation
```python
# Both data sources are combined for comprehensive context
combined_context = f"""
ChatGPT Research:
- Company: Answerthink Consulting Group
- Industry: Information Technology and Services
- Revenue: Not disclosed
- Employees: 501-1000

Website Data:
- Company: Answerthink
- Tagline: Looking for a First-Class Business Consultant?
- Technologies: Google Analytics, WordPress
"""
```

## Benefits

### 🎯 **Highly Personalized Conversations**
- Agent knows client's exact role, company, and industry
- Real-time market intelligence for relevant discussions
- Current company information for accurate references

### 📊 **Comprehensive Intelligence**
- **ChatGPT**: Market trends, financial data, recent news, decision makers
- **Website Scraping**: Current products, technologies, team, contact info
- **Combined**: Complete picture for informed sales conversations

### ⚡ **Real-time Data**
- Both systems fetch current information
- No outdated or stale data
- Always relevant to current market conditions

### 🔄 **Automatic Integration**
- Seamlessly integrated into voice agent
- No manual intervention required
- Data automatically included in conversation context

## Usage Examples

### Example 1: Client with Accessible Website
```
Client: Tom Hanzlick (Answerthink Consulting Group)
Domain: answerthink.com

ChatGPT Data:
- Industry: Information Technology and Services
- Employees: 501-1000
- Market position and recent news

Website Data:
- Tagline: "Looking for a First-Class Business Consultant?"
- Technologies: Google Analytics, WordPress
- Current services and offerings
```

### Example 2: Client with Protected Website
```
Client: Will Blue (Vanguard Truck Centers)
Domain: vanguardtrucks.com

ChatGPT Data:
- Industry: Automotive
- Revenue: Not publicly disclosed
- Employees: 500-1000
- Market intelligence and pain points

Website Data:
- Blocked by 403 error
- Falls back to ChatGPT data only
- Still provides comprehensive intelligence
```

## Technical Features

### Website Scraper Capabilities
- **Anti-bot protection handling** with multiple user agents
- **Fallback methods** for blocked websites
- **Comprehensive data extraction** from various page elements
- **Error handling** and graceful degradation

### ChatGPT Integration
- **Real-time API calls** to OpenAI
- **Structured data extraction** with JSON responses
- **Comprehensive research templates** for different data types
- **Error handling** and retry logic

### Voice Agent Integration
- **Automatic client identification** by phone number
- **Seamless data combination** in conversation context
- **Real-time data fetching** during conversations
- **Persistent data storage** for conversation history

## Files Created

1. **`company_website_scraper.py`** - Main website scraping functionality
2. **`test_website_scraper.py`** - Test script for website scraping
3. **`test_combined_web_data.py`** - Test script for combined functionality
4. **`COMBINED_WEB_DATA_README.md`** - This documentation

## Testing

Run the test scripts to see the system in action:

```bash
# Test website scraping only
python test_website_scraper.py

# Test combined web data system
python test_combined_web_data.py
```

## Future Enhancements

1. **More sophisticated anti-bot detection** for protected websites
2. **Caching system** to reduce API calls and improve performance
3. **Additional data sources** (LinkedIn, Crunchbase, etc.)
4. **Machine learning** for better data extraction and analysis
5. **Real-time alerts** for trigger events and opportunities

## Conclusion

The combined web data system provides the sales voice agent with unprecedented intelligence capabilities. By combining real-time market research with current website data, the agent can conduct highly informed, personalized sales conversations that are relevant, accurate, and effective. 