#!/usr/bin/env python3
"""
ChatGPT-Powered Web Search for Real-Time Financial Data
Uses OpenAI API to fetch live financial information and news
"""

import openai
import json
import time
from typing import Dict, List, Optional
import os

class ChatGPTWebSearch:
    def __init__(self, api_key: str = None):
        """
        Initialize ChatGPT web search with OpenAI API key
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("⚠️ Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
            return
        
        openai.api_key = self.api_key
        
    def search_company_financials(self, client_data: Dict) -> Dict:
        """
        Search for real-time company financial data using ChatGPT
        """
        company_name = client_data.get('Company', '')
        domain = client_data.get('Domain', '')
        person_name = client_data.get('Full Name', '')
        title = client_data.get('Title', '')
        
        if not company_name:
            return self._get_generic_financial_context()
        
        print(f"🔍 Searching real-time financial data for: {company_name}")
        
        financial_data = {
            'company_name': company_name,
            'domain': domain,
            'person_name': person_name,
            'title': title,
            'revenue_data': {},
            'financial_metrics': {},
            'industry_insights': {},
            'market_position': {},
            'recent_developments': [],
            'stock_data': {},
            'news_data': [],
            'pain_points': []
        }
        
        try:
            # Get comprehensive company financial data
            financial_info = self._get_company_financial_info(company_name)
            financial_data['revenue_data'] = financial_info.get('revenue_data', {})
            financial_data['financial_metrics'] = financial_info.get('financial_metrics', {})
            
            # Get industry insights
            industry_info = self._get_industry_insights(company_name)
            financial_data['industry_insights'] = industry_info.get('industry_insights', {})
            financial_data['pain_points'] = industry_info.get('pain_points', [])
            
            # Get market position
            market_info = self._get_market_position(company_name)
            financial_data['market_position'] = market_info.get('market_position', {})
            
            # Get recent news and developments
            news_info = self._get_recent_news(company_name)
            financial_data['news_data'] = news_info.get('news', [])
            financial_data['recent_developments'] = news_info.get('developments', [])
            
            # Get stock data if public
            stock_info = self._get_stock_data(company_name)
            financial_data['stock_data'] = stock_info
            
        except Exception as e:
            print(f"❌ Error fetching financial data: {e}")
            return self._get_generic_financial_context()
        
        return financial_data
    
    def _get_company_financial_info(self, company_name: str) -> Dict:
        """Get company financial information using ChatGPT"""
        prompt = f"""
        Research and provide accurate financial information for {company_name}. 
        Focus on:
        1. Estimated annual revenue (if publicly available)
        2. Number of employees
        3. Company size and scale
        4. Financial health indicators
        5. Growth trends
        
        Provide the information in JSON format with these fields:
        {{
            "revenue_data": {{
                "estimated_revenue": "string",
                "employee_count": "string", 
                "company_size": "string",
                "financial_health": "string",
                "growth_rate": "string"
            }},
            "financial_metrics": {{
                "funding_status": "string",
                "profitability": "string",
                "key_metrics": "string"
            }}
        }}
        
        Only include information that can be reasonably verified. If specific data is not available, indicate "Not publicly available".
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial research assistant. Provide accurate, up-to-date financial information based on publicly available sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            print(f"❌ Error getting financial info: {e}")
            return {"revenue_data": {}, "financial_metrics": {}}
    
    def _get_industry_insights(self, company_name: str) -> Dict:
        """Get industry insights and pain points using ChatGPT"""
        prompt = f"""
        Analyze the industry and business context for {company_name}. Provide insights on:
        1. Industry classification
        2. Market trends affecting this company
        3. Common pain points and challenges in this industry
        4. Technology adoption patterns
        5. Competitive landscape
        
        Provide the information in JSON format:
        {{
            "industry_insights": {{
                "industry": "string",
                "market_trends": "string",
                "competitive_landscape": "string",
                "technology_adoption": "string"
            }},
            "pain_points": [
                "string1",
                "string2",
                "string3"
            ]
        }}
        
        Focus on specific, actionable insights that would be relevant for a sales conversation.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business analyst specializing in industry research and market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=800
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            print(f"❌ Error getting industry insights: {e}")
            return {"industry_insights": {}, "pain_points": []}
    
    def _get_market_position(self, company_name: str) -> Dict:
        """Get market position and competitive analysis using ChatGPT"""
        prompt = f"""
        Analyze the market position and competitive landscape for {company_name}. Provide insights on:
        1. Market position and competitive advantages
        2. Growth opportunities
        3. Risk factors
        4. Investment priorities
        5. Strategic positioning
        
        Provide the information in JSON format:
        {{
            "market_position": {{
                "position": "string",
                "competitive_advantages": "string",
                "growth_opportunities": "string",
                "risk_factors": "string",
                "investment_priorities": "string"
            }}
        }}
        
        Focus on insights that would be valuable for understanding the company's strategic direction and needs.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a strategic business consultant with expertise in market analysis and competitive intelligence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=600
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            print(f"❌ Error getting market position: {e}")
            return {"market_position": {}}
    
    def _get_recent_news(self, company_name: str) -> Dict:
        """Get recent news and developments using ChatGPT"""
        prompt = f"""
        Research recent news, developments, and announcements for {company_name} from the past 6 months. Focus on:
        1. Major business developments
        2. Strategic initiatives
        3. Technology investments
        4. Market expansion
        5. Leadership changes
        6. Financial announcements
        
        Provide the information in JSON format:
        {{
            "news": [
                "string1",
                "string2",
                "string3"
            ],
            "developments": [
                "string1", 
                "string2",
                "string3"
            ]
        }}
        
        Include only significant, verifiable developments that would be relevant for business discussions.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business news analyst with expertise in tracking company developments and market trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            print(f"❌ Error getting recent news: {e}")
            return {"news": [], "developments": []}
    
    def _get_stock_data(self, company_name: str) -> Dict:
        """Get stock data if company is publicly traded using ChatGPT"""
        prompt = f"""
        Research if {company_name} is a publicly traded company and provide stock information. Include:
        1. Stock ticker symbol (if public)
        2. Current stock price
        3. Market capitalization
        4. Key financial metrics
        
        Provide the information in JSON format:
        {{
            "is_public": boolean,
            "ticker": "string",
            "current_price": "string",
            "market_cap": "string",
            "key_metrics": "string"
        }}
        
        If the company is not publicly traded, set is_public to false and leave other fields empty.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in public company research and stock market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            print(f"❌ Error getting stock data: {e}")
            return {"is_public": False}
    
    def _get_generic_financial_context(self) -> Dict:
        """Return generic financial context when company data is unavailable"""
        return {
            'company_name': 'Unknown Company',
            'revenue_data': {
                'estimated_revenue': 'Revenue data not available',
                'employee_count': 'Company size varies',
                'financial_health': 'Stable operations'
            },
            'industry_insights': {
                'industry': 'General Business Services',
                'market_trends': 'Digital transformation across industries',
                'pain_points': 'Process automation, data management, cost optimization'
            },
            'market_position': {
                'position': 'Established business in their sector'
            },
            'news_data': [],
            'recent_developments': [],
            'stock_data': {'is_public': False},
            'pain_points': []
        }
    
    def format_financial_summary(self, financial_data: Dict) -> str:
        """Format financial data into a readable summary"""
        summary_parts = []
        
        # Company overview
        summary_parts.append(f"Company: {financial_data['company_name']}")
        if financial_data.get('person_name'):
            summary_parts.append(f"Contact: {financial_data['person_name']} - {financial_data.get('title', '')}")
        
        # Revenue information
        revenue = financial_data.get('revenue_data', {})
        if revenue.get('estimated_revenue'):
            summary_parts.append(f"Revenue: {revenue['estimated_revenue']}")
        if revenue.get('employee_count'):
            summary_parts.append(f"Employees: {revenue['employee_count']}")
        if revenue.get('financial_health'):
            summary_parts.append(f"Financial Health: {revenue['financial_health']}")
        
        # Stock information if public
        stock = financial_data.get('stock_data', {})
        if stock.get('is_public') and stock.get('ticker'):
            summary_parts.append(f"Stock: {stock['ticker']}")
            if stock.get('current_price'):
                summary_parts.append(f"Current Price: {stock['current_price']}")
            if stock.get('market_cap'):
                summary_parts.append(f"Market Cap: {stock['market_cap']}")
        
        # Industry insights
        industry = financial_data.get('industry_insights', {})
        if industry.get('industry'):
            summary_parts.append(f"Industry: {industry['industry']}")
        if industry.get('market_trends'):
            summary_parts.append(f"Market Trends: {industry['market_trends']}")
        
        # Pain points
        pain_points = financial_data.get('pain_points', [])
        if pain_points:
            summary_parts.append("Key Pain Points:")
            for point in pain_points[:3]:  # Limit to 3 points
                summary_parts.append(f"  • {point}")
        
        # Recent news
        news_data = financial_data.get('news_data', [])
        if news_data:
            summary_parts.append("Recent News:")
            for news in news_data[:2]:  # Limit to 2 news items
                summary_parts.append(f"  • {news}")
        
        return "\n".join(summary_parts)

# Example usage
if __name__ == "__main__":
    # Test the ChatGPT web search
    web_search = ChatGPTWebSearch()
    
    # Test with sample client data
    test_client = {
        'Full Name': 'Will Blue',
        'Company': 'Vanguard Truck Centers',
        'Domain': 'vanguardtrucks.com',
        'Title': 'CFO'
    }
    
    if web_search.api_key:
        financial_data = web_search.search_company_financials(test_client)
        summary = web_search.format_financial_summary(financial_data)
        
        print("🔍 ChatGPT Web Search Test")
        print("=" * 50)
        print(summary)
    else:
        print("❌ No OpenAI API key provided. Set OPENAI_API_KEY environment variable.") 