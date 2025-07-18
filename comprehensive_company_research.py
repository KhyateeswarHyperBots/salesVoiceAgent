#!/usr/bin/env python3
"""
Comprehensive Company Research Module
Fetches detailed company information for sales intelligence
"""

import openai
import json
import time
from typing import Dict, List, Optional
import os
from datetime import datetime
from config import OPENAI_API_KEY

class ComprehensiveCompanyResearch:
    def __init__(self, api_key: str = None):
        """
        Initialize comprehensive company research with OpenAI API key
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            print("⚠️ Warning: No OpenAI API key provided. Set OPENAI_API_KEY in config.py.")
            return
        
        openai.api_key = self.api_key
        
    def research_company(self, client_data: Dict) -> Dict:
        """
        Conduct comprehensive company research with client-specific focus
        """
        company_name = client_data.get('Company', '')
        domain = client_data.get('Domain', '')
        person_name = client_data.get('Full Name', '')
        title = client_data.get('Title', '')
        email = client_data.get('Email', '')
        phone = client_data.get('Phone', '')
        
        if not company_name:
            return self._get_empty_research_template()
        
        print(f"🔍 Conducting comprehensive research for: {person_name} at {company_name}")
        
        research_data = {
            'cover_section': self._get_cover_section(company_name, person_name),
            'company_overview': self._get_company_overview_with_client(company_name, domain, person_name, title),
            'financial_ap_operations': self._get_financial_ap_operations_with_client(company_name, person_name, title),
            'key_teams_decision_makers': self._get_key_teams_decision_makers(company_name, person_name, title),
            'recent_news_trigger_events': self._get_recent_news_trigger_events_with_client(company_name, person_name),
            'intent_scorecard': self._calculate_intent_scorecard(company_name),
            'messaging_angle': self._get_messaging_angle(company_name, person_name, title)
        }
        
        return research_data
    
    def _get_cover_section(self, company_name: str, person_name: str) -> Dict:
        """Generate cover section"""
        return {
            'report_title': f"Comprehensive Research Report: {company_name}",
            'prepared_by': 'AI Sales Intelligence System',
            'date': datetime.now().strftime("%B %d, %Y"),
            'target_contact': person_name
        }
    
    def _get_company_overview_with_client(self, company_name: str, domain: str, person_name: str, title: str) -> Dict:
        """Get comprehensive company overview"""
        prompt = f"""
        Research and provide comprehensive company information for the EXACT company: "{company_name}" where {person_name} works as {title}.
        
        CLIENT CONTEXT: {person_name} is a {title} at {company_name}. This research should focus on the company where this specific person works.
        
        IMPORTANT: Research ONLY "{company_name}" - do not research similar companies or variations. 
        If you find multiple companies with similar names, focus on the one that matches "{company_name}" exactly.
        Verify this is the company where {person_name} ({title}) actually works.
        
        Provide the information in JSON format with these exact fields:
        {{
            "company_name": "{company_name}",
            "website": "string",
            "linkedin_page": "string", 
            "wikipedia_page": "string or null",
            "industry": "string",
            "sub_industry": "string",
            "company_description": "string",
            "year_founded": "string or null",
            "ownership_investor_info": "string",
            "company_type": "Public/Private",
            "headquarters_location": "string",
            "number_of_employees": "string",
            "annual_revenue": "string",
            "revenue_source": "string",
            "products_services": ["string1", "string2"],
            "business_model": "string",
            "revenue_model": "string"
        }}
        
        Only include information that can be reasonably verified from public sources for "{company_name}" specifically, where {person_name} works as {title}.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business intelligence analyst specializing in comprehensive company research. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to extract JSON if response contains extra text
            if result.startswith('```json'):
                result = result.split('```json')[1].split('```')[0].strip()
            elif result.startswith('```'):
                result = result.split('```')[1].split('```')[0].strip()
            
            return json.loads(result)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in company overview: {e}")
            print(f"Raw response: {result[:200]}...")
            return {}
        except Exception as e:
            print(f"❌ Error getting company overview: {e}")
            return {}
    
    def _get_financial_ap_operations_with_client(self, company_name: str, person_name: str, title: str) -> Dict:
        """Get financial and AP operations information"""
        prompt = f"""
        Research the financial and accounts payable operations for the EXACT company: "{company_name}" where {person_name} works as {title}.
        
        CLIENT CONTEXT: {person_name} is a {title} at {company_name}. Focus on the financial operations that this person would be involved with.
        
        IMPORTANT: Research ONLY "{company_name}" - do not research similar companies or variations.
        Verify this is the company where {person_name} ({title}) actually works.
        
        Provide the information in JSON format:
        {{
            "erp_systems_used": ["string1", "string2"],
            "ap_automation_tools": ["string1", "string2"],
            "ap_automation_maturity": "string (Low/Medium/High)",
            "shared_services_center": "Yes/No",
            "finance_ap_team_size": "string",
            "ap_volume_invoices_per_month": "string",
            "dpo": "string or null",
            "ap_analyst_job_postings": ["string1", "string2"],
            "ap_automation_mentioned_in_jd": "Yes/No",
            "known_pain_points": ["string1", "string2", "string3"],
            "exploring_automation_mentions": ["string1", "string2"],
            "rpa_ai_ml_mentions": ["string1", "string2"],
            "competitor_benchmarking": ["string1", "string2"]
        }}
        
        Focus on publicly available information about "{company_name}"'s technology stack and automation initiatives specifically, relevant to {person_name}'s role as {title}.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial technology analyst specializing in AP automation and ERP systems. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1200
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to extract JSON if response contains extra text
            if result.startswith('```json'):
                result = result.split('```json')[1].split('```')[0].strip()
            elif result.startswith('```'):
                result = result.split('```')[1].split('```')[0].strip()
            
            return json.loads(result)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in financial AP operations: {e}")
            print(f"Raw response: {result[:200]}...")
            return {}
        except Exception as e:
            print(f"❌ Error getting financial AP operations: {e}")
            return {}
    
    def _get_key_teams_decision_makers(self, company_name: str, person_name: str, title: str) -> Dict:
        """Get key teams and decision makers information"""
        prompt = f"""
        Research key decision makers and teams at the EXACT company: "{company_name}", with special focus on {person_name} ({title}).
        
        IMPORTANT: Research ONLY "{company_name}" - do not research similar companies or variations.
        Focus specifically on {person_name} who works at "{company_name}" as {title}.
        
        Provide the information in JSON format:
        {{
            "key_decision_makers": [
                {{
                    "name": "string",
                    "title": "string", 
                    "linkedin_url": "string or null",
                    "background_highlights": "string",
                    "ai_automation_activity": "string",
                    "public_quotes": ["string1", "string2"],
                    "connections": "string"
                }}
            ],
            "finance_team_structure": "string",
            "ap_team_structure": "string",
            "technology_team": "string"
        }}
        
        Include information about {person_name} at "{company_name}" and other key finance/technology decision makers at this specific company.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business intelligence analyst specializing in organizational research and decision maker identification. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to extract JSON if response contains extra text
            if result.startswith('```json'):
                result = result.split('```json')[1].split('```')[0].strip()
            elif result.startswith('```'):
                result = result.split('```')[1].split('```')[0].strip()
            
            return json.loads(result)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in key teams: {e}")
            print(f"Raw response: {result[:200]}...")
            return {}
        except Exception as e:
            print(f"❌ Error getting key teams: {e}")
            return {}
    
    def _get_recent_news_trigger_events_with_client(self, company_name: str, person_name: str) -> Dict:
        """Get recent news and trigger events"""
        prompt = f"""
        Research recent news, developments, and trigger events for the EXACT company: "{company_name}" where {person_name} works, from the past 12 months.
        
        CLIENT CONTEXT: {person_name} works at {company_name}. Focus on news that would be relevant to this person's role and the company's financial operations.
        
        IMPORTANT: Research ONLY "{company_name}" - do not research similar companies or variations.
        Verify this is the company where {person_name} actually works.
        
        Provide the information in JSON format:
        {{
            "funding_ma_activity": ["string1", "string2"],
            "leadership_changes": ["string1", "string2"],
            "tech_initiatives_ai_projects": ["string1", "string2"],
            "esg_ipo_transformation": ["string1", "string2"],
            "press_releases": [
                {{
                    "title": "string",
                    "link": "string or null",
                    "key_takeaways": "string"
                }}
            ],
            "trigger_events": ["string1", "string2", "string3"]
        }}
        
        Focus on events specific to "{company_name}" that would create urgency or interest in automation solutions, relevant to {person_name}'s role.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business news analyst specializing in identifying trigger events and business developments. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to extract JSON if response contains extra text
            if result.startswith('```json'):
                result = result.split('```json')[1].split('```')[0].strip()
            elif result.startswith('```'):
                result = result.split('```')[1].split('```')[0].strip()
            
            return json.loads(result)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in recent news: {e}")
            print(f"Raw response: {result[:200]}...")
            return {}
        except Exception as e:
            print(f"❌ Error getting recent news: {e}")
            return {}
    
    def _calculate_intent_scorecard(self, company_name: str) -> Dict:
        """Calculate intent scorecard based on research data"""
        # This would be calculated based on the research data
        # For now, return a template
        return {
            'ap_team_size_score': 0,
            'erp_used_score': 0,
            'ap_automation_mention_score': 0,
            'competitor_fomo_score': 0,
            'recent_executive_hires_score': 0,
            'exploring_automation_score': 0,
            'intent_total': 0,
            'trust_total': 0,
            'overall_score': 0
        }
    
    def _get_messaging_angle(self, company_name: str, person_name: str, title: str) -> Dict:
        """Generate messaging angle and personalization notes"""
        prompt = f"""
        Based on the research about the EXACT company "{company_name}" and {person_name} ({title}), create personalized messaging angles.
        
        IMPORTANT: Focus ONLY on "{company_name}" and {person_name} who works at this specific company.
        
        Provide the information in JSON format:
        {{
            "why_hyprbots_can_help": "string",
            "case_study_fomo_reference": "string",
            "hook_for_linkedin_email_call": "string",
            "suggested_subject_line": "string",
            "suggested_opener": "string",
            "personalization_notes": "string",
            "key_value_propositions": ["string1", "string2", "string3"],
            "objection_handling": ["string1", "string2"],
            "next_steps": "string"
        }}
        
        Focus on how Hyprbots' AI-powered finance automation can help {person_name} and "{company_name}" specifically.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a sales strategist specializing in personalized messaging and value proposition development. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to extract JSON if response contains extra text
            if result.startswith('```json'):
                result = result.split('```json')[1].split('```')[0].strip()
            elif result.startswith('```'):
                result = result.split('```')[1].split('```')[0].strip()
            
            return json.loads(result)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in messaging angle: {e}")
            print(f"Raw response: {result[:200]}...")
            return {}
        except Exception as e:
            print(f"❌ Error getting messaging angle: {e}")
            return {}
    
    def _get_empty_research_template(self) -> Dict:
        """Return empty research template"""
        return {
            'cover_section': {},
            'company_overview': {},
            'financial_ap_operations': {},
            'key_teams_decision_makers': {},
            'recent_news_trigger_events': {},
            'intent_scorecard': {},
            'messaging_angle': {}
        }
    
    def format_research_report(self, research_data: Dict) -> str:
        """Format research data into a comprehensive report"""
        report_parts = []
        
        # Cover Section
        cover = research_data.get('cover_section', {})
        report_parts.append("=" * 80)
        report_parts.append(f"REPORT: {cover.get('report_title', 'Company Research Report')}")
        report_parts.append(f"Prepared by: {cover.get('prepared_by', 'AI System')}")
        report_parts.append(f"Date: {cover.get('date', 'N/A')}")
        report_parts.append("=" * 80)
        
        # Section 01: Company Overview
        company = research_data.get('company_overview', {})
        report_parts.append("\nSECTION 01: COMPANY OVERVIEW")
        report_parts.append("-" * 40)
        if company.get('company_name'):
            report_parts.append(f"Company Name: {company['company_name']}")
        if company.get('website'):
            report_parts.append(f"Website: {company['website']}")
        if company.get('industry'):
            report_parts.append(f"Industry: {company['industry']}")
        if company.get('annual_revenue'):
            report_parts.append(f"Annual Revenue: {company['annual_revenue']}")
        if company.get('number_of_employees'):
            report_parts.append(f"Employees: {company['number_of_employees']}")
        
        # Section 02: Financial & AP Operations
        financial = research_data.get('financial_ap_operations', {})
        report_parts.append("\nSECTION 02: FINANCIAL & AP OPERATIONS")
        report_parts.append("-" * 40)
        if financial.get('erp_systems_used'):
            report_parts.append(f"ERP Systems: {', '.join(financial['erp_systems_used'])}")
        if financial.get('ap_automation_maturity'):
            report_parts.append(f"AP Automation Maturity: {financial['ap_automation_maturity']}")
        if financial.get('known_pain_points'):
            report_parts.append("Known Pain Points:")
            for point in financial['known_pain_points']:
                report_parts.append(f"  • {point}")
        
        # Section 03: Key Teams & Decision Makers
        teams = research_data.get('key_teams_decision_makers', {})
        report_parts.append("\nSECTION 03: KEY TEAMS & DECISION MAKERS")
        report_parts.append("-" * 40)
        if teams.get('key_decision_makers'):
            for person in teams['key_decision_makers']:
                report_parts.append(f"Name: {person.get('name', 'N/A')}")
                report_parts.append(f"Title: {person.get('title', 'N/A')}")
                if person.get('background_highlights'):
                    report_parts.append(f"Background: {person['background_highlights']}")
                report_parts.append("")
        
        # Section 04: Recent News & Trigger Events
        news = research_data.get('recent_news_trigger_events', {})
        report_parts.append("\nSECTION 04: RECENT NEWS & TRIGGER EVENTS")
        report_parts.append("-" * 40)
        if news.get('trigger_events'):
            report_parts.append("Trigger Events:")
            for event in news['trigger_events']:
                report_parts.append(f"  • {event}")
        
        # Section 05: Intent Scorecard
        intent = research_data.get('intent_scorecard', {})
        report_parts.append("\nSECTION 05: INTENT SCORECARD")
        report_parts.append("-" * 40)
        report_parts.append(f"Overall Score: {intent.get('overall_score', 'N/A')}")
        report_parts.append(f"Intent Total: {intent.get('intent_total', 'N/A')}")
        report_parts.append(f"Trust Total: {intent.get('trust_total', 'N/A')}")
        
        # Section 06: Messaging Angle
        messaging = research_data.get('messaging_angle', {})
        report_parts.append("\nSECTION 06: MESSAGING ANGLE & PERSONALIZATION")
        report_parts.append("-" * 40)
        if messaging.get('why_hyprbots_can_help'):
            report_parts.append(f"Why Hyprbots Can Help: {messaging['why_hyprbots_can_help']}")
        if messaging.get('suggested_subject_line'):
            report_parts.append(f"Suggested Subject Line: {messaging['suggested_subject_line']}")
        if messaging.get('key_value_propositions'):
            report_parts.append("Key Value Propositions:")
            for prop in messaging['key_value_propositions']:
                report_parts.append(f"  • {prop}")
        
        return "\n".join(report_parts)

# Example usage
if __name__ == "__main__":
    # Test the comprehensive research
    research = ComprehensiveCompanyResearch()
    
    # Test with sample client data
    test_client = {
        'Full Name': 'Will Blue',
        'Company': 'Vanguard Truck Centers',
        'Domain': 'vanguardtrucks.com',
        'Title': 'CFO'
    }
    
    if research.api_key:
        research_data = research.research_company(test_client)
        report = research.format_research_report(research_data)
        
        print("🔍 Comprehensive Company Research Test")
        print("=" * 80)
        print(report)
    else:
        print("❌ No OpenAI API key provided. Set OPENAI_API_KEY environment variable.") 