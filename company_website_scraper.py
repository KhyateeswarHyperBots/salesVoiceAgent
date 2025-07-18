import requests
from bs4 import BeautifulSoup, Tag
import re
import json
from urllib.parse import urljoin, urlparse
import time
from typing import Dict, List, Optional, Any, Union
import logging
import os
import hashlib
from datetime import datetime, timedelta
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompanyWebsiteScraper:
    def __init__(self, cache_dir="scraping_cache", cache_duration_hours=24, max_workers=3):
        # Configure session with connection pooling and retries
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Update headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.timeout = 15
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    async def scrape_multiple_websites(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """Scrape multiple websites concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._scrape_single_async(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                url: result if not isinstance(result, Exception) else {'error': str(result)}
                for url, result in zip(urls, results)
            }
    
    async def _scrape_single_async(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Scrape a single website asynchronously"""
        try:
            # Check cache first
            cached_data = self._load_from_cache(url)
            if cached_data:
                return cached_data
            
            # Normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Fetch with aiohttp
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract data
                company_data = self._extract_all_data(soup, url)
                
                # Save to cache
                self._save_to_cache(url, company_data)
                
                return company_data
                
        except Exception as e:
            logger.error(f"Async scraping error for {url}: {e}")
            return {'error': str(e), 'website_url': url}
    
    def _extract_all_data(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract all company data from soup"""
        return {
            'website_url': url,
            'scraping_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'company_overview': self._extract_company_overview(soup, url),
            'contact_information': self._extract_contact_information(soup, url),
            'products_services': self._extract_products_services_structured(soup),
            'team_leadership': self._extract_team_leadership(soup),
            'technology_stack': self._extract_technology_stack(soup),
            'social_media_presence': self._extract_social_media_presence(soup),
            'recent_news_articles': self._extract_recent_news_articles(soup),
            'company_metadata': self._extract_company_metadata(soup),
            'financial_indicators': self._extract_financial_indicators(soup),
            'business_operations': self._extract_business_operations(soup)
        }
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, url: str) -> str:
        """Get cache file path for URL"""
        cache_key = self._get_cache_key(url)
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, url: str) -> Optional[Dict[str, Any]]:
        """Load scraped data from cache if valid"""
        cache_path = self._get_cache_path(url)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached_data.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_time < self.cache_duration:
                logger.info(f"Using cached data for {url}")
                return cached_data
            else:
                logger.info(f"Cache expired for {url}")
                return None
                
        except Exception as e:
            logger.warning(f"Error loading cache for {url}: {e}")
            return None
    
    def _save_to_cache(self, url: str, data: Dict[str, Any]) -> None:
        """Save scraped data to cache"""
        try:
            cache_path = self._get_cache_path(url)
            data['cached_at'] = datetime.now().isoformat()
            data['cache_source'] = url
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Cached data for {url}")
        except Exception as e:
            logger.warning(f"Error saving cache for {url}: {e}")
    
    def _clean_old_cache(self) -> None:
        """Remove expired cache files"""
        try:
            current_time = datetime.now()
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if current_time - file_time > self.cache_duration:
                        os.remove(file_path)
                        logger.info(f"Removed expired cache: {filename}")
        except Exception as e:
            logger.warning(f"Error cleaning cache: {e}")
    
    def _extract_company_name(self, soup: BeautifulSoup, url: str) -> str:
        """Extract company name from various sources"""
        # Try title tag first
        title = soup.find('title')
        if title:
            title_text = title.get_text().strip()
            # Clean up common title patterns
            if '|' in title_text:
                title_text = title_text.split('|')[0].strip()
            if '-' in title_text:
                title_text = title_text.split('-')[0].strip()
            if title_text and len(title_text) < 100:
                return title_text
        
        # Try h1 tags
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags:
            text = h1.get_text().strip()
            if text and len(text) < 50:
                return text
        
        # Try logo alt text
        logos = soup.find_all('img', alt=True)
        for logo in logos:
            alt_text = logo.get('alt', '').strip()
            if alt_text and 'logo' in alt_text.lower():
                return alt_text.replace('logo', '').strip()
        
        # Extract from domain
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.split('.')[0].title()
    
    def _extract_tagline(self, soup: BeautifulSoup) -> str:
        """Extract company tagline or slogan"""
        # Look for common tagline patterns
        selectors = [
            '.tagline', '.slogan', '.hero-subtitle', '.subtitle',
            'h2', 'h3', '.lead', '.hero-text', '.banner-text'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) < 200 and len(text) > 10:
                    return text
        
        return ""
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract company description"""
        # Try meta description first
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and isinstance(meta_desc, Tag):
            content = meta_desc.get('content')
            if content:
                return content.strip()
        
        # Try Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and isinstance(og_desc, Tag):
            content = og_desc.get('content')
            if content:
                return content.strip()
        
        # Try first paragraph
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            if isinstance(p, Tag):
                text = p.get_text().strip()
                if text and len(text) > 50 and len(text) < 500:
                    return text
        
        return ""
    
    def _extract_contact_info(self, soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {
            'email': '',
            'phone': '',
            'address': '',
            'contact_page': ''
        }
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        page_text = soup.get_text()
        emails = re.findall(email_pattern, page_text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Extract phone numbers
        phone_patterns = [
            r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',
            r'\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',
            r'[0-9]{3}[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, page_text)
            if phones:
                contact_info['phone'] = phones[0]
                break
        
        # Look for contact page links
        contact_links = soup.find_all('a', href=True)
        for link in contact_links:
            href = link.get('href', '').lower()
            text = link.get_text().lower()
            if 'contact' in href or 'contact' in text:
                contact_url = urljoin(base_url, link.get('href'))
                contact_info['contact_page'] = contact_url
                break
        
        return contact_info
    
    def _extract_products_services(self, soup: BeautifulSoup) -> List[str]:
        """Extract products and services"""
        products = []
        
        # Look for common product/service sections
        selectors = [
            '.products', '.services', '.solutions', '.offerings',
            '.features', '.what-we-do', '.our-work'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                # Look for list items or headings
                items = element.find_all(['li', 'h3', 'h4'])
                for item in items:
                    text = item.get_text().strip()
                    if text and len(text) > 3 and len(text) < 100:
                        products.append(text)
        
        # Also look for navigation menu items
        nav_items = soup.find_all('nav')
        for nav in nav_items:
            links = nav.find_all('a')
            for link in links:
                text = link.get_text().strip()
                if text and len(text) > 3 and len(text) < 50:
                    products.append(text)
        
        return list(set(products))[:10]  # Remove duplicates and limit
    
    def _extract_team_members(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract team member information"""
        team = []
        
        # Look for team/about sections
        team_selectors = [
            '.team', '.about', '.leadership', '.staff',
            '.employees', '.people', '.crew'
        ]
        
        for selector in team_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Look for team member cards
                cards = element.find_all(['div', 'article'], class_=re.compile(r'team|member|person|employee'))
                for card in cards:
                    member = {}
                    
                    # Extract name
                    name_elem = card.find(['h3', 'h4', 'h5'])
                    if name_elem:
                        member['name'] = name_elem.get_text().strip()
                    
                    # Extract title/role
                    title_elem = card.find(['p', 'span', 'div'], class_=re.compile(r'title|role|position'))
                    if title_elem:
                        member['title'] = title_elem.get_text().strip()
                    
                    # Extract image
                    img = card.find('img')
                    if img and img.get('src'):
                        member['image'] = img.get('src')
                    
                    if member.get('name'):
                        team.append(member)
        
        return team[:10]  # Limit to 10 team members
    
    def _extract_social_media(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract social media links"""
        social_media = {}
        
        social_patterns = {
            'linkedin': r'linkedin\.com',
            'twitter': r'twitter\.com|x\.com',
            'facebook': r'facebook\.com',
            'instagram': r'instagram\.com',
            'youtube': r'youtube\.com',
            'github': r'github\.com'
        }
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href', '').lower()
            for platform, pattern in social_patterns.items():
                if re.search(pattern, href):
                    social_media[platform] = href
                    break
        
        return social_media
    
    def _extract_technologies(self, soup: BeautifulSoup) -> List[str]:
        """Extract technologies used (from meta tags, scripts, etc.)"""
        technologies = []
        
        # Check for common tech stack indicators
        tech_indicators = {
            'React': ['react', 'reactjs'],
            'Angular': ['angular'],
            'Vue': ['vue'],
            'WordPress': ['wordpress', 'wp-content'],
            'Shopify': ['shopify'],
            'Magento': ['magento'],
            'Drupal': ['drupal'],
            'Joomla': ['joomla'],
            'Bootstrap': ['bootstrap'],
            'jQuery': ['jquery'],
            'Node.js': ['node', 'nodejs'],
            'Python': ['python'],
            'PHP': ['php'],
            'Ruby': ['ruby'],
            'Java': ['java'],
            'ASP.NET': ['asp.net', 'aspnet'],
            'Google Analytics': ['google-analytics', 'gtag'],
            'Google Tag Manager': ['gtm'],
            'Facebook Pixel': ['facebook', 'fbq'],
            'Hotjar': ['hotjar'],
            'Intercom': ['intercom']
        }
        
        page_content = soup.get_text().lower()
        scripts = soup.find_all('script', src=True)
        
        for tech, indicators in tech_indicators.items():
            for indicator in indicators:
                if indicator in page_content:
                    technologies.append(tech)
                    break
                # Check script sources
                for script in scripts:
                    if indicator in script.get('src', '').lower():
                        technologies.append(tech)
                        break
        
        return list(set(technologies))
    
    def _extract_news_articles(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract recent news or blog articles"""
        articles = []
        
        # Look for blog/news sections
        article_selectors = [
            'article', '.post', '.blog-post', '.news-item',
            '.entry', '.story', '.content-item'
        ]
        
        for selector in article_selectors:
            elements = soup.select(selector)
            for element in elements:
                article = {}
                
                # Extract title
                title_elem = element.find(['h2', 'h3', 'h4'])
                if title_elem:
                    article['title'] = title_elem.get_text().strip()
                
                # Extract date
                date_elem = element.find(['time', 'span', 'div'], class_=re.compile(r'date|time|published'))
                if date_elem:
                    article['date'] = date_elem.get_text().strip()
                
                # Extract excerpt
                excerpt_elem = element.find(['p', 'div'], class_=re.compile(r'excerpt|summary|description'))
                if excerpt_elem:
                    article['excerpt'] = excerpt_elem.get_text().strip()
                
                if article.get('title'):
                    articles.append(article)
        
        return articles[:5]  # Limit to 5 articles
    
    def _extract_meta_data(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract meta data and structured data"""
        meta_data = {}
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                meta_data[name] = content
        
        # Extract JSON-LD structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    meta_data['json_ld'] = data
                elif isinstance(data, list):
                    meta_data['json_ld'] = data[0] if data else {}
            except:
                continue
        
        return meta_data

    def _extract_industry_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract industry information"""
        page_text = soup.get_text().lower()
        
        # Common industry keywords
        industries = {
            'technology': ['tech', 'software', 'saas', 'ai', 'machine learning', 'digital'],
            'finance': ['financial', 'banking', 'investment', 'fintech', 'payments'],
            'healthcare': ['health', 'medical', 'pharmaceutical', 'biotech'],
            'manufacturing': ['manufacturing', 'industrial', 'production', 'factory'],
            'retail': ['retail', 'ecommerce', 'shopping', 'consumer'],
            'consulting': ['consulting', 'advisory', 'professional services'],
            'education': ['education', 'learning', 'training', 'academic']
        }
        
        found_industry = ''
        found_sub_industry = ''
        
        for industry, keywords in industries.items():
            if any(keyword in page_text for keyword in keywords):
                found_industry = industry.title()
                break
        
        return {
            'industry': found_industry,
            'sub_industry': found_sub_industry
        }
    
    def _extract_location_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract location information"""
        page_text = soup.get_text()
        
        # Look for address patterns
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr)',
            r'(?:Headquarters|HQ|Office|Main Office)[:\s]+([A-Za-z\s,]+)',
            r'(?:Located in|Based in|Headquartered in)[:\s]+([A-Za-z\s,]+)'
        ]
        
        headquarters = ''
        for pattern in address_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                headquarters = match.group(1) if len(match.groups()) > 0 else match.group(0)
                break
        
        return {
            'headquarters': headquarters.strip() if headquarters else ''
        }
    
    def _extract_company_type(self, soup: BeautifulSoup) -> str:
        """Extract company type (Public/Private)"""
        page_text = soup.get_text().lower()
        
        if any(term in page_text for term in ['public company', 'traded on', 'nyse', 'nasdaq', 'stock exchange']):
            return 'Public'
        elif any(term in page_text for term in ['private company', 'privately held', 'family owned']):
            return 'Private'
        else:
            return 'Unknown'
    
    def _extract_founded_year(self, soup: BeautifulSoup) -> str:
        """Extract company founding year"""
        page_text = soup.get_text()
        
        # Look for founding year patterns
        year_patterns = [
            r'(?:Founded|Established|Started|Since)\s+(?:in\s+)?(\d{4})',
            r'(\d{4})\s+(?:Founded|Established|Started)',
            r'(?:Since|From)\s+(\d{4})'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                year = match.group(1)
                if 1800 <= int(year) <= 2024:
                    return year
        
        return ''
    
    def _extract_linkedin_page(self, soup: BeautifulSoup) -> str:
        """Extract LinkedIn page URL"""
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href', '').lower()
            if 'linkedin.com/company/' in href:
                return href
        return ''
    
    def _extract_ownership_info(self, soup: BeautifulSoup) -> str:
        """Extract ownership and investor information"""
        page_text = soup.get_text()
        
        ownership_patterns = [
            r'(?:Owned by|Acquired by|Merged with|Part of)\s+([A-Za-z\s&]+)',
            r'(?:Investor|Backed by|Funded by)\s+([A-Za-z\s&]+)',
            r'(?:Subsidiary of|Division of)\s+([A-Za-z\s&]+)'
        ]
        
        for pattern in ownership_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ''
    
    def _extract_employee_count(self, soup: BeautifulSoup) -> str:
        """Extract employee count"""
        page_text = soup.get_text()
        
        employee_patterns = [
            r'(\d+(?:,\d+)?)\s+(?:employees|staff|team members)',
            r'(?:Over|More than|Up to)\s+(\d+(?:,\d+)?)\s+(?:employees|staff)',
            r'(?:Team of|Staff of)\s+(\d+(?:,\d+)?)'
        ]
        
        for pattern in employee_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ''
    
    def _extract_revenue_info(self, soup: BeautifulSoup) -> str:
        """Extract revenue information"""
        page_text = soup.get_text()
        
        revenue_patterns = [
            r'(\$[\d,]+(?:\.\d+)?)\s+(?:million|billion|revenue|annual revenue)',
            r'(?:Revenue of|Annual revenue)\s+(\$[\d,]+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s+(?:million|billion)\s+(?:dollars|USD)'
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ''
    
    def _extract_business_model(self, soup: BeautifulSoup) -> str:
        """Extract business model information"""
        page_text = soup.get_text().lower()
        
        if 'saas' in page_text or 'software as a service' in page_text:
            return 'SaaS'
        elif 'consulting' in page_text or 'advisory' in page_text:
            return 'Consulting'
        elif 'ecommerce' in page_text or 'online retail' in page_text:
            return 'E-commerce'
        elif 'manufacturing' in page_text:
            return 'Manufacturing'
        else:
            return 'Unknown'
    
    def _extract_revenue_model(self, soup: BeautifulSoup) -> str:
        """Extract revenue model information"""
        page_text = soup.get_text().lower()
        
        if 'subscription' in page_text or 'monthly' in page_text or 'annual' in page_text:
            return 'Subscription'
        elif 'one-time' in page_text or 'license' in page_text:
            return 'License'
        elif 'commission' in page_text or 'percentage' in page_text:
            return 'Commission'
        else:
            return 'Unknown'
    
    def _extract_office_locations(self, soup: BeautifulSoup) -> List[str]:
        """Extract office locations"""
        page_text = soup.get_text()
        locations = []
        
        # Look for location patterns
        location_patterns = [
            r'(?:Office|Location|Branch)\s+(?:in|at)\s+([A-Za-z\s,]+)',
            r'(?:Located in|Based in)\s+([A-Za-z\s,]+)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            locations.extend(matches)
        
        return list(set(locations))
    
    def _extract_support_contact(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract support contact information"""
        page_text = soup.get_text()
        
        support_patterns = {
            'support_email': r'support@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            'support_phone': r'(?:Support|Help|Contact)\s+(?:Phone|Tel|Call)[:\s]+([\d\s\-\(\)]+)',
            'support_hours': r'(?:Support|Help)\s+(?:Hours|Available)[:\s]+([A-Za-z\s\d\-:]+)'
        }
        
        support_info = {}
        for key, pattern in support_patterns.items():
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                support_info[key] = match.group(1) if len(match.groups()) > 0 else match.group(0)
        
        return support_info
    
    def _extract_industry_solutions(self, soup: BeautifulSoup) -> List[str]:
        """Extract industry-specific solutions"""
        page_text = soup.get_text()
        solutions = []
        
        # Look for industry solution patterns
        solution_patterns = [
            r'(?:Solutions for|Services for|Specializing in)\s+([A-Za-z\s&]+)',
            r'([A-Za-z\s&]+)\s+(?:Industry|Sector|Market)'
        ]
        
        for pattern in solution_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            solutions.extend(matches)
        
        return list(set(solutions))
    
    def _extract_tech_platforms(self, soup: BeautifulSoup) -> List[str]:
        """Extract technology platforms"""
        page_text = soup.get_text()
        platforms = []
        
        # Common platform keywords
        platform_keywords = [
            'cloud', 'mobile', 'web', 'desktop', 'api', 'integration',
            'platform', 'software', 'application', 'system', 'solution'
        ]
        
        for keyword in platform_keywords:
            if keyword in page_text.lower():
                platforms.append(keyword.title())
        
        return platforms
    
    def _extract_pricing_info(self, soup: BeautifulSoup) -> List[str]:
        """Extract pricing information"""
        page_text = soup.get_text()
        pricing_models = []
        
        pricing_patterns = [
            r'(?:Free|No cost|Complimentary)',
            r'(?:Subscription|Monthly|Annual)',
            r'(?:One-time|Single|License)',
            r'(?:Pay-per-use|Usage-based)',
            r'(?:Enterprise|Custom)'
        ]
        
        for pattern in pricing_patterns:
            if re.search(pattern, page_text, re.IGNORECASE):
                pricing_models.append(re.search(pattern, page_text, re.IGNORECASE).group(0))
        
        return pricing_models
    
    def _extract_leadership_backgrounds(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract leadership background information"""
        backgrounds = []
        
        # Look for leadership sections
        leadership_selectors = [
            '.leadership', '.team', '.about', '.executives',
            'h2:contains("Leadership")', 'h3:contains("Team")'
        ]
        
        for selector in leadership_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Extract background information
                text = element.get_text()
                if any(keyword in text.lower() for keyword in ['experience', 'background', 'previously', 'former']):
                    backgrounds.append({
                        'section': selector,
                        'content': text[:200] + '...' if len(text) > 200 else text
                    })
        
        return backgrounds
    
    def _extract_automation_mentions(self, soup: BeautifulSoup) -> List[str]:
        """Extract automation mentions"""
        page_text = soup.get_text().lower()
        automation_terms = []
        
        automation_keywords = [
            'automation', 'rpa', 'robotic process automation', 'ai', 'artificial intelligence',
            'machine learning', 'ml', 'digital transformation', 'process improvement',
            'workflow automation', 'intelligent automation'
        ]
        
        for keyword in automation_keywords:
            if keyword in page_text:
                automation_terms.append(keyword.title())
        
        return automation_terms
    
    def _extract_social_activity(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract social media activity indicators"""
        page_text = soup.get_text().lower()
        
        return {
            'has_blog': 'blog' in page_text or 'articles' in page_text,
            'has_news': 'news' in page_text or 'press' in page_text,
            'has_events': 'events' in page_text or 'webinars' in page_text,
            'has_resources': 'resources' in page_text or 'downloads' in page_text
        }
    
    def _extract_press_releases(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract press releases"""
        press_releases = []
        
        # Look for press release links
        press_links = soup.find_all('a', href=True)
        for link in press_links:
            href = link.get('href', '').lower()
            text = link.get_text().lower()
            if 'press' in href or 'press' in text or 'release' in text:
                press_releases.append({
                    'title': link.get_text().strip(),
                    'url': link.get('href'),
                    'date': ''  # Would need to parse individual pages
                })
        
        return press_releases[:5]  # Limit to 5
    
    def _extract_blog_posts(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract blog posts"""
        blog_posts = []
        
        # Look for blog post links
        blog_links = soup.find_all('a', href=True)
        for link in blog_links:
            href = link.get('href', '').lower()
            text = link.get_text().lower()
            if 'blog' in href or 'article' in href or 'post' in href:
                blog_posts.append({
                    'title': link.get_text().strip(),
                    'url': link.get('href'),
                    'date': ''  # Would need to parse individual pages
                })
        
        return blog_posts[:5]  # Limit to 5
    
    def _extract_news_mentions(self, soup: BeautifulSoup) -> List[str]:
        """Extract news mentions"""
        page_text = soup.get_text()
        mentions = []
        
        # Look for news mention patterns
        news_patterns = [
            r'(?:Featured in|Mentioned in|Covered by)\s+([A-Za-z\s&]+)',
            r'(?:Press coverage|Media coverage|News coverage)',
            r'(?:Industry recognition|Awards|Accolades)'
        ]
        
        for pattern in news_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            mentions.extend(matches)
        
        return mentions
    
    def _extract_company_updates(self, soup: BeautifulSoup) -> List[str]:
        """Extract company updates"""
        page_text = soup.get_text()
        updates = []
        
        # Look for update patterns
        update_patterns = [
            r'(?:New|Recent|Latest|Announced)\s+([A-Za-z\s]+)',
            r'(?:Launched|Released|Introduced)\s+([A-Za-z\s]+)',
            r'(?:Expanded|Grew|Increased)\s+([A-Za-z\s]+)'
        ]
        
        for pattern in update_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            updates.extend(matches)
        
        return updates[:10]  # Limit to 10
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, microdata)"""
        structured_data = {}
        
        # Extract JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                structured_data['json_ld'] = data
            except:
                continue
        
        # Extract microdata
        microdata = soup.find_all(attrs={'itemtype': True})
        if microdata:
            structured_data['microdata'] = [item.get('itemtype') for item in microdata]
        
        return structured_data
    
    def _extract_seo_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract SEO keywords"""
        keywords = []
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            keywords.extend(meta_keywords.get('content').split(','))
        
        # Meta description keywords
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            # Extract key terms from description
            desc_text = meta_desc.get('content').lower()
            key_terms = ['solution', 'service', 'platform', 'technology', 'industry', 'business']
            for term in key_terms:
                if term in desc_text:
                    keywords.append(term)
        
        return list(set(keywords))
    
    def _extract_canonical_url(self, soup: BeautifulSoup) -> str:
        """Extract canonical URL"""
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        return canonical.get('href') if canonical else ''
    
    def _extract_revenue_mentions(self, soup: BeautifulSoup) -> List[str]:
        """Extract revenue mentions"""
        page_text = soup.get_text()
        mentions = []
        
        revenue_patterns = [
            r'(\$[\d,]+(?:\.\d+)?)\s+(?:million|billion|revenue)',
            r'(?:Revenue|Sales|Income)\s+(?:of|reached)\s+(\$[\d,]+(?:\.\d+)?)',
            r'(?:Annual|Yearly)\s+(?:revenue|sales)\s+(\$[\d,]+(?:\.\d+)?)'
        ]
        
        for pattern in revenue_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            mentions.extend(matches)
        
        return mentions
    
    def _extract_funding_mentions(self, soup: BeautifulSoup) -> List[str]:
        """Extract funding mentions"""
        page_text = soup.get_text()
        mentions = []
        
        funding_patterns = [
            r'(\$[\d,]+(?:\.\d+)?)\s+(?:funding|investment|raised)',
            r'(?:Raised|Secured|Received)\s+(\$[\d,]+(?:\.\d+)?)\s+(?:in funding|investment)',
            r'(?:Series [A-Z]|Seed|Angel|Venture)\s+(?:funding|investment)'
        ]
        
        for pattern in funding_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            mentions.extend(matches)
        
        return mentions
    
    def _extract_growth_indicators(self, soup: BeautifulSoup) -> List[str]:
        """Extract growth indicators"""
        page_text = soup.get_text()
        indicators = []
        
        growth_patterns = [
            r'(\d+(?:\.\d+)?%)\s+(?:growth|increase|expansion)',
            r'(?:Grew|Increased|Expanded)\s+(?:by\s+)?(\d+(?:\.\d+)?%)',
            r'(?:Growth|Increase|Expansion)\s+(?:of\s+)?(\d+(?:\.\d+)?%)'
        ]
        
        for pattern in growth_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            indicators.extend(matches)
        
        return indicators
    
    def _extract_financial_metrics(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract financial metrics"""
        page_text = soup.get_text()
        metrics = {}
        
        # Look for various financial metrics
        metric_patterns = {
            'profit_margin': r'(\d+(?:\.\d+)?%)\s+(?:profit margin|margin)',
            'roi': r'(\d+(?:\.\d+)?%)\s+(?:roi|return on investment)',
            'ebitda': r'(\$[\d,]+(?:\.\d+)?)\s+(?:ebitda|earnings)',
            'market_cap': r'(\$[\d,]+(?:\.\d+)?)\s+(?:market cap|market capitalization)'
        }
        
        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                metrics[metric] = match.group(1)
        
        return metrics
    
    def _extract_investor_relations(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract investor relations information"""
        page_text = soup.get_text()
        ir_info = {}
        
        # Look for investor relations patterns
        ir_patterns = {
            'investor_page': r'(?:Investor|IR|Shareholder)\s+(?:Relations|Page|Section)',
            'earnings_call': r'(?:Earnings|Quarterly)\s+(?:Call|Report|Release)',
            'sec_filings': r'(?:SEC|Securities)\s+(?:Filing|Report|Document)'
        }
        
        for key, pattern in ir_patterns.items():
            if re.search(pattern, page_text, re.IGNORECASE):
                ir_info[key] = 'Available'
        
        return ir_info
    
    def _extract_digital_transformation(self, soup: BeautifulSoup) -> List[str]:
        """Extract digital transformation mentions"""
        page_text = soup.get_text().lower()
        transformations = []
        
        transformation_keywords = [
            'digital transformation', 'digitalization', 'digitization',
            'modernization', 'technology upgrade', 'system upgrade',
            'cloud migration', 'digital strategy', 'innovation'
        ]
        
        for keyword in transformation_keywords:
            if keyword in page_text:
                transformations.append(keyword.title())
        
        return transformations
    
    def _extract_process_improvements(self, soup: BeautifulSoup) -> List[str]:
        """Extract process improvement mentions"""
        page_text = soup.get_text().lower()
        improvements = []
        
        improvement_keywords = [
            'process improvement', 'efficiency', 'optimization',
            'streamlining', 'workflow', 'productivity', 'performance',
            'best practices', 'standardization'
        ]
        
        for keyword in improvement_keywords:
            if keyword in page_text:
                improvements.append(keyword.title())
        
        return improvements
    
    def _extract_tech_initiatives(self, soup: BeautifulSoup) -> List[str]:
        """Extract technology initiatives"""
        page_text = soup.get_text()
        initiatives = []
        
        initiative_patterns = [
            r'(?:New|Recent|Latest)\s+(?:technology|tech|digital|software)\s+([A-Za-z\s]+)',
            r'(?:Launched|Introduced|Implemented)\s+(?:new|latest)\s+([A-Za-z\s]+)',
            r'(?:Technology|Digital|Software)\s+(?:initiative|project|program)'
        ]
        
        for pattern in initiative_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            initiatives.extend(matches)
        
        return initiatives
    
    def _extract_operational_metrics(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract operational metrics"""
        page_text = soup.get_text()
        metrics = {}
        
        # Look for operational metrics
        metric_patterns = {
            'customer_count': r'(\d+(?:,\d+)?)\s+(?:customers|clients|users)',
            'project_count': r'(\d+(?:,\d+)?)\s+(?:projects|implementations|deployments)',
            'uptime': r'(\d+(?:\.\d+)?%)\s+(?:uptime|availability)',
            'response_time': r'(\d+(?:\.\d+)?)\s+(?:seconds|minutes|hours)\s+(?:response|resolution)'
        }
        
        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                metrics[metric] = match.group(1)
        
        return metrics

    def _extract_company_overview(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract company overview in structured format"""
        company_name = self._extract_company_name(soup, url)
        description = self._extract_description(soup)
        tagline = self._extract_tagline(soup)
        
        # Extract additional company information
        industry_info = self._extract_industry_info(soup)
        location_info = self._extract_location_info(soup)
        company_type = self._extract_company_type(soup)
        founded_year = self._extract_founded_year(soup)
        
        return {
            'company_name': company_name,
            'website': url,
            'linkedin_page': self._extract_linkedin_page(soup),
            'wikipedia_page': None,  # Would need external API
            'industry': industry_info.get('industry', ''),
            'sub_industry': industry_info.get('sub_industry', ''),
            'company_description': description,
            'tagline': tagline,
            'year_founded': founded_year,
            'ownership_investor_info': self._extract_ownership_info(soup),
            'company_type': company_type,
            'headquarters_location': location_info.get('headquarters', ''),
            'number_of_employees': self._extract_employee_count(soup),
            'annual_revenue': self._extract_revenue_info(soup),
            'revenue_source': 'Website scraping',
            'products_services': self._extract_products_services(soup),
            'business_model': self._extract_business_model(soup),
            'revenue_model': self._extract_revenue_model(soup)
        }
    
    def _extract_contact_information(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        """Extract comprehensive contact information"""
        contact_info = self._extract_contact_info(soup, base_url)
        
        return {
            'email_addresses': [contact_info.get('email')] if contact_info.get('email') else [],
            'phone_numbers': [contact_info.get('phone')] if contact_info.get('phone') else [],
            'address': contact_info.get('address', ''),
            'contact_page': contact_info.get('contact_page', ''),
            'office_locations': self._extract_office_locations(soup),
            'support_contact': self._extract_support_contact(soup)
        }
    
    def _extract_products_services_structured(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract products and services in structured format"""
        products = self._extract_products_services(soup)
        
        return {
            'primary_products': products[:5] if products else [],
            'services_offered': products[5:10] if len(products) > 5 else [],
            'solutions_by_industry': self._extract_industry_solutions(soup),
            'technology_platforms': self._extract_tech_platforms(soup),
            'pricing_models': self._extract_pricing_info(soup)
        }
    
    def _extract_team_leadership(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract team and leadership information"""
        team_members = self._extract_team_members(soup)
        
        # Categorize team members by role
        leadership = []
        finance_team = []
        other_team = []
        
        for member in team_members:
            title = member.get('title', '').lower()
            if any(keyword in title for keyword in ['ceo', 'cfo', 'cto', 'president', 'director', 'vp', 'head']):
                leadership.append(member)
            elif any(keyword in title for keyword in ['finance', 'accounting', 'controller', 'ap', 'ar']):
                finance_team.append(member)
            else:
                other_team.append(member)
        
        return {
            'key_decision_makers': leadership,
            'finance_ap_team': finance_team,
            'other_team_members': other_team,
            'team_size_estimate': len(team_members),
            'leadership_backgrounds': self._extract_leadership_backgrounds(soup)
        }
    
    def _extract_technology_stack(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract technology stack information"""
        technologies = self._extract_technologies(soup)
        
        return {
            'frontend_technologies': [tech for tech in technologies if any(x in tech.lower() for x in ['react', 'angular', 'vue', 'bootstrap', 'jquery'])],
            'backend_technologies': [tech for tech in technologies if any(x in tech.lower() for x in ['node', 'python', 'php', 'java', 'ruby', 'asp'])],
            'cms_platforms': [tech for tech in technologies if any(x in tech.lower() for x in ['wordpress', 'drupal', 'joomla', 'shopify'])],
            'analytics_tools': [tech for tech in technologies if any(x in tech.lower() for x in ['google', 'analytics', 'gtag', 'gtm'])],
            'erp_systems': [tech for tech in technologies if any(x in tech.lower() for x in ['sap', 'oracle', 'netsuite', 'dynamics'])],
            'automation_tools': self._extract_automation_mentions(soup)
        }
    
    def _extract_social_media_presence(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract social media presence"""
        social_media = self._extract_social_media(soup)
        
        return {
            'linkedin': social_media.get('linkedin', ''),
            'twitter': social_media.get('twitter', ''),
            'facebook': social_media.get('facebook', ''),
            'instagram': social_media.get('instagram', ''),
            'youtube': social_media.get('youtube', ''),
            'github': social_media.get('github', ''),
            'social_media_activity': self._extract_social_activity(soup)
        }
    
    def _extract_recent_news_articles(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract recent news and articles"""
        articles = self._extract_news_articles(soup)
        
        return {
            'recent_articles': articles,
            'press_releases': self._extract_press_releases(soup),
            'blog_posts': self._extract_blog_posts(soup),
            'news_mentions': self._extract_news_mentions(soup),
            'company_updates': self._extract_company_updates(soup)
        }
    
    def _extract_company_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract company metadata"""
        meta_data = self._extract_meta_data(soup)
        
        return {
            'meta_tags': meta_data,
            'structured_data': self._extract_structured_data(soup),
            'seo_keywords': self._extract_seo_keywords(soup),
            'page_title': soup.find('title').get_text() if soup.find('title') else '',
            'canonical_url': self._extract_canonical_url(soup)
        }
    
    def _extract_financial_indicators(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract financial indicators from website"""
        return {
            'revenue_mentions': self._extract_revenue_mentions(soup),
            'funding_mentions': self._extract_funding_mentions(soup),
            'growth_indicators': self._extract_growth_indicators(soup),
            'financial_metrics': self._extract_financial_metrics(soup),
            'investor_relations': self._extract_investor_relations(soup)
        }
    
    def _extract_business_operations(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract business operations information"""
        return {
            'automation_mentions': self._extract_automation_mentions(soup),
            'digital_transformation': self._extract_digital_transformation(soup),
            'process_improvements': self._extract_process_improvements(soup),
            'technology_initiatives': self._extract_tech_initiatives(soup),
            'operational_metrics': self._extract_operational_metrics(soup)
        }

def main():
    """Test the scraper with example companies"""
    scraper = CompanyWebsiteScraper()
    
    # Example companies to test
    test_urls = [
        "https://www.hyprbots.com",
        "https://www.apple.com",
        "https://www.microsoft.com"
    ]
    
    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Scraping: {url}")
        print(f"{'='*60}")
        
        data = scraper.scrape_company_website(url)
        
        if 'error' in data:
            print(f"Error: {data['error']}")
        else:
            print(f"Company: {data['company_overview'].get('company_name', 'N/A')}")
            print(f"Tagline: {data['company_overview'].get('tagline', 'N/A')}")
            print(f"Description: {data['company_overview'].get('description', 'N/A')[:200]}...")
            print(f"Contact: {data['contact_information']}")
            print(f"Products/Services: {data['products_services'][:5]}")
            print(f"Technologies: {data['technology_stack']}")
            print(f"Social Media: {list(data['social_media_presence'].keys())}")
            
            # Save to file
            filename = f"scraped_data_{data['company_overview'].get('company_name', 'unknown').replace(' ', '_').lower()}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Data saved to: {filename}")

if __name__ == "__main__":
    main() 