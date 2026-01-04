import logging
import time
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse
from bs4 import NavigableString
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from loguru import logger


"""This class handles web scraping for given URLs using requests and selenium."""

class WebScraper: 
    def __init__(self, timeout: int = 10, max_retries: int = 3): 
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })

    #Scraping for static HTML content
    def scrape_with_requests(self, url: str) -> Optional[str]: 
        try: 
            logger.info(f"Scraping {url} with requests")
            # Better user agent list to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Referer': 'https://www.google.com/'
            }
            response = self.session.get(url, timeout=self.timeout, headers=headers, allow_redirects=True)
            logger.info(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            
            if len(response.text) < 100:
                logger.warning(f"Response too small ({len(response.text)} bytes), might be blocked or error page")
                return None
                
            return response.text
        except requests.exceptions.Timeout:
            logger.error(f"Timeout scraping {url}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"Requests scraping failed for {url}: {e}")
            return None
    
    #Scraping for dynamic content using seleniumm, Best for dynamic/JavaScript-heavy content
    def scrape_with_selenium(self, url: str) -> Optional[str]: 
        driver = None 
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')


            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            driver.get(url)

            #Wait for dynamic content to load 
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )

            html_content = driver.page_source
            logger.info(f"Successfully scraped {url} with selenium")
            return html_content
        except (TimeoutException, WebDriverException) as e:
            logger.error(f"Selenium scraping failed for {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()

    #Get HTML content with retry logic and Selenium fallback
    def get_html(self, url: str) -> Optional[str]: 
        # First try with requests
        for attempt in range(self.max_retries): 
            try: 
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: Trying requests for {url}")
                html = self.scrape_with_requests(url)
                
                if html: 
                    logger.info(f"✓ Successfully scraped {url} with requests")
                    return html
            
                time.sleep(2 ** attempt) 
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                time.sleep(2 ** attempt)
        
        # Fallback to Selenium
        try:
            html = self.scrape_with_selenium(url)
            if html:
                logger.info(f"✓ Successfully scraped {url} with Selenium")
                return html
        except Exception as e:
            logger.error(f"Selenium also failed for {url}: {e}")
        
        logger.error(f"✗ Failed to scrape {url} with both requests")
        return None
    
#This class cleans and extracts text from HTML content 

class HTMLCleaner: 
    #Tags to be removed (content + tag)
    TAGS_TO_REMOVE = [
        'script', 'style', 'meta', 'link', 'noscript',
        'nav', 'footer', '.navbar', '.footer', '.sidebar',
        'advertisement', 'ad', '.ad', '.advertisement'
    ]

    #CSS classes/IDs to be removed
    
    NOISE_CLASSES = [
        'navbar', 'sidebar', 'footer', 'cookie', 'banner',
        'advertisement', 'ad-', 'sponsor', 'share', 'social',
        'subscribe', 'newsletter', 'modal', 'popup'
    ]

    #Remove all common noise elements from the HTML soup
    @staticmethod
    def remove_noise(soup: BeautifulSoup) -> BeautifulSoup:  
        for tag in soup(['script', 'style', 'meta', 'link', 'noscript']): 
            tag.decompose()

        # Remove by class and id patterns 
        elements_to_remove = []
        for element in soup.find_all(True):
            # Skip NavigableString objects (text nodes) they don't have attrs
            if isinstance(element, NavigableString):
                continue
            
            try:
                classes = element.get('class', [])
                element_id = element.get('id', '')

                if any(noise in ' '.join(str(c) for c in classes).lower() for noise in HTMLCleaner.NOISE_CLASSES): 
                    elements_to_remove.append(element)
                elif any(noise in str(element_id).lower() for noise in HTMLCleaner.NOISE_CLASSES): 
                    elements_to_remove.append(element)
            except:
                pass
        
        # Now remove the collected elements
        for element in elements_to_remove:
            try:
                element.decompose()
            except:
                pass
                
        return soup
    
    #Extract clean text from HTML content
    @staticmethod
    def extract_text(html: str) -> str: 
        soup = BeautifulSoup(html, 'html.parser')

        #remove noise 
        soup = HTMLCleaner.remove_noise(soup)

        #Extract main content
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find(class_=lambda x: x and 'content' in x.lower()) or
            soup.find(class_=lambda x: x and 'container' in x.lower()) or
            soup.body or
            soup
        )

        #Get text with proper spacing
        text = main_content.get_text(separator='\n', strip=True)

        #clean up excess whitespace 
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)

        return text 
    
    #Extract structured information from HTML
    @staticmethod
    def extract_structure(html: str) -> Dict[str, any]: 
        soup = BeautifulSoup(html, 'html.parser')
        return {
            'title': soup.title.string if soup.title else '', 
            'meta_description': (
                soup.find('meta', attrs={'name': 'description'})
                .get('content', '') if soup.find('meta', attrs={'name': 'description'}) else ''
            ),
            'headings': {
                'h1': [h.get_text(strip=True) for h in soup.find_all('h1')], 
                'h2': [h.get_text(strip=True) for h in soup.find_all('h2')],
                'h3': [h.get_text(strip=True) for h in soup.find_all('h3')], 
            }, 
            'links': [
                {
                    'text': a.get_text(strip=True), 
                    'href': a.get('href', '')
                }
                for a in soup.find_all('a') if a.get('href')
            ][:20] #Limit to first 20 links 
        }
    
#Maps website structure and discovers all pages 
class SiteMapper: 

    def __init__(self, base_url: str, max_pages: int = 100): 
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visted_urls = set()
        self.scraper = WebScraper()
        
    def is_valid_url(self, url: str) -> bool:
        try: 
            parsed = urlparse(url)
            #Check the same domain 
            if parsed.netloc != self.domain: 
                return False 
            #Skip certain file types 
            if any(url.lower().endswith(ext) for ext in ['.pdf', '.zip', '.exe']):
                return False 
            return True
        except: 
            return False

    #Extract all valid URLs from a page  
    def get_all_urls(self, url: str) -> List[str]: 
        html = self.scraper.get_html(url)
        if not html: 
            return [] 
        
        soup = BeautifulSoup(html, 'html.parser')
        urls = []

        for link in soup.find_all('a', href=True): 
            absolute_url = urljoin(url, link['href'])
            if self.is_valid_url(absolute_url):
                urls.append(absolute_url)
        return urls
    
    #Discover all pages on website using BFS
    def discover_pages(self) -> List[str]: 
        to_visit = [self.base_url]
        discovered = []

        while to_visit and len(discovered) < self.max_pages:
            url = to_visit.pop(0)
            if url in self.visted_urls:
                continue  

            logger.info(f"Discovering page: {url}")
            self.visted_urls.add(url)
            discovered.append(url)

            #Get new urls within this page 
            new_urls = self.get_all_urls(url) 
            for new_url in new_urls: 
                if new_url not in self.visted_urls:
                    to_visit.append(new_url)

            time.sleep(1)  # Add delay 

        logger.info(f"Discovered {len(discovered)} pages")
        return discovered     



            

    





    






