from datetime import datetime
import traceback
from typing import Dict, List, Optional
from loguru import logger
from src.scraper.web_scraper import WebScraper, HTMLCleaner, SiteMapper
from src.scraper.pdf_generator import PDFGenerator, DocumentStorage


#Class for end-to-end ingestion pipeline from URL to PDF storage

class IngestionPipeline:
    def __init__(self, name: str, base_url: str): 
        self.name = name 
        self.base_url = base_url

        self.scrapper = WebScraper()
        self.pdf_generator = PDFGenerator()
        self.storage = DocumentStorage()
        self.site_mapper = SiteMapper(base_url)

    #Run the complee ingestion pipeline return metadata
    def run(self, scrape_all_page: bool = False) -> Dict: 
        logger.info(f"Starting ingestion pipeline for {self.name} at {self.base_url}")

        try: 
            #Discover pages
            if scrape_all_page: 
                logger.info(f"Discovering pages for {self.base_url}")
                urls = self.site_mapper.discover_pages()
            else: 
                urls = [self.base_url]
            
            logger.info(f"Found {len(urls)} pages to scrape")   

            #Scrape and clean content 
            all_content = []
            scraped_urls = [] 

            for url in urls: 
                logger.info(f"Processing: {url}")
                html = self.scrapper.get_html(url)
                if not html:
                    logger.warning(f"Failed to retrieve content from {url}")
                    continue 

                #Extract text 
                text = HTMLCleaner.extract_text(html)
                structure = HTMLCleaner.extract_structure(html)

                all_content.append({
                    'url': url,
                    'text': text,
                    'structure': structure
                })
                scraped_urls.append(url)

            if not all_content: 
                raise Exception("No content scraped from any pages.")
                

            #Combine content for PDF generation
            combined_text = self._combine_content(all_content)

            metadata = self._prepare_metadata(all_content)

            #Generate PDF
            pdf_path = self.pdf_generator.generate_pdf(
                content = combined_text, 
                metaData = metadata,
                charity_name = self.name
            )

            #Store the document 
            doc_record = self.storage.save_document(
                pdf_path = pdf_path,
                metaData = metadata,
                charity_name = self.name
            )

            result = {
                'status': 'success',
                'charity_name': self.name,
                'pdf_path': pdf_path,
                'urls_scraped': len(scraped_urls),
                'pages_scraped': scraped_urls,
                'document_record': doc_record
            }

            logger.info(f"pipeline completed successfully for {self.name}")
            return result
        except Exception as e:
            logger.error(f"Pipeline failed for {self.name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'charity_name': self.name,
                'error': str(e)
            }

    #Combine content from multiple pages    
    def _combine_content(self, contents: List[Dict]) -> str:
        combined = []

        for item in contents: 
            combined.append(f"# {item['structure'].get('title', 'Page')}\n")
            combined.append(f"URL: {item['url']}\n")
            combined.append(item['text'])
            combined.append("\n" + "="*80 + "\n")
        
        return "\n".join(combined)
    
    #Prepare metadata from scraped content 
    def _prepare_metadata(self, contents: List[Dict]) -> Dict:
        first_item = contents[0]
        structure = first_item['structure']

        return {
            'title': structure.get('title', self.name),
            'description': structure.get('meta_description', ''),
            'source_url': self.base_url,
            'scraped_date': datetime.now().isoformat(),
            'charity_name': self.name,
            'pages_included': len(contents)
        }

