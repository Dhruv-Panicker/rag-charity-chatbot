import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.ingestion_pipeline import IngestionPipeline

def main():
    # Example: Test with a non-profit website
    # Using Wikipedia article as test (less likely to block)
    charity_name = "Sewa Canada"
    charity_url = "https://sewacanada.com/"
    
    print("="*80)
    print("INGESTION PIPELINE TEST - LIVE WEBSITE")
    print("="*80)
    print(f"Target URL: {charity_url}")
    print("="*80 + "\n")
    
    pipeline = IngestionPipeline(
        name=charity_name,
        base_url=charity_url
    )
    
    # Run pipeline (scrape only homepage for testing)
    result = pipeline.run(scrape_all_page=True)
    
    print("\n" + "="*80)
    print("INGESTION PIPELINE RESULT")
    print("="*80)
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Charity: {result['charity_name']}")
        print(f"PDF Path: {result['pdf_path']}")
        print(f"URLs Scraped: {result['urls_scraped']}")
        print(f"Pages: {result['pages_scraped']}")
        print("\n✓ SUCCESS: Pipeline completed successfully!")
    else:
        print(f"\n✗ ERROR: {result.get('error')}")

if __name__ == "__main__":
    main()
