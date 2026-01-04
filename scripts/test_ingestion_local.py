import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.web_scraper import HTMLCleaner
from src.scraper.pdf_generator import PDFGenerator, DocumentStorage
from datetime import datetime

def main():
    # Sample HTML content (simulating website)
    sample_html = """
    <html>
        <head>
            <title>Charity Water - Bringing Safe Water to Everyone</title>
            <meta name="description" content="charity: water is a non-profit organization bringing safe water and sanitation to people in Africa, South Asia and Latin America.">
        </head>
        <body>
            <nav>Navigation Menu - Ignored</nav>
            <main>
                <h1>About Charity Water</h1>
                <p>charity: water is a nonprofit that brings clean, safe drinking water and sanitation to people in Africa, South Asia and Latin America.</p>
                
                <h2>Our Mission</h2>
                <p>We're committed to bringing clean water and sanitation to developing countries. Since our founding in 2006, charity: water has completed 103,000+ water projects in 31 countries.</p>
                
                <h2>How to Donate</h2>
                <p>You can donate online at our website. 100% of public donations go directly to water projects. We cover all operating costs through private donors.</p>
                
                <h2>Our Impact</h2>
                <ul>
                    <li>1 billion+ people have benefited from our projects</li>
                    <li>31 countries across Africa, South Asia, and Latin America</li>
                    <li>103,000+ water projects completed</li>
                    <li>12 million+ people with access to clean water</li>
                </ul>
            </main>
            <footer>Footer - Ignored</footer>
        </body>
    </html>
    """
    
    print("="*80)
    print("LOCAL INGESTION PIPELINE TEST")
    print("="*80)
    
    try:
        # Step 1: Extract text and structure from HTML
        print("\n1. Extracting content from HTML...")
        text = HTMLCleaner.extract_text(sample_html)
        structure = HTMLCleaner.extract_structure(sample_html)
        
        print(f"   ✓ Title: {structure['title']}")
        print(f"   ✓ Extracted {len(text)} characters of text")
        
        # Step 2: Prepare metadata
        print("\n2. Preparing metadata...")
        metadata = {
            'title': structure['title'],
            'description': structure['meta_description'],
            'source_url': 'https://www.charitywater.org',
            'scraped_date': datetime.now().isoformat(),
            'charity_name': 'Charity Water',
            'pages_included': 1
        }
        print(f"   ✓ Metadata prepared")
        
        # Step 3: Generate PDF
        print("\n3. Generating PDF...")
        pdf_generator = PDFGenerator()
        pdf_path = pdf_generator.generate_pdf(
            content=text,
            metaData=metadata,
            charity_name='Charity Water'
        )
        print(f"   ✓ PDF generated: {pdf_path}")
        
        # Step 4: Store document metadata
        print("\n4. Storing document metadata...")
        storage = DocumentStorage()
        doc_record = storage.save_document(
            pdf_path=pdf_path,
            metaData=metadata,
            charity_name='Charity Water'
        )
        print(f"   ✓ Document stored")
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Charity Name: {doc_record['charity_name']}")
        print(f"Title: {doc_record['title']}")
        print(f"PDF Path: {doc_record['pdf_path']}")
        print(f"File Size: {doc_record['file_size']} bytes")
        print(f"Created: {doc_record['created_at']}")
        print(f"Status: {doc_record['status']}")
        print("\n✓ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
