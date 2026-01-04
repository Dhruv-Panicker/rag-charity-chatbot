import os
import json 
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from loguru import logger


#Generates PDF from extracted content 
class PDFGenerator:
    def __init__(self, output_dir: str = "data/generated_pdfs"): 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        #Defining styles for PDF
        self.styles = getSampleStyleSheet()
        self.__setup_custom_styles()

   
    #Setting up custom styles for PDF elements
    def __setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=4  # Justify
            )
        )

    #Generates PDF from content dictionary, returns path to generated PDF
    def generate_pdf(self, content: str, metaData: Dict, charity_name: str, filename: Optional[str] = None) -> str: 
        
        #create subdirectory 
        charity_dir = self.output_dir / charity_name.replace(" ", "_")
        charity_dir.mkdir(parents=True, exist_ok=True)

        #Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{charity_name}_{timestamp}.pdf"
        
        filepath = charity_dir / filename

        #Create PDF document 
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        #Build content for PDF
        story = [] 

        #Title
        story.append(Paragraph(metaData.get('title', 'Document'), self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))

        #Add metadata section 
        if metaData.get('description'): 
            story.append(Paragraph(metaData.get('description'), self.styles['CustomBody']))
            story.append(Spacer(1, 0.2*inch))
        
        # Add metadata table
        metadata_data = self._create_metadata_table(metaData)
        if metadata_data:
            story.append(self._create_metadata_table_element(metadata_data))
            story.append(Spacer(1, 0.3*inch))
        
        #add page before main content 
        story.append(PageBreak())

        #Add main content 
        paragraph = content.split('\n\n')
        for para in paragraph: 
            if para.strip(): 
                story.append(Paragraph(para.strip(), self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
        
        #Build PDF
        try: 
            doc.build(story)
            logger.info(f"PDF generated at {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise 
    
    #Create the metadata table
    def _create_metadata_table(self, metaData: Dict) -> List[List[str]]: 
        data = [['Field', 'Value']]

        if 'source_url' in metaData: 
            data.append(['Source URL', metaData['source_url']])
        if 'scraped_date' in metaData: 
            data.append(['Scraped Date', metaData['scraped_date']])
        if 'charity_name' in metaData: 
            data.append(['Organization', metaData['charity_name']])

        return data if len(data) > 1 else None 
    
    #Create Table element for metadata
    def _create_metadata_table_element(self, data: List[List[str]]):
        table = Table(data, colWidths=[1.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table


#Class for managing document storage and metadata 

class DocumentStorage: 
    def __init__(self, storage_dir: str = "data/document_storage"): 
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.jsonl"


    #Save Document 
    def save_document(self, pdf_path: str, metaData: Dict, charity_name: str) -> Dict: 
        doc_record = {
            'pdf_path': pdf_path,
            'charity_name': charity_name,
            'title': metaData.get('title'),
            'source_url': metaData.get('source_url'),
            'scraped_date': metaData.get('scraped_date'),
            'created_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(pdf_path),
            'status': 'active'
        }
        #Append metadata to file
        with open(self.metadata_file, 'a') as f: 
            f.write(json.dumps(doc_record) + "\n")
        logger.info(f"Document stored: {charity_name}")
        return doc_record
    
    #Retrieve documents by charity name
    def get_charity_documents(self, charity_name: str) -> List[Dict]: 
        documents = []

        if self.metadata_file.exists(): 
            with open(self.metadata_file, 'r') as f: 
                for line in f: 
                    if line.strip(): 
                        doc = json.loads(line)
                        if doc['charity_name'] == charity_name: 
                            documents.append(doc)
        
        return documents
    
    #Get most recently scraped document 
    def get_latest_document(self, charity_name: str) -> Optional[Dict]:
        documents = self.get_charity_documents(charity_name)
        if documents: 
            return max(documents, key=lambda x: x['created_at'])
        return None
     
