"""
Index sample charity documents into the vector database
This creates the collections and embeddings needed for RAG tests
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.embeddings.chunking import ChunkConfig
from src.embeddings.embedding_generator import EmbeddingConfig
from loguru import logger


# Sample charity documents
SAMPLE_CHARITIES = {
    "Red Cross": """
Red Cross - Global Humanitarian Organization

MISSION:
The International Committee of the Red Cross (ICRC) is a humanitarian organization 
that protects human life and dignity during armed conflicts and natural disasters.

EDUCATIONAL PROGRAMS:
- Primary and secondary schools in 15 countries
- STEM education curriculum with focus on science and technology
- Teacher training programs in rural and underserved areas
- Adult literacy programs reaching 50,000+ adults annually
- Health education and first aid training
- Disaster preparedness workshops

HEALTHCARE SERVICES:
- Operates 50+ clinics across 20 countries
- Emergency medical response teams
- Disaster relief and trauma care
- Blood donation and transfusion services
- Mental health and psychological support
- COVID-19 response and vaccination programs

VOLUNTEER OPPORTUNITIES:
- Local Red Cross chapters need volunteers
- Training provided for all volunteers
- Flexible scheduling options
- Opportunities in: fundraising, disaster relief, health services
- Contact: volunteers@redcross.org
- Phone: 1-800-HELP-NOW

HOW TO DONATE:
- Online: www.redcross.org/donate
- Monthly giving programs available
- Special donation campaigns for emergency response
- 90% of donations go directly to services
- Tax-deductible donations

IMPACT STATISTICS:
- Serves 100+ million people annually
- Present in 80+ countries
- Trained 5 million first responders
- Provided $2 billion in disaster relief (2023)
""",

    "Doctors Without Borders": """
Doctors Without Borders (Médecins Sans Frontières)

MISSION:
Doctors Without Borders provides emergency medical assistance to people affected by 
armed conflicts, natural disasters, and epidemics, regardless of their race, religion, or politics.

MEDICAL PROGRAMS:
- Emergency medical care in conflict zones
- Surgical services and trauma care
- Disease outbreak response (COVID-19, Ebola, Cholera, Measles)
- Maternal and child health services
- Vaccination programs
- Mental health and psychosocial support
- Water and sanitation projects

GEOGRAPHIC PRESENCE:
- Works in 70+ countries worldwide
- Focus on areas with limited healthcare access
- Rapid deployment to humanitarian crises
- Average response time: 48 hours from crisis alert
- Maintains 1000+ expatriate staff members

MEDICAL TRAINING:
- Trains local healthcare workers
- Establishes medical facilities
- Provides ongoing supervision and support
- Capacity building in underserved regions

HOW TO SUPPORT:
- Donate online: www.doctorswithoutborders.org/donate
- Become a monthly donor
- Donate medical supplies
- Volunteer as medical professional
- Advocacy and awareness campaigns

CRISIS RESPONSE:
- 24/7 emergency response team
- Mobile medical units
- Field hospitals
- Telemedicine consultations
- Emergency supplies stockpiling

IMPACT (2023):
- Treated 10 million patients
- Performed 200,000+ surgeries
- Vaccinated 2 million people
- Present during 15 major crises
- Zero profit organization - all donations help patients
""",

    "UNICEF": """
UNICEF - United Nations Children's Fund

MISSION:
UNICEF provides humanitarian aid and developmental assistance to children and mothers 
in developing countries. We believe every child deserves a healthy start in life.

EDUCATIONAL PROGRAMS:
- Primary education initiatives in 150+ countries
- School feeding programs serving 9 million children
- Teacher training and education quality improvement
- Scholarship programs for disadvantaged children
- Education in emergency situations (conflicts, disasters)
- Digital literacy and technology access
- Pre-primary education expansion

HEALTHCARE FOR CHILDREN:
- Childhood vaccination programs (99% coverage in most countries)
- Maternal healthcare and safe delivery
- Treatment for malaria, diarrhea, pneumonia
- Nutrition programs addressing malnutrition
- HIV/AIDS treatment and prevention
- Mental health services for children
- Water, sanitation, and hygiene (WASH) programs

CHILD PROTECTION:
- Child labor prevention and rehabilitation
- Protection against violence and abuse
- Support for orphaned and vulnerable children
- Child marriage elimination programs
- Refugee and displaced child support

VOLUNTEER & PARTNERSHIP:
- Become a UNICEF volunteer
- Corporate partnerships
- Individual donations
- Fundraising opportunities
- Ambassador programs
- Contact: partnerships@unicef.org

GIVING OPTIONS:
- Monthly giving program
- Emergency relief donations
- Donate supplies (school kits, medical supplies)
- Legacy giving and planned giving
- Workplace giving programs
- All donations are tax-deductible

PROGRAMS BY REGION:
- Sub-Saharan Africa: Focus on child mortality reduction
- Asia-Pacific: Education and nutrition programs
- Latin America: Healthcare and protection services
- Middle East: Emergency response and education
- Europe: Refugee and displaced children support

IMPACT (2023):
- Reached 295 million children
- Vaccinated 95 million children
- Provided education to 61 million children
- Helped 2 million children in humanitarian crises
- Provided safe water to 140 million people
"""
}


def index_sample_documents():
    """Index sample charity documents into the vector database"""
    
    print("\n" + "="*80)
    print("INDEXING SAMPLE CHARITY DOCUMENTS")
    print("="*80)
    
    try:
        # Initialize embedding pipeline
        print("\n1. Initializing embedding pipeline...")
        pipeline = EmbeddingPipeline(
            chunk_config=ChunkConfig(
                strategy="fixed",
                chunk_size=256,
                overlap=50
            ),
            embedding_config=EmbeddingConfig()
        )
        print("   ✅ Pipeline initialized")
        
        # Index each charity's documents
        total_chunks = 0
        
        for charity_name, document_text in SAMPLE_CHARITIES.items():
            print(f"\n2. Indexing '{charity_name}'...")
            
            try:
                result = pipeline.process_charity(
                    charity_name=charity_name,
                    document_text=document_text
                )
                
                if result.get('status') == 'success':
                    num_chunks = result.get('chunking_stats', {}).get('total_chunks', 0)
                    total_chunks += num_chunks
                    
                    print(f"   ✅ Indexed successfully")
                    print(f"      - Created {num_chunks} chunks")
                    print(f"      - Avg tokens per chunk: {result.get('chunking_stats', {}).get('avg_tokens', 0):.0f}")
                    print(f"      - Collection: {result.get('collection_name', 'unknown')}")
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                print(f"   ❌ Error indexing {charity_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        print("INDEXING COMPLETE")
        print("="*80)
        print(f"\n✅ Successfully indexed {len(SAMPLE_CHARITIES)} charities")
        print(f"   Total chunks created: {total_chunks}")
        print(f"   Collections available for querying:")
        for charity_name in SAMPLE_CHARITIES.keys():
            collection_name = charity_name.lower().replace(' ', '_')
            print(f"      - {collection_name}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    index_sample_documents()
