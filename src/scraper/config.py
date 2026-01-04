from dataclasses import dataclass
from typing import Optional

@dataclass
class ScraperConfig:
    """Configuration for web scraping"""
    timeout: int = 10
    max_retries: int = 3
    max_pages: int = 100
    respect_robots_txt: bool = True
    delay_between_requests: float = 1.0

@dataclass
class PDFConfig:
    """Configuration for PDF generation"""
    page_size: str = "A4"
    font_size: int = 11
    margin_top: float = 0.75
    margin_bottom: float = 0.75
    margin_left: float = 1.0
    margin_right: float = 1.0
    include_toc: bool = True
    include_metadata: bool = True