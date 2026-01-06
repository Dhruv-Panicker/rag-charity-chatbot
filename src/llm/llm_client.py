from typing import Optional, Dict, List
from dataclasses import dataclass
from loguru import logger
import openai
from abc import ABC, abstractmethod

#Configuration for the LLM client
@dataclass
class LLMConfig: 
    provider: str = "openai"  
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500 
    timeout: int = 30  

#Abstract base class for LLM providers 
class LLMProvider(ABC): 

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: 
        pass



#OpenAI LLM client implementation
class OpenAIProvider(LLMProvider): 
    def __init__(self, config: LLMConfig): 
        self.config = config

        try: 
            openai.api_key = config.api_key
            self.client = openai.OpenAI(api_key=config.api_key)
            logger.info(f"Initialized OpenAI provider with {config.model_name}")
        except ImportError: 
            logger.error("OpenAI package not installed")
            raise

    #Genrate response using OpenAI API 
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str: 
        try: 
            response = self.client.chat.completions.create(
                model=self.config.model_name, 
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_provider_info(self) -> Dict: 
        return {
            'provider': 'openai',
            'model_name': self.config.model_name,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }
    








