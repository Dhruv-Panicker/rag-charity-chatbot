from dataclasses import dataclass
from typing import Optional, Dict

#Template configuration for LLM prompts
@dataclass
class PromptTemplateConfig: 
    name: str
    system_prompt: str
    user_template: str 

    #format user prompt with variables 
    def format_user_prompt(self, **kwargs) -> str: 
        return self.user_template.format(**kwargs)


#Collection of prompt templates for different scenarios 
class PromptLibrary: 

    #System prompt for RAG-based responses
    @staticmethod
    def get_rag_system_prompt() -> str: 
        return """You are a helpful assistant providing information about charitable organizations.
        Your responses should:
1. Be based ONLY on the provided context/documents
2. Be accurate and factual
3. Cite information sources when possible
4. Say "I don't have information about that" if the context doesn't cover the topic
5. Be concise and clear

You should NOT:
- Make up or invent information not in the context
- Provide information outside the context
- Make assumptions about details not mentioned

Always prioritize accuracy over a perfect answer."""

    #User prompt template for RAG-based Q&A
    @staticmethod
    def get_rag_user_template() -> str: 
        return """Based on the following context about {charity_name}, answer this question: {query}

CONTEXT:
{context}

ANSWER:"""

    #Template for charity-specific information retrieval
    @staticmethod
    def get_charity_search_template() -> str: 
        return """You are an assistant helping people find information about {charity_name}.

Context about this organization:
{context}

User question: {query}

Provide a helpful, accurate answer based only on the provided context."""

    #Template when no relevant context is found 
    @staticmethod
    def get_fallback_template() -> str: 
        return """I don't have enough information from {charity_name}'s documents to answer: {query}

You might want to:
1. Visit their website directly
2. Contact them for more specific information
3. Try a different question"""

#Class that formats prompts with context and variables
class PromptFormatter: 

    #Format a RAG prompt returns system and user keys
    @staticmethod
    def format_rag_prompt(query: str, context: str, charity_name: str = "this organization") -> Dict[str, str]:
        template = PromptTemplateConfig(
            name="rag", 
            system_prompt=PromptLibrary.get_rag_system_prompt(),
            user_template=PromptLibrary.get_rag_user_template()
        )

        user_prompt = template.format_user_prompt(
            query=query,
            context=context,
            charity_name=charity_name
        )

        return {
            'system': template.system_prompt,
            'user': user_prompt
        }
    
    #Format a fallback prompt when no context is found
    @staticmethod
    def format_fallback_prompt(query: str, charity_name: str = "this organization") -> str: 
        return PromptLibrary.get_fallback_template().format(
            query=query,    
            charity_name=charity_name
        )




