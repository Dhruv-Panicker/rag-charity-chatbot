from typing import List, Dict, Optional
from dataclasses import dataclass, field
from loguru import logger

#Manages a sliding context window for conversation 
@dataclass
class ConversationWindow: 
    max_messages: int = 10 
    max_tokens: int = 4000 

    messages: List[Dict] = field(default_factory=list)

    #Add message to the window 
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            'role': role,
            'content': content
        })
        #Trim window if exceeds limits
        self._trim_window()

    #Trim messages to fit within token and message limits
    def _trim_window(self) -> None:
        #Keep only recent messages within token limit 
        if len(self.messages) > self.max_messages: 
            #slice from max_messages to the end 
            self.messages = self.messages[-self.max_messages:]
            logger.info(f"Trimmed conversation window to last {self.max_messages} messages")
    
    #Get formatted conversation context 
    def get_context(self) -> str: 
        if not self.messages: 
            return ""
        
        context_lines = [] 
        for msg in self.messages[-4:]: #Get last 4 messages for context
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_lines)
    
    #Clear conversation history
    def clear(self) -> None: 
        self.messages = []
        logger.info("Cleared conversation window")
    
    #Get all messages in the window 
    def get_messages(self) -> List[Dict]: 
        return self.messages.copy()
    
#Manages multi-turn conversation with RAG system
class ConversationManager:
    def __init__(self, max_messages: int = 10, max_tokens: int = 4000): 
        self.window = ConversationWindow(
            max_messages=max_messages, 
            max_tokens=max_tokens
        )
        logger.info("Conversation Manager initialized")

    #Add user message
    def add_user_message(self, content: str) -> None: 
        self.window.add_message('user', content)
        logger.debug(f"Added user message: {content[:50]}...")
    
    #Add assistant message
    def add_assistant_message(self, content: str) -> None:
        self.window.add_message('assistant', content)
        logger.debug(f"Added assistant message: {content[:50]}...")

    #Check if conversation should be summarized
    def should_summarize(self) -> bool:
        return len(self.window.messages) > 20
    
    #Get context for continuing conversation 
    def get_conversation_context(self) -> str: 
        return self.window.get_context()
    
    #Get messages from client
    def get_messages_for_openai(self) -> List[Dict]:
        return [
            {
                'role': msg['role'],
                'content': msg['content']
            }
            for msg in self.window.get_messages()
        ]
    
    #Clear conversation 
    def clear(self) -> None: 
        self.window.clear()

    #Get conversation stats 
    def get_stats(self) -> Dict: 
        return {
            'total_messages': len(self.window.messages),
            'user_messages': sum(
                1 for m in self.window.messages if m['role'] == 'user'
            ),
            'assistant_messages': sum(
                1 for m in self.window.messages if m['role'] == 'assistant'
            )
        }
    


    

