"""Conversation summarizer for rolling summaries."""
from typing import List, Dict, Optional
from src.utils.llm import get_llm_client


class ConversationSummarizer:
    """Creates and updates rolling conversation summaries."""
    
    def __init__(self):
        self.llm = get_llm_client()
        self.max_turns_before_summarize = 5  # Summarize every 5 turns
    
    def summarize(
        self,
        conversation_history: List[Dict[str, str]],
        previous_summary: Optional[str] = None,
    ) -> str:
        """
        Create or update rolling conversation summary.
        
        Args:
            conversation_history: Full conversation history
            previous_summary: Previous summary (if updating)
            
        Returns:
            Updated summary
        """
        if not conversation_history:
            return ""
        
        # Format recent turns
        recent_turns = conversation_history[-self.max_turns_before_summarize:]
        turns_str = "\n".join([
            f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
            for turn in recent_turns
        ])
        
        if previous_summary:
            # Update existing summary
            prompt = f"""Update the conversation summary with new information from recent turns.

Previous summary:
{previous_summary}

Recent conversation turns:
{turns_str}

Create an updated rolling summary that:
1. Preserves key information from the previous summary
2. Adds important new information from recent turns
3. Removes redundant or less important details to keep it concise (max 300 words)
4. Focuses on: user preferences, key topics discussed, decisions made, important facts mentioned

Updated summary:"""
        else:
            # Create initial summary
            prompt = f"""Create a rolling conversation summary from these turns.

Conversation:
{turns_str}

Create a concise summary (max 300 words) covering:
1. Main topics discussed
2. User preferences or concerns mentioned
3. Key facts or decisions
4. Important context for future responses

Summary:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        summary = self.llm.call(
            messages=messages,
            temperature=0.3,
            max_tokens=400,
        )
        
        return summary.strip()

