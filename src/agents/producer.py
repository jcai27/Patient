"""Agent 1: Producer - generates neutral factual drafts."""
from typing import List, Dict, Any, Optional
from src.utils.llm import get_llm_client
from src.data.models import CanonicalFact


class Producer:
    """Agent 1: Produces neutral, factual drafts from retrieved notes."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def produce(
        self,
        query: str,
        retrieved_notes: List[Dict[str, Any]],
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a neutral, factual answer from retrieved notes.
        
        Args:
            query: User's question
            retrieved_notes: List of retrieved facts with keys: fact, fact_id, text, etc.
            
        Returns:
            Neutral factual response ready for stylistic refinement
        """
        if not retrieved_notes:
            history_snippets: List[str] = []
            if conversation_history:
                for turn in conversation_history[-3:]:
                    user_turn = turn.get("user", "").strip()
                    assistant_turn = turn.get("assistant", "").strip()
                    if user_turn or assistant_turn:
                        history_snippets.append(
                            f"User: {user_turn}\nAssistant: {assistant_turn}"
                        )
            history_block = "\n\n".join(history_snippets) if history_snippets else "No prior conversation available."
            
            prompt = f"""You are drafting a neutral base reply for a persona-driven assistant.

Conversation history:
{history_block}

Latest user message: {user_message}

Guidelines:
1. Acknowledge the user's situation using only details provided.
2. Offer a supportive, informative, or curiosity-driven follow-up that keeps the dialogue going.
3. Avoid first-person language or persona-specific style; stay neutral so another component can adapt the voice.
4. Do not mention missing data, citations, or internal processes.
5. Keep the response to 2-3 sentences.

Neutral response:"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.call(
                messages=messages,
                temperature=0.4,
                max_tokens=200,
            )
            return response.strip()
        
        # Format notes for prompt
        notes_text = []
        for note in retrieved_notes:
            fact = note["fact"]
            note_id = note["fact_id"]
            text = note["text"]
            confidence = note.get("confidence", 0.8)
            
            note_str = f"[{note_id}] {text}"
            if confidence < 0.5:
                note_str += " (lower confidence)"
            notes_text.append(note_str)
        
        notes_block = "\n".join(notes_text)
        
        prompt = f"""You are extracting factual information from notes. Using ONLY the following notes, write a concise, factual answer.

Notes:
{notes_block}

User question: {query}

Instructions:
1. Use ONLY information from the notes above.
2. Write in 2-4 sentences with a neutral, third-person tone.
3. If information is missing or uncertain, communicate that plainly.
4. Do NOT include citation brackets, note IDs, or metadata in the response.
5. Keep stylistic choices minimal so another component can adapt the voice later.

Neutral factual answer:"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.call(
            messages=messages,
            temperature=0.3,  # Low temperature for factual content
            max_tokens=500,
        )
        
        return response.strip()

