"""Agent 2: Orchestrator - main loop controller."""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from src.retriever.index import HybridRetriever
from src.retriever.rerank import Reranker
from src.agents.producer import Producer
from src.agents.contextor import Contextor
from src.agents.refiner import StyleRefiner
from src.agents.judge import Judge
from src.memory.episodic import EpisodicMemory
from src.memory.summarizer import ConversationSummarizer
from src.config import MAX_REVISE_LOOPS, K_RETRIEVE
import json


class Orchestrator:
    """Agent 2: Main orchestration loop controller."""
    
    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.retriever = HybridRetriever(persona_name)
        self.reranker = Reranker()
        self.producer = Producer()
        self.contextor = Contextor(persona_name)
        self.refiner = StyleRefiner()
        self.judge = Judge()
        self.memory = EpisodicMemory()
        self.summarizer = ConversationSummarizer()
        
        # Load persona profile for judge
        self._load_persona_profile()
    
    def _load_persona_profile(self):
        """Load persona profile for judge."""
        from pathlib import Path
        from src.config import PERSONA_DIR
        from src.data.models import PersonaProfile
        
        profile_file = Path(PERSONA_DIR) / self.persona_name / "persona_profile.json"
        if profile_file.exists():
            with open(profile_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.persona_profile_obj = PersonaProfile(**data)
                self.persona_profile_dict = data
        else:
            self.persona_profile_obj = None
            self.persona_profile_dict = {}
    
    def process_turn(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None,
        conversation_history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single conversation turn through the full pipeline.
        
        Returns:
            Dict with keys: response, session_id, citations, scores, revised, trace_id, trace
        """
        if conversation_history is None:
            conversation_history = []
        
        # Generate session_id if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        trace_id = str(uuid.uuid4())
        trace = {
            "trace_id": trace_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "iterations": 0,
        }
        
        # Step 1: Build conversation-aware query
        entity_mentions = self._extract_entities(user_message)
        query = self.retriever.build_conversation_query(
            user_message,
            conversation_history,
            entity_mentions,
        )
        
        # Step 2: Retrieve top-k notes
        initial_results = self.retriever.search(query, k=20)
        trace["initial_retrieval_count"] = len(initial_results)
        
        # Step 3: Rerank (if we have retrieval results)
        if initial_results:
            reranked_results = self.reranker.rerank(query, initial_results, top_k=K_RETRIEVE)
        else:
            reranked_results = []
        
        trace["retrieval_results"] = [
            {
                "fact_id": r["fact_id"],
                "text": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                "confidence": r.get("confidence", 0.8),
                "source": r.get("source", ""),
            }
            for r in reranked_results
        ]
        
        # Calculate average confidence (fallback to low confidence when nothing retrieved)
        if reranked_results:
            avg_confidence = sum(r.get("confidence", 0.8) for r in reranked_results) / len(reranked_results)
        else:
            avg_confidence = 0.35
        
        # Step 4: Producer → neutral content
        neutral_draft = self.producer.produce(
            query=query,
            retrieved_notes=reranked_results,
            user_message=user_message,
            conversation_history=conversation_history,
        )
        trace["producer_output"] = neutral_draft
        
        # Step 5: Contextor → Style+Policy Pack
        style_pack = self.contextor.build_pack(
            user_message,
            conversation_history,
            retrieved_confidence=avg_confidence,
        )
        trace["contextor_output"] = {
            "tone": style_pack.tone,
            "hedging_level": style_pack.hedging_level,
            "formality": style_pack.formality,
            "target_len_tokens": style_pack.target_len_tokens,
        }
        
        # Step 6: Style Refiner → styled message
        styled_response = self.refiner.refine(
            neutral_draft,
            style_pack,
            user_message,
            persona_name=self.persona_name,
            persona_profile=self.persona_profile_obj,
        )
        trace["refiner_output"] = styled_response
        
        # Step 7: Judge → accept or revise
        iterations = 0
        final_response = styled_response
        judge_scores = None
        judge_edits = []
        
        while iterations < MAX_REVISE_LOOPS:
            # Convert style_pack to dict (Pydantic model)
            style_pack_dict = {
                "tone": style_pack.tone,
                "hedging_level": style_pack.hedging_level,
                "formality": style_pack.formality,
                "emoji_policy": style_pack.emoji_policy,
                "target_len_tokens": style_pack.target_len_tokens,
                "signature_moves": style_pack.signature_moves,
                "taboos": style_pack.taboos,
            }
            
            judge_decision = self.judge.judge(
                final_response,
                user_message,
                reranked_results,
                self.persona_profile_dict,
                style_pack_dict,
            )
            
            judge_scores = judge_decision.scores.dict()
            trace[f"judge_iteration_{iterations + 1}"] = {
                "scores": judge_scores,
                "accept": judge_decision.accept,
                "edits": judge_decision.targeted_edits,
            }
            
            if judge_decision.accept:
                break
            
            # Apply edits
            if judge_decision.targeted_edits:
                judge_edits = judge_decision.targeted_edits
                final_response = self.judge.apply_edits(
                    final_response,
                    judge_decision.targeted_edits,
                    user_message,
                )
                iterations += 1
            else:
                break
        
        trace["iterations"] = iterations
        trace["final_response"] = final_response
        trace["judge_scores"] = judge_scores
        trace["judge_edits"] = judge_edits
        
        # Extract citations
        citations = self._extract_citations(final_response)
        trace["notes_used"] = citations if citations else [r["fact_id"] for r in reranked_results]
        
        # Update memory
        self._update_memory(
            user_id,
            session_id,
            user_message,
            final_response,
            conversation_history,
        )
        
        # Store trace (in production, this would go to a database or log)
        trace["stored"] = True
        
        return {
            "response": final_response,
            "session_id": session_id,
            "citations": citations,
            "scores": judge_scores,
            "revised": iterations > 0,
            "trace_id": trace_id,
            "trace": trace,
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity mentions (simple heuristic, can be enhanced with NER)."""
        # Simple: extract capitalized words/phrases
        words = text.split()
        entities = []
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                entities.append(word)
        return entities[:5]  # Limit to 5
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation IDs from response (e.g., [D3], [D7])."""
        import re
        pattern = r'\[([A-Z]+\d+)\]'
        citations = re.findall(pattern, text)
        return list(set(citations))  # Unique citations
    
    def _update_memory(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        conversation_history: List[Dict[str, str]],
    ):
        """Update episodic memory and conversation summary."""
        # Add current turn to history
        full_history = conversation_history + [
            {"user": user_message, "assistant": assistant_response}
        ]
        
        # Update summary every few turns
        if len(full_history) % 5 == 0:
            summary_record = self.memory.get_summary(session_id)
            previous_summary = summary_record["rolling_summary"] if summary_record else None
            
            new_summary = self.summarizer.summarize(full_history, previous_summary)
            self.memory.update_summary(session_id, user_id, new_summary, len(full_history))
        
        # Add episodic note for important information (heuristic)
        if any(word in user_message.lower() for word in ["prefer", "like", "dislike", "always", "never"]):
            self.memory.add_note(
                user_id,
                f"User mentioned: {user_message[:100]}",
                metadata={"response": assistant_response[:100]},
            )

