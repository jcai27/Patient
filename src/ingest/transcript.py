"""Transcript ingestion: chunking, indexing, artifact generation."""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import (
    PERSONA_DIR,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
)
from src.utils.llm import get_llm_client
from src.data.models import CanonicalFact, Example, PersonaProfile, SpeakingStyle
import uuid


class TranscriptIngester:
    """Ingests transcripts and generates persona artifacts."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def ingest(
        self,
        transcript_path: str,
        persona_name: str,
        transcript_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a transcript and generate all persona artifacts.
        
        Returns:
            Dict with counts: facts_count, examples_count, status
        """
        # Read transcript
        if transcript_text:
            transcript = transcript_text
        else:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read()
        
        # Create persona directory
        persona_dir = PERSONA_DIR / persona_name
        persona_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Chunk transcript
        chunks = self._chunk_transcript(transcript)
        
        # Step 2: Generate canonical facts
        facts = self._extract_facts(chunks, transcript_path)
        facts_count = len(facts)
        
        # Step 3: Generate persona profile
        profile = self._generate_profile(transcript, persona_name)
        
        # Step 4: Generate style rules
        style_rules = self._generate_style_rules(transcript, profile)
        
        # Step 5: Generate examples
        examples = self._generate_examples(chunks, profile)
        examples_count = len(examples)
        
        # Step 6: Generate taboo list (minimal, user can edit)
        taboos = self._generate_taboos(transcript)
        
        # Step 7: Save all artifacts
        self._save_artifacts(
            persona_dir,
            profile,
            style_rules,
            examples,
            facts,
            taboos,
        )
        
        # Step 8: Build vector index (triggered on first use of HybridRetriever)
        # This will happen automatically when the retriever is initialized
        
        return {
            "persona_name": persona_name,
            "facts_count": facts_count,
            "examples_count": examples_count,
            "chunks_count": len(chunks),
            "status": "success",
        }
    
    def _chunk_transcript(self, text: str) -> List[Dict[str, Any]]:
        """Chunk transcript into 120-180 word passages with overlap."""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + CHUNK_SIZE_WORDS]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "start_word": i,
                "end_word": min(i + CHUNK_SIZE_WORDS, len(words)),
                "word_count": len(chunk_words),
            })
            
            # Move forward with overlap
            i += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
            if i >= len(words):
                break
        
        return chunks
    
    def _extract_facts(self, chunks: List[Dict[str, Any]], source: str) -> List[CanonicalFact]:
        """Extract canonical facts from chunks using LLM."""
        facts = []
        
        # Process chunks in batches to avoid token limits
        for i, chunk in enumerate(chunks):
            prompt = f"""Extract factual claims from this transcript excerpt. Return as JSON array of facts.

Excerpt:
{chunk["text"]}

For each fact, provide:
- id: unique ID (e.g., "D{i}-1", "D{i}-2")
- text: the factual claim (concise, 20-50 words)
- source: "{source}"
- date: if mentioned, else null
- stance: if opinion/preference, else null
- confidence: 0.0-1.0 based on clarity in excerpt
- entities: array of named entities mentioned

Return JSON array only, no explanation."""

            messages = [{"role": "user", "content": prompt}]
            
            try:
                response = self.llm.call(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=500,
                )
                
                # Parse JSON
                json_str = response.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                
                chunk_facts = json.loads(json_str)
                if isinstance(chunk_facts, list):
                    for j, fact_data in enumerate(chunk_facts):
                        fact_data["source"] = f"{source}.chunk{i}"
                        if "id" not in fact_data:
                            fact_data["id"] = f"D{i}-{j+1}"
                        facts.append(CanonicalFact(**fact_data))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Skip this chunk if parsing fails
                continue
        
        return facts
    
    def _generate_profile(self, transcript: str, persona_name: str) -> PersonaProfile:
        """Generate persona profile from transcript."""
        prompt = f"""Analyze this transcript and extract a persona profile.

Transcript (excerpt):
{transcript[:2000]}

Generate a JSON persona profile:
{{
  "name": "{persona_name}",
  "backstory": "brief 2-3 sentence summary of who this person is",
  "values": ["value1", "value2"],
  "topics_of_expertise": ["topic1", "topic2"],
  "speaking_style": {{
    "avg_sentence_len": [min, max],
    "hedging_level": 0-5,
    "formality": 0-5,
    "emoji_policy": "none|light|rich",
    "signature_phrases": ["phrase1", "phrase2"]
  }},
  "taboos_refs": []
}}

Return JSON only."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm.call(
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        try:
            data = json.loads(json_str)
            return PersonaProfile(**data)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback profile
            return PersonaProfile(
                name=persona_name,
                backstory="Person based on transcript",
                values=[],
                topics_of_expertise=[],
                speaking_style=SpeakingStyle(),
            )
    
    def _generate_style_rules(self, transcript: str, profile: PersonaProfile) -> str:
        """Generate style rules markdown."""
        # Convert profile to dict manually
        profile_dict = {
            "name": profile.name,
            "backstory": profile.backstory,
            "values": profile.values,
            "topics_of_expertise": profile.topics_of_expertise,
            "speaking_style": {
                "avg_sentence_len": profile.speaking_style.avg_sentence_len,
                "hedging_level": profile.speaking_style.hedging_level,
                "formality": profile.speaking_style.formality,
                "emoji_policy": profile.speaking_style.emoji_policy,
                "signature_phrases": profile.speaking_style.signature_phrases,
            },
        }
        
        prompt = f"""Generate style rules in markdown format based on this persona profile.

Profile:
{json.dumps(profile_dict, indent=2)}

Create a markdown document with:
- Do's: what to do (sentence length targets, questions per 4-6 turns, etc.)
- Don'ts: what to avoid
- Specific examples from the transcript

Return markdown only."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm.call(
            messages=messages,
            temperature=0.4,
            max_tokens=500,
        )
        
        return response.strip()
    
    def _generate_examples(self, chunks: List[Dict[str, Any]], profile: PersonaProfile) -> List[Example]:
        """Generate few-shot examples from chunks."""
        examples = []
        
        # Find chunks with dialogue-like patterns
        for chunk in chunks[:10]:  # Sample first 10 chunks
            text = chunk["text"]
            
            # Look for question-answer patterns or statements
            if "?" in text or len(text.split()) > 30:
                # Convert speaking_style to dict manually
                style_dict = {
                    "avg_sentence_len": profile.speaking_style.avg_sentence_len,
                    "hedging_level": profile.speaking_style.hedging_level,
                    "formality": profile.speaking_style.formality,
                    "emoji_policy": profile.speaking_style.emoji_policy,
                    "signature_phrases": profile.speaking_style.signature_phrases,
                }
                
                prompt = f"""Extract or create a user-assistant example pair from this excerpt that demonstrates the persona style.

Excerpt:
{text}

Persona style:
{json.dumps(style_dict, indent=2)}

Return JSON:
{{
  "user": "user question/statement",
  "assistant": "persona response in their style",
  "intent": "advice|storytelling|opinion|chit-chat|default"
}}

Return JSON only."""

                messages = [{"role": "user", "content": prompt}]
                
                try:
                    response = self.llm.call(
                        messages=messages,
                        temperature=0.5,
                        max_tokens=300,
                    )
                    
                    json_str = response.strip()
                    if json_str.startswith("```json"):
                        json_str = json_str[7:]
                    if json_str.startswith("```"):
                        json_str = json_str[3:]
                    if json_str.endswith("```"):
                        json_str = json_str[:-3]
                    json_str = json_str.strip()
                    
                    data = json.loads(json_str)
                    examples.append(Example(**data))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        return examples[:5]  # Limit to 5 examples
    
    def _generate_taboos(self, transcript: str) -> str:
        """Generate basic taboo list (user should customize)."""
        return """# Taboo List

Topics and phrases to avoid or handle carefully.

## Topics
- [User should add topics to avoid]

## Phrases
- [User should add forbidden phrases]

## Refusal Language
- "I'd prefer not to discuss that."
- "Let's talk about something else instead."
"""
    
    def _save_artifacts(
        self,
        persona_dir: Path,
        profile: PersonaProfile,
        style_rules: str,
        examples: List[Example],
        facts: List[CanonicalFact],
        taboos: str,
    ):
        """Save all artifacts to files."""
        # Save profile (convert to dict manually)
        profile_dict = {
            "name": profile.name,
            "backstory": profile.backstory,
            "values": profile.values,
            "topics_of_expertise": profile.topics_of_expertise,
            "speaking_style": {
                "avg_sentence_len": profile.speaking_style.avg_sentence_len,
                "hedging_level": profile.speaking_style.hedging_level,
                "formality": profile.speaking_style.formality,
                "emoji_policy": profile.speaking_style.emoji_policy,
                "signature_phrases": profile.speaking_style.signature_phrases,
            },
            "taboos_refs": profile.taboos_refs,
        }
        with open(persona_dir / "persona_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_dict, f, indent=2, ensure_ascii=False)
        
        # Save style rules
        with open(persona_dir / "style_rules.md", "w", encoding="utf-8") as f:
            f.write(style_rules)
        
        # Save examples
        with open(persona_dir / "examples.jsonl", "w", encoding="utf-8") as f:
            for ex in examples:
                ex_dict = {
                    "user": ex.user,
                    "assistant": ex.assistant,
                    "intent": ex.intent,
                }
                f.write(json.dumps(ex_dict, ensure_ascii=False) + "\n")
        
        # Save facts
        with open(persona_dir / "canonical_facts.jsonl", "w", encoding="utf-8") as f:
            for fact in facts:
                fact_dict = {
                    "id": fact.id,
                    "text": fact.text,
                    "source": fact.source,
                    "date": fact.date,
                    "stance": fact.stance,
                    "confidence": fact.confidence,
                    "entities": fact.entities,
                }
                f.write(json.dumps(fact_dict, ensure_ascii=False) + "\n")
        
        # Save taboos
        with open(persona_dir / "taboo_list.md", "w", encoding="utf-8") as f:
            f.write(taboos)

