"""Agent responsible for crafting an extended persona life story."""
from __future__ import annotations

import json
import uuid
from typing import List, Tuple

from src.data.models import CanonicalFact, PersonaProfile
from src.utils.llm import get_llm_client


class LifeStoryAgent:
    """Generates rich persona history artifacts from transcript-derived facts."""

    def __init__(self):
        self.llm = get_llm_client()

    def generate_history(
        self,
        transcript: str,
        profile: PersonaProfile,
        existing_facts: List[CanonicalFact],
    ) -> Tuple[str, List[CanonicalFact]]:
        """Generate extended biography markdown and supplemental facts."""
        anchors = [
            f"- [{fact.id}] {fact.text}"
            for fact in existing_facts[:15]
        ]
        anchor_block = "\n".join(anchors) if anchors else "No structured facts extracted yet."
        transcript_excerpt = transcript[:2500]

        prompt = f"""You are building a richly detailed backstory for a persona-based assistant.

Anchor facts pulled from transcript:
{anchor_block}

Transcript excerpt (for tone and detail):
{transcript_excerpt}

Requirements:
1. Expand the persona's history with plausible, specific anecdotes that respect and never contradict the anchor facts.
2. You may invent additional details (events, relationships, routines) as long as they are consistent with the anchors.
3. Provide enough material that the persona can speak about their life, work, and quirks comfortably.

Return JSON only in the following format:
{{
  "biography_markdown": "Markdown with sections like ## Origins, ## Career, ## Relationships, ## Daily Rhythm, ## Personal Philosophy. 500-800 words.",
  "additional_facts": [
    {{
      "text": "Concise factual statement the assistant can quote about themselves.",
      "stance": "likes|dislikes|neutral|null",
      "date": "optional ISO date or null",
      "confidence": 0.0-1.0 (lower if speculative but still consistent),
      "entities": ["list", "of", "related", "entities"]
    }}
  ]
}}

Do not include markdown fences. Facts must stay aligned with the anchors and overall tone."""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.call(
                messages=messages,
                temperature=0.7,
                max_tokens=900,
            )

            payload = self._clean_json_block(response)
            data = json.loads(payload)

            biography_markdown = str(data.get("biography_markdown", "")).strip()
            additional_facts_data = data.get("additional_facts", [])

            history_facts: List[CanonicalFact] = []
            for fact_data in additional_facts_data[:20]:
                text_value = str(fact_data.get("text", "")).strip()
                if not text_value:
                    continue

                confidence_val = float(fact_data.get("confidence", 0.65))
                confidence_val = max(0.0, min(1.0, confidence_val))

                entities_val = fact_data.get("entities", [])
                if isinstance(entities_val, list):
                    entities = [str(entity) for entity in entities_val if isinstance(entity, str)]
                else:
                    entities = []

                history_facts.append(
                    CanonicalFact(
                        id=f"HX-{uuid.uuid4().hex[:8]}",
                        text=text_value,
                        source=f"{profile.name}_history",
                        date=fact_data.get("date"),
                        stance=fact_data.get("stance"),
                        confidence=confidence_val,
                        entities=entities,
                    )
                )

            if not biography_markdown:
                biography_markdown = self.fallback_biography(profile)

            return biography_markdown, history_facts

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return self.fallback_biography(profile), []

    @staticmethod
    def fallback_biography(profile: PersonaProfile) -> str:
        """Fallback biography when LLM parsing fails."""
        return f"""# {profile.name if profile.name else "Persona"}: Extended Snapshot

## Origins
{profile.backstory}

## Career & Impact
- Draws on expertise in {', '.join(profile.topics_of_expertise) or 'several domains'} to support others.

## Relationships & Values
- Guided by core values: {', '.join(profile.values) or 'pragmatism and empathy'}.

## Personal Philosophy
- Frequently reflects on experiences and shares lessons learned in a conversational, grounded tone.
"""

    @staticmethod
    def _clean_json_block(raw: str) -> str:
        """Strip markdown fences or whitespace around JSON payloads."""
        json_str = raw.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        return json_str.strip()
