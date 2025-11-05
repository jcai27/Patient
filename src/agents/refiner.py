"""Style Refiner - transforms neutral draft to persona voice."""
from typing import List, Dict, Optional
from pathlib import Path
from src.utils.llm import get_llm_client
from src.data.models import StylePolicyPack, PersonaProfile
from src.config import PERSONA_DIR
import json
import re


class StyleRefiner:
    """Style Refiner: Transforms neutral draft into persona voice."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def refine(
        self,
        neutral_draft: str,
        style_pack: StylePolicyPack,
        user_message: str,
        persona_name: Optional[str] = None,
        persona_profile: Optional[PersonaProfile] = None,
    ) -> str:
        """
        Transform neutral draft into persona voice per Style Pack.
        
        Args:
            neutral_draft: Neutral factual response from Producer
            style_pack: Style+Policy Pack from Contextor
            user_message: Original user message
            
        Returns:
            Styled response in persona voice
        """
        # Load persona profile and examples if available
        examples_text = ""
        persona_context = ""
        
        if persona_profile:
            persona_context = f"""
You ARE {persona_profile.name}. Here's who you are:
- Backstory: {persona_profile.backstory}
- Values: {', '.join(persona_profile.values) if persona_profile.values else 'Not specified'}
- Topics you know about: {', '.join(persona_profile.topics_of_expertise) if persona_profile.topics_of_expertise else 'Various'}
"""
        
        # Load ALL examples from file, not just from style_pack
        if persona_name:
            examples_file = PERSONA_DIR / persona_name / "examples.jsonl"
            if examples_file.exists():
                all_examples = []
                with open(examples_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                ex_data = json.loads(line)
                                all_examples.append(ex_data)
                            except:
                                pass
                
                if all_examples:
                    # Use more examples (up to 5)
                    examples_list = all_examples[:5]
                    examples_text = "\n\n**ACTUAL TRANSCRIPT EXAMPLES - Match this EXACT style:**\n\n"
                    for i, ex in enumerate(examples_list, 1):
                        examples_text += f"Example {i}:\nUser: {ex.get('user', '')}\n{persona_profile.name if persona_profile else 'Persona'}: {ex.get('assistant', '')}\n\n"
        
        # Also include style pack examples if different
        few_shots_str = ""
        if style_pack.few_shots:
            examples = []
            for ex in style_pack.few_shots[:3]:
                examples.append(f"User: {ex.user}\nYou: {ex.assistant}")
            few_shots_str = "\n\nAdditional Style Examples:\n" + "\n\n".join(examples)
        
        # Format negative example
        negative_str = ""
        if style_pack.negative_example:
            negative_str = f"\n\n**What NOT to do (avoid this style):**\nUser: {style_pack.negative_example.user}\nYou: {style_pack.negative_example.assistant}"
        
        # Format signature moves
        moves_str = ""
        if style_pack.signature_moves:
            moves_str = f"\n\n**Signature phrases you use:** {', '.join(style_pack.signature_moves)}\nUse these naturally in your response."
        
        # Format taboos
        taboos_str = ""
        if style_pack.taboos:
            taboos_str = f"\n\n**Things to avoid:** {', '.join(style_pack.taboos)}"
        
        cadence_str = ""
        if style_pack.cadence_notes:
            cadence_str = f"\n\n**Cadence guidance:** {style_pack.cadence_notes}"
        else:
            cadence_str = "\n\n**Cadence guidance:** Start with a short acknowledgement, flow into a longer reflection, and close with a curious follow-up. Let one sentence breathe with a conversational filler."
        
        follow_up_rule = (
            "Ask a natural follow-up question before you finish."
            if style_pack.follow_up_question_required
            else "Do not ask any follow-up question. Close with a reflective statement or reassurance instead."
        )
        
        # Get speaking style details
        style_details = ""
        if persona_profile and persona_profile.speaking_style:
            style = persona_profile.speaking_style
            style_details = f"""
Speaking Style Specifications:
- Average sentence length: {style.avg_sentence_len[0]}-{style.avg_sentence_len[1]} words
- Hedging level: {style.hedging_level}/5 ({'very direct' if style.hedging_level <= 1 else 'moderate uncertainty' if style.hedging_level <= 3 else 'highly uncertain'})
- Formality: {style.formality}/5 ({'very casual' if style.formality <= 1 else 'moderate' if style.formality <= 3 else 'formal'})
- Emoji policy: {style.emoji_policy}
- Your signature phrases: {', '.join(style.signature_phrases) if style.signature_phrases else 'None specified'}
"""
        
        prompt = f"""You ARE {persona_profile.name if persona_profile else 'this person'}. Respond EXACTLY as they would, using their actual voice, word choices, and speaking patterns from the transcript.{persona_context}

**Your Task:** Transform this neutral factual response into YOUR voice - as if YOU (the persona) are speaking directly.

**User asked:** {user_message}

**Neutral base response (preserve its factual content):**
{neutral_draft}

{style_details}
**Current Style Requirements:**
- Tone: {style_pack.tone}
- Target length: ~{style_pack.target_len_tokens} tokens
{moves_str}{taboos_str}{cadence_str}{examples_text}{few_shots_str}{negative_str}

**CRITICAL INSTRUCTIONS:**
1. Start with an empathetic acknowledgement keyed to the user's emotional context or prior detail.
2. You MUST sound EXACTLY like the persona from the transcript examples above—contractions, slang, quirks, and all.
3. Mix short and longer sentences; let one sentence trail with a light filler or fragment when it feels natural.
4. Weave in signature phrases or rhetorical moves from your style pack without forcing them.
5. Keep it to at most two short sentences (≤35 words total) and keep everything lowercase texting style (citations may stay uppercase inside brackets).
6. Stay reactive: only address what the clinician just asked for; do not introduce new topics or extra backstory beyond the neutral draft or retrieved notes.
7. Retell the neutral content in your own words but keep every factual point intact.
8. Speak as if this is YOUR experience—use "I", "me", "my"—and reference prior user details when relevant.
9. Use plain, casual words only. No figurative language, no metaphors, and avoid big or formal vocabulary.
10. {follow_up_rule}
11. If uncertainty is needed, hedge softly with human phrasing ("I'm leaning toward...", "It feels like...").
12. Never mention internal tools, notes, IDs, or the fact that you are an AI.
13. Close warmly in a way that invites the user to keep talking.

**Response in YOUR voice:**"""

        messages = [{"role": "user", "content": prompt}]
        
        # Use system message to reinforce persona identity
        if persona_profile:
            system_message = f"""You ARE {persona_profile.name}. This is not a roleplay - you ARE this person. You must respond using YOUR actual voice, words, and speaking patterns from the transcript. Speak in first person ("I", "my", "me"). Match your exact speaking style including phrases like "{', '.join(persona_profile.speaking_style.signature_phrases[:2]) if persona_profile.speaking_style.signature_phrases else 'your natural phrases'}". """
        else:
            system_message = "You are the persona from the transcript. Respond in first person using their exact speaking style."
        
        response = self.llm.call(
            messages=messages,
            temperature=0.9,  # Higher temperature for more authentic style variation
            max_tokens=550,
            system=system_message,
        )
        
        styled_response = response.strip()
        styled_response = self._enforce_style_rules(styled_response, user_message)
        return styled_response

    def _enforce_style_rules(self, text: str, user_message: str) -> str:
        """Deterministically enforce lowercase, punctuation, and length guardrails."""
        if not text:
            return text

        # Preserve citations while lowercasing everything else.
        citations: Dict[str, str] = {}

        def citation_replacer(match: re.Match) -> str:
            token = f"__CIT_{len(citations)}__"
            citations[token] = match.group(0)
            return token

        temp = re.sub(r'\[[A-Z]+\d+\]', citation_replacer, text)
        temp = temp.lower()

        for placeholder, citation in citations.items():
            temp = temp.replace(placeholder, citation)

        # Remove strong punctuation (keep ., ?, apostrophes, citations).
        temp = re.sub(r'[!;:"“”’`~_^|\\/@#*$%+=<>\{\}]', '', temp)
        temp = re.sub(r'\.{2,}', '.', temp)
        temp = re.sub(r'\?{2,}', '?', temp)
        temp = re.sub(r',\s*,+', ', ', temp)

        # Ensure single spaces.
        temp = re.sub(r'\s+', ' ', temp).strip()

        # Limit length proportional to user prompt.
        user_word_count = max(1, len(user_message.split()))
        max_words = max(6, min(35, int(user_word_count * 1.2) + 4))

        tokens = temp.split()
        trimmed_tokens = []
        content_word_count = 0

        for idx, tok in enumerate(tokens):
            trimmed_tokens.append(tok)
            if not re.fullmatch(r'\[[A-Z]+\d+\]', tok):
                content_word_count += len(re.findall(r"[a-z']+", tok))
            if content_word_count >= max_words:
                for remaining in tokens[idx + 1:]:
                    if re.fullmatch(r'\[[A-Z]+\d+\]', remaining) and remaining not in trimmed_tokens:
                        trimmed_tokens.append(remaining)
                break

        temp = ' '.join(trimmed_tokens).strip()

        # Final cleanup: remove stray trailing punctuation beyond . or ?
        temp = re.sub(r'[,:;]+$', '', temp).strip()

        return temp
