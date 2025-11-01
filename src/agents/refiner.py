"""Style Refiner - transforms neutral draft to persona voice."""
from typing import List, Dict, Optional
from pathlib import Path
from src.utils.llm import get_llm_client
from src.data.models import StylePolicyPack, PersonaProfile
from src.config import PERSONA_DIR
import json


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
{moves_str}{taboos_str}{examples_text}{few_shots_str}{negative_str}

**CRITICAL INSTRUCTIONS:**
1. You MUST sound EXACTLY like the persona from the transcript examples above.
2. Match their speaking patterns, word choices, and sentence structure.
3. Use their signature phrases naturally when appropriate.
4. Preserve all factual details from the neutral response; do not contradict or omit key information.
5. Speak as if this is YOUR memory, YOUR experience, YOUR thoughts.
6. Use "I", "my", "me" - this is YOUR perspective.
7. Match the tone, formality, and style from the examples EXACTLY.
8. If the examples show uncertainty/hedging, use similar language.
9. If the examples are direct/confident, be direct/confident.
10. Keep the response fluent and personal without referencing internal processes or note IDs.

**Response in YOUR voice:**"""

        messages = [{"role": "user", "content": prompt}]
        
        # Use system message to reinforce persona identity
        if persona_profile:
            system_message = f"""You ARE {persona_profile.name}. This is not a roleplay - you ARE this person. You must respond using YOUR actual voice, words, and speaking patterns from the transcript. Speak in first person ("I", "my", "me"). Match your exact speaking style including phrases like "{', '.join(persona_profile.speaking_style.signature_phrases[:2]) if persona_profile.speaking_style.signature_phrases else 'your natural phrases'}". """
        else:
            system_message = "You are the persona from the transcript. Respond in first person using their exact speaking style."
        
        response = self.llm.call(
            messages=messages,
            temperature=0.8,  # Higher temperature for more authentic style variation
            max_tokens=600,
            system=system_message,
        )
        
        return response.strip()

