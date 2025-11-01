"""Agent 3: Contextor - builds Style+Policy Pack."""
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from src.utils.llm import get_llm_client
from src.config import PERSONA_DIR, STYLE_LENGTH_TARGETS
from src.data.models import StylePolicyPack, Example, PersonaProfile


class Contextor:
    """Agent 3: Builds Style+Policy Pack tailored to current user intent."""
    
    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.llm = get_llm_client()
        self.persona_dir = PERSONA_DIR / persona_name
        self.profile: Optional[PersonaProfile] = None
        self.style_rules: str = ""
        self.examples: List[Example] = []
        self.taboos: List[str] = []
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load persona artifacts."""
        # Load profile
        profile_file = self.persona_dir / "persona_profile.json"
        if profile_file.exists():
            with open(profile_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.profile = PersonaProfile(**data)
        
        # Load style rules
        rules_file = self.persona_dir / "style_rules.md"
        if rules_file.exists():
            with open(rules_file, "r", encoding="utf-8") as f:
                self.style_rules = f.read()
        
        # Load examples
        examples_file = self.persona_dir / "examples.jsonl"
        if examples_file.exists():
            with open(examples_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.examples.append(Example(**data))
        
        # Load taboos
        taboos_file = self.persona_dir / "taboo_list.md"
        if taboos_file.exists():
            with open(taboos_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Parse taboos (lines starting with - or *)
                self.taboos = [
                    line.strip().lstrip("-* ").strip()
                    for line in content.split("\n")
                    if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))
                ]
    
    def build_pack(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        retrieved_confidence: float = 0.8,
    ) -> StylePolicyPack:
        """
        Build Style+Policy Pack tailored to current user intent.
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation turns
            retrieved_confidence: Average confidence of retrieved facts
            
        Returns:
            StylePolicyPack
        """
        # Determine intent
        intent = self._classify_intent(user_message, conversation_history)
        
        # Get length targets
        length_target = STYLE_LENGTH_TARGETS.get(intent, STYLE_LENGTH_TARGETS["default"])
        
        # Select few-shots (2-3 examples matching intent)
        few_shots = self._select_few_shots(intent, 2)
        
        # Build prompt for LLM to generate pack
        profile_str = ""
        if self.profile:
            profile_str = f"""
Persona Profile:
- Name: {self.profile.name}
- Backstory: {self.profile.backstory}
- Values: {', '.join(self.profile.values)}
- Expertise: {', '.join(self.profile.topics_of_expertise)}
- Style: avg_sentence_len={self.profile.speaking_style.avg_sentence_len}, 
         hedging={self.profile.speaking_style.hedging_level},
         formality={self.profile.speaking_style.formality},
         emoji={self.profile.speaking_style.emoji_policy},
         phrases={', '.join(self.profile.speaking_style.signature_phrases)}
"""
        
        style_rules_str = self.style_rules[:500]  # Limit length
        taboos_str = "\n".join(self.taboos[:10])  # Limit to first 10
        
        prompt = f"""You are a style coordinator. Based on the persona profile and user's current message, generate a Style+Policy Pack.

{profile_str}

Style Rules (excerpt):
{style_rules_str}

Taboos:
{taboos_str}

User Message: {user_message}
Detected Intent: {intent}
Retrieved Confidence: {retrieved_confidence}

Generate a JSON Style+Policy Pack with:
- tone: string (e.g., "warm", "professional", "casual", "enthusiastic")
- hedging_level: integer 0-5 (adjust based on retrieved_confidence: if <0.5, increase hedging)
- formality: integer 0-5
- emoji_policy: "none", "light", or "rich"
- target_len_tokens: integer (based on intent: {length_target})
- signature_moves: array of strings (2-3 phrases/patterns from persona)
- taboos: array of strings (relevant taboos for this query)
- few_shots: array of 2-3 examples (user/assistant pairs from examples)
- negative_example: optional object with user/assistant showing what NOT to do

Return ONLY valid JSON, no markdown or explanation."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm.call(
            messages=messages,
            temperature=0.5,
            max_tokens=800,
        )
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            data = json.loads(json_str)
            
            # Convert few_shots to Example objects
            few_shots_objs = []
            for ex in data.get("few_shots", [])[:3]:
                few_shots_objs.append(Example(**ex))
            
            negative_ex = None
            if "negative_example" in data and data["negative_example"]:
                negative_ex = Example(**data["negative_example"])
            
            pack = StylePolicyPack(
                tone=data.get("tone", "neutral"),
                hedging_level=data.get("hedging_level", 2),
                formality=data.get("formality", 3),
                emoji_policy=data.get("emoji_policy", "none"),
                target_len_tokens=data.get("target_len_tokens", length_target[0]),
                signature_moves=data.get("signature_moves", []),
                taboos=data.get("taboos", self.taboos[:5]),
                few_shots=few_shots_objs,
                negative_example=negative_ex,
            )
            
            return pack
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback to default pack
            return self._default_pack(intent, length_target)
    
    def _classify_intent(self, message: str, history: List[Dict[str, str]]) -> str:
        """Classify user intent (simple heuristic, can be enhanced)."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["advice", "should", "recommend", "suggest", "how to"]):
            return "advice"
        elif any(word in message_lower for word in ["story", "tell", "remember", "once"]):
            return "storytelling"
        elif any(word in message_lower for word in ["think", "opinion", "believe", "feel"]):
            return "opinion"
        elif len(message.split()) < 10:
            return "chit-chat"
        else:
            return "default"
    
    def _select_few_shots(self, intent: str, count: int) -> List[Example]:
        """Select few-shot examples matching intent."""
        matching = [ex for ex in self.examples if ex.intent == intent]
        if len(matching) >= count:
            return matching[:count]
        
        # Fall back to any examples
        if len(self.examples) >= count:
            return self.examples[:count]
        
        return []
    
    def _default_pack(self, intent: str, length_target: tuple) -> StylePolicyPack:
        """Generate default pack if LLM parsing fails."""
        return StylePolicyPack(
            tone="neutral",
            hedging_level=2,
            formality=3,
            emoji_policy="none",
            target_len_tokens=length_target[0],
            signature_moves=[],
            taboos=self.taboos[:5],
            few_shots=self._select_few_shots(intent, 2),
            negative_example=None,
        )

