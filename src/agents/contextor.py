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
        self.allow_follow_up_questions: bool = True
        self._load_artifacts()
        self.allow_follow_up_questions = self._infer_follow_up_permission()
    
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
        length_target_avg = int((length_target[0] + length_target[1]) / 2)
        
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
        follow_up_default = "true" if self.allow_follow_up_questions else "false"
        follow_up_guidance = (
            "This persona must not ask the clinician any questions or follow-ups."
            if not self.allow_follow_up_questions
            else "This persona may ask a gentle follow-up question when it feels natural."
        )
        
        prompt = f"""You are a style coordinator. Based on the persona profile and user's current message, craft a Style+Policy Pack that nudges the assistant toward sounding like a living, breathing human.

{profile_str}

Style Rules (excerpt):
{style_rules_str}

Taboos:
{taboos_str}

Global constraint: {follow_up_guidance}

User Message: {user_message}
Detected Intent: {intent}
Retrieved Confidence: {retrieved_confidence}

Generate a JSON Style+Policy Pack with:
- tone: evocative string describing emotional posture (e.g., "wry but warm", "softly enthusiastic")
- hedging_level: integer 0-5 (if retrieved_confidence < 0.5, set to ≥3 and bake in gentle uncertainty phrases)
- formality: integer 0-5 (lean casual unless conversation history demands otherwise)
- emoji_policy: "none", "light", or "rich"
- target_len_tokens: integer (aim for {length_target_avg} and explicitly allow ±{length_target[1] - length_target_avg} variation)
- cadence_notes: short paragraph on mixing sentence lengths, inserting natural pauses, and sprinkling conversational fillers
- signature_moves: array of 3-5 persona quirks (rhetorical questions, metaphors, callbacks, small confessions)
- taboos: array of strings (only items relevant right now, keep ≤5)
- few_shots: array of 2-3 examples (user/assistant pairs from examples) that showcase empathy and curiosity
- negative_example: optional object showing a robotic or disengaged answer to avoid
- follow_up_question_required: boolean (default {follow_up_default}); set to false if the persona must not ask questions

The assistant must:
1. Acknowledge the user's emotional weather before giving advice or facts.
2. Mirror or reference at least one detail the user previously shared if available.
3. When follow_up_question_required is true, ask a natural follow-up question; when false, avoid asking any questions and close with reflection instead.
4. Avoid stiff AI telltales such as "As an AI" or "Based on the data".
5. Keep language grounded, human, and lightly imperfect (contractions, occasional fragments).

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
                target_len_tokens=data.get("target_len_tokens", length_target_avg),
                cadence_notes=data.get("cadence_notes"),
                follow_up_question_required=data.get(
                    "follow_up_question_required",
                    self.allow_follow_up_questions,
                ),
                signature_moves=data.get("signature_moves", []),
                taboos=data.get("taboos", self.taboos[:5]),
                few_shots=few_shots_objs,
                negative_example=negative_ex,
            )

            return self._apply_persona_overrides(pack, user_message)
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback to default pack
            return self._default_pack(intent, length_target, length_target_avg, user_message)
    
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
    
    def _infer_follow_up_permission(self) -> bool:
        """Infer whether persona allows asking follow-up questions."""
        corpus: List[str] = []
        if self.style_rules:
            corpus.append(self.style_rules.lower())
        if self.taboos:
            corpus.extend(taboo.lower() for taboo in self.taboos)
        if self.profile and self.profile.backstory:
            corpus.append(self.profile.backstory.lower())
        
        combined = " ".join(corpus)
        disallow_markers = [
            "never ask",
            "no questions",
            "answer-only replies",
            "do not ask back",
            "don't ask me questions",
            "no follow-up questions",
        ]
        for marker in disallow_markers:
            if marker in combined:
                return False
        return True
    
    def _apply_persona_overrides(self, pack: StylePolicyPack, user_message: str) -> StylePolicyPack:
        """Clamp tuning knobs based on persona speaking style instructions."""
        if self.profile and self.profile.speaking_style:
            avg_len = self.profile.speaking_style.avg_sentence_len
            if avg_len and len(avg_len) == 2:
                _, max_words = avg_len
                # Aim for at most ~two short sentences worth of tokens.
                approx_tokens = max(18, int(max_words * 2 * 1.2))
                pack.target_len_tokens = min(pack.target_len_tokens, approx_tokens)

        message_word_count = max(1, len(user_message.split()))
        proportional_word_cap = max(6, min(40, int(message_word_count * 1.2) + 4))
        proportional_token_cap = int(proportional_word_cap * 1.3)
        pack.target_len_tokens = min(pack.target_len_tokens, proportional_token_cap)

        style_lower = self.style_rules.lower() if self.style_rules else ""
        if "lowercase" in style_lower or "short, lowercase" in style_lower:
            if pack.cadence_notes:
                pack.cadence_notes += " Keep responses to one or two short, lowercase sentences with natural pauses and basic punctuation."
            else:
                pack.cadence_notes = "Keep responses to one or two short, lowercase sentences with natural pauses and basic punctuation."

            lowercase_move = "keeps replies to one or two short lowercase sentences"
            if lowercase_move not in pack.signature_moves:
                pack.signature_moves = [lowercase_move] + list(pack.signature_moves)

        casual_move = "sticks to chill words with no metaphors or big vocab"
        if casual_move not in pack.signature_moves:
            pack.signature_moves.append(casual_move)

        return pack
    
    def _default_pack(self, intent: str, length_target: tuple, length_target_avg: int, user_message: str) -> StylePolicyPack:
        """Generate default pack if LLM parsing fails."""
        default_pack = StylePolicyPack(
            tone="warm",
            hedging_level=3,
            formality=2,
            emoji_policy="light",
            target_len_tokens=length_target_avg,
            cadence_notes="Blend a quick empathetic acknowledgement with longer reflective sentences. Sprinkle in gentle fillers like 'honestly' or 'you know?' and avoid sounding scripted.",
            follow_up_question_required=self.allow_follow_up_questions,
            signature_moves=(
                ["references earlier user details"]
                if not self.allow_follow_up_questions
                else ["asks a gentle follow-up question", "references earlier user details"]
            ),
            taboos=self.taboos[:5],
            few_shots=self._select_few_shots(intent, 2),
            negative_example=None,
        )
        return self._apply_persona_overrides(default_pack, user_message)
