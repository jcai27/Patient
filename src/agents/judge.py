"""Agent 4: Judge - scores responses and issues targeted edits."""
import json
from typing import Dict, Any, List
from src.utils.llm import get_llm_client
from src.data.models import JudgeScores, JudgeDecision


class Judge:
    """Agent 4: Judges responses on Factuality, Persona, Helpfulness, Safety."""
    
    def __init__(self):
        self.llm = get_llm_client()
        self.threshold = 4.0  # Minimum score to accept
    
    def judge(
        self,
        response: str,
        user_message: str,
        retrieved_notes: List[Dict[str, Any]],
        persona_profile: Dict[str, Any],
        style_pack: Dict[str, Any],
    ) -> JudgeDecision:
        """
        Judge the response and decide accept/reject with edits.
        
        Args:
            response: The styled response to judge
            user_message: Original user message
            retrieved_notes: Retrieved facts used
            persona_profile: Persona profile info
            style_pack: Style pack used
            
        Returns:
            JudgeDecision with scores and edits
        """
        # Format retrieved notes
        notes_summary = "\n".join([
            f"- [{note['fact_id']}] {note['text']}" 
            for note in retrieved_notes[:5]
        ])
        
        prompt = f"""You are a quality judge evaluating a persona chatbot response. Score it on four dimensions (1-5 scale) and decide if it should be accepted or revised.

User message: {user_message}

Retrieved facts used:
{notes_summary}

Persona profile excerpt:
- Name: {persona_profile.get('name', 'Unknown')}
- Speaking style: {persona_profile.get('speaking_style', {})}

Style requirements:
- Tone: {style_pack.get('tone', 'neutral')}
- Formality: {style_pack.get('formality', 3)}
- Taboos: {', '.join(style_pack.get('taboos', [])[:5])}

Response to judge:
{response}

Evaluation Criteria:
1. **Factuality (1-5)**: Are all claims supported by retrieved notes? Are citations correct? No hallucinations?
2. **Persona (1-5)**: Does it sound like the persona? Matches tone, formality, style?
3. **Helpfulness (1-5)**: Does it address the user's question? Useful and relevant?
4. **Safety (1-5)**: No taboo violations? Appropriate content?

Return JSON with:
{{
  "factuality": <1-5>,
  "persona": <1-5>,
  "helpfulness": <1-5>,
  "safety": <1-5>,
  "overall": <average>,
  "accept": <true if all scores >= 4.0, else false>,
  "targeted_edits": [<array of specific edit instructions if not accepting>],
  "reasoning": "<brief explanation>"
}}

If accept is false, provide 1-3 specific, actionable edits in targeted_edits. Return ONLY valid JSON."""

        messages = [{"role": "user", "content": prompt}]
        
        response_text = self.llm.call(
            messages=messages,
            temperature=0.2,  # Low temperature for consistent judging
            max_tokens=400,
        )
        
        # Parse JSON
        try:
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            data = json.loads(json_str)
            
            scores = JudgeScores(
                factuality=float(data.get("factuality", 3.0)),
                persona=float(data.get("persona", 3.0)),
                helpfulness=float(data.get("helpfulness", 3.0)),
                safety=float(data.get("safety", 5.0)),
                overall=float(data.get("overall", 3.0)),
            )
            
            accept = data.get("accept", False)
            if not accept:
                # Also check if any score is below threshold
                if (scores.factuality < self.threshold or 
                    scores.persona < self.threshold or 
                    scores.helpfulness < self.threshold or 
                    scores.safety < self.threshold):
                    accept = False
                else:
                    accept = True
            
            decision = JudgeDecision(
                accept=accept,
                scores=scores,
                targeted_edits=data.get("targeted_edits", []),
                reasoning=data.get("reasoning"),
            )
            
            return decision
        
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # Fallback: conservative rejection
            return JudgeDecision(
                accept=False,
                scores=JudgeScores(
                    factuality=3.0,
                    persona=3.0,
                    helpfulness=3.0,
                    safety=5.0,
                    overall=3.5,
                ),
                targeted_edits=["Unable to parse judge response. Please review manually."],
                reasoning=f"Parsing error: {str(e)}",
            )
    
    def apply_edits(
        self,
        original_response: str,
        edits: List[str],
        user_message: str,
    ) -> str:
        """
        Apply targeted edits to a response.
        
        Args:
            original_response: Original response
            edits: List of edit instructions
            user_message: Original user message
            
        Returns:
            Revised response
        """
        edits_str = "\n".join([f"{i+1}. {edit}" for i, edit in enumerate(edits)])
        
        prompt = f"""Apply the following edits to improve this response.

Original response:
{original_response}

Edit instructions:
{edits_str}

User message: {user_message}

Apply the edits and return the revised response. Keep all facts and citations intact."""

        messages = [{"role": "user", "content": prompt}]
        
        revised = self.llm.call(
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        
        return revised.strip()

