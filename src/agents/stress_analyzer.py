"""Stress Analyzer Agent - mines text and audio stress cues."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.llm import get_llm_client
from src.data.models import StressProfile, StressHotspot

try:
    import parselmouth  # type: ignore
except ImportError:
    parselmouth = None  # pragma: no cover

logger = logging.getLogger(__name__)

HOTSPOT_PROMPT = """You analyze interviews for stress signals.
Return ONLY JSON shaped like:
[
  {{
    "quote": "verbatim span",
    "reason": "brief note",
    "markers": ["hesitation", "filler"]
  }}
]
Find up to 20 strong examples. Quote exact wording, no narration.

Transcript:
<<<
{transcript}
>>>"""


class StressAnalyzer:
    """Agent that fuses transcript + acoustic stress cues."""

    def __init__(
        self,
        whisper_model: str = "base",
        max_text_hotspots: int = 20,
        audio_window: float = 3.0,
        audio_step: float = 1.5,
    ):
        self.llm = get_llm_client()
        self.whisper_model = whisper_model
        self.max_text_hotspots = max_text_hotspots
        self.audio_window = audio_window
        self.audio_step = audio_step
        self._parselmouth = parselmouth

        # Audio analysis is optional; warn once if unavailable.
        if self._parselmouth is None:
            logger.warning(
                "praat-parselmouth not installed; audio stress analysis disabled."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        persona_name: str,
        transcript_text: str,
        audio_dir: Optional[Path] = None,
        max_audio_files: int = 4,
    ) -> Optional[StressProfile]:
        """Analyze transcript/audio and return aggregated stress profile."""
        if not transcript_text.strip():
            return None

        text_hotspots = self._detect_text_hotspots(transcript_text)

        audio_hotspots: List[StressHotspot] = []
        audio_files_processed = 0
        if audio_dir and audio_dir.exists() and self._parselmouth:
            audio_hotspots, audio_files_processed = self._process_audio_dir(
                audio_dir, max_audio_files
            )
        elif audio_dir:
            logger.warning(
                "Audio directory provided (%s) but dependencies missing.",
                audio_dir,
            )

        hotspots = text_hotspots + audio_hotspots
        if not hotspots:
            return None

        summary = self._summarize_hotspots(hotspots)

        profile = StressProfile(
            persona_name=persona_name,
            summary=summary,
            hotspots=[h for h in hotspots],
            audio_files_processed=audio_files_processed,
            text_passages_scored=len(text_hotspots),
            created_at=datetime.utcnow(),
        )
        return profile

    # ------------------------------------------------------------------ #
    # Text analysis
    # ------------------------------------------------------------------ #
    def _detect_text_hotspots(self, transcript_text: str) -> List[StressHotspot]:
        """Call LLM over transcript chunks to pull textual stress cues."""
        chunks = self._chunk_text(transcript_text, max_chars=6000)
        hotspots: List[StressHotspot] = []

        for chunk in chunks:
            prompt = HOTSPOT_PROMPT.format(transcript=chunk)
            try:
                response = self.llm.call(
                    messages=[
                        {
                            "role": "system",
                            "content": "You find stress cues in interview transcripts.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=900,
                )
                data = self._safe_json(response)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Text stress extraction failed: %s", exc)
                continue

            for entry in data:
                quote = entry.get("quote", "").strip()
                if not quote:
                    continue

                markers_field = entry.get("markers", [])
                markers = self._normalize_markers(markers_field)
                reason = entry.get("reason")
                linguistic_score = self._compute_linguistic_score(", ".join(markers))
                content_score = self._compute_content_score(reason or "")
                stress_score = self._combine_scores(
                    linguistic_score, acoustic_score=0, content_score=content_score
                )

                hotspots.append(
                    StressHotspot(
                        source="text",
                        quote=quote,
                        reason=reason,
                        markers=markers,
                        scores={
                            "linguistic": linguistic_score,
                            "acoustic": 0,
                            "content": content_score,
                            "stress": stress_score,
                        },
                        metrics={},
                    )
                )

                if len(hotspots) >= self.max_text_hotspots:
                    break
            if len(hotspots) >= self.max_text_hotspots:
                break

        return hotspots

    # ------------------------------------------------------------------ #
    # Audio analysis
    # ------------------------------------------------------------------ #
    def _process_audio_dir(
        self,
        audio_dir: Path,
        max_audio_files: int,
    ) -> Tuple[List[StressHotspot], int]:
        if not self._parselmouth:
            return [], 0

        audio_hotspots: List[StressHotspot] = []
        processed = 0

        for wav_path in sorted(audio_dir.glob("*.wav")):
            if processed >= max_audio_files:
                break
            try:
                file_hotspots = self._scan_audio_file(wav_path)
                audio_hotspots.extend(file_hotspots)
                processed += 1
            except Exception as exc:  # pragma: no cover - audio optional
                logger.warning("Audio stress analysis failed for %s: %s", wav_path, exc)

        return audio_hotspots, processed

    def _scan_audio_file(self, wav_path: Path) -> List[StressHotspot]:
        """Scan a single wav file for acoustic spikes."""
        if not self._parselmouth:
            return []

        sound = self._parselmouth.Sound(str(wav_path))
        baseline = self._compute_global_baseline(sound)

        hotspots: List[StressHotspot] = []
        duration = sound.duration
        t = 0.0
        while t + self.audio_window <= duration:
            stats = self._analyze_segment(sound, t, t + self.audio_window)
            if not stats:
                t += self.audio_step
                continue
            deltas = {
                f"{key}_delta": round(stats[key] - baseline.get(key, stats[key]), 2)
                for key in baseline
                if key in stats
            }
            acoustic_score = self._compute_acoustic_score_from_deltas(deltas)
            if acoustic_score >= 2:
                reason = self._describe_acoustic_reason(deltas)
                stress_score = self._combine_scores(
                    linguistic_score=0,
                    acoustic_score=acoustic_score,
                    content_score=1,
                )
                metrics = {**stats, **deltas}
                hotspots.append(
                    StressHotspot(
                        source="audio",
                        quote=None,
                        reason=reason,
                        markers=["acoustic_hotspot"],
                        audio_file=wav_path.name,
                        start_sec=round(t, 2),
                        end_sec=round(t + self.audio_window, 2),
                        scores={
                            "linguistic": 0,
                            "acoustic": acoustic_score,
                            "content": 1,
                            "stress": stress_score,
                        },
                        metrics=metrics,
                    )
                )
            t += self.audio_step

        return hotspots

    # ------------------------------------------------------------------ #
    # Summaries + helpers
    # ------------------------------------------------------------------ #
    def _summarize_hotspots(self, hotspots: List[StressHotspot]) -> Dict[str, Any]:
        if not hotspots:
            return {}

        stress_scores = [
            h.scores.get("stress", 0) for h in hotspots if h.scores.get("stress") is not None
        ]
        markers = Counter()
        for h in hotspots:
            markers.update(h.markers)

        avg_stress = round(sum(stress_scores) / len(stress_scores), 2) if stress_scores else 0
        top_markers = [marker for marker, _ in markers.most_common(5)]

        narrative_hint = ""
        if "hesitation" in top_markers or "pause" in top_markers:
            narrative_hint = "Under strain the persona slows down, leans on pauses, and hedges."
        elif "filler" in top_markers or "fragment" in top_markers:
            narrative_hint = "Stress nudges the persona into clipped fragments and filler words."
        elif "acoustic_hotspot" in top_markers:
            narrative_hint = "Voice spikes in pitch and loudness when the persona is pressured."

        return {
            "avg_stress_score": avg_stress,
            "top_markers": top_markers,
            "hotspot_count": len(hotspots),
            "narrative_hint": narrative_hint,
        }

    def _chunk_text(self, text: str, max_chars: int = 6000) -> List[str]:
        clean_text = text.strip()
        if len(clean_text) <= max_chars:
            return [clean_text]

        chunks: List[str] = []
        start = 0
        while start < len(clean_text):
            end = min(len(clean_text), start + max_chars)
            chunk = clean_text[start:end]
            chunks.append(chunk)
            start = end
        return chunks

    @staticmethod
    def _safe_json(raw: str) -> List[Dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            data = json.loads(text.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            logger.warning("Stress hotspot JSON parsing failed.")
        return []

    @staticmethod
    def _normalize_markers(field: any) -> List[str]:
        if isinstance(field, list):
            return [str(item).strip() for item in field if str(item).strip()]
        if isinstance(field, str):
            return [item.strip() for item in field.split(",") if item.strip()]
        return []

    # ------------------------------------------------------------------ #
    # Metrics copied from reference workflow
    # ------------------------------------------------------------------ #
    def _analyze_segment(
        self,
        sound: "parselmouth.Sound",  # type: ignore[name-defined]
        start: float,
        end: float,
    ) -> Dict[str, float]:
        segment = sound.extract_part(from_time=start, to_time=end, preserve_times=False)
        duration = max(end - start, 1e-6)

        try:
            pitch_obj = segment.to_pitch(pitch_floor=75, pitch_ceiling=500)
            pitch_values = pitch_obj.selected_array["frequency"]
            pitch_values = pitch_values[pitch_values > 0]
            if len(pitch_values) > 0:
                pitch_mean = float(pitch_values.mean())
                pitch_range = float(pitch_values.max() - pitch_values.min())
            else:
                pitch_mean = math.nan
                pitch_range = math.nan
        except Exception:
            pitch_mean = math.nan
            pitch_range = math.nan
            pitch_values = []

        try:
            point_process = self._parselmouth.praat.call(
                segment, "To PointProcess (periodic, cc)", 75, 500
            )
            jitter = (
                self._parselmouth.praat.call(
                    point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
                )
                * 100
            )
        except Exception:
            jitter = math.nan

        try:
            intensity_obj = self._parselmouth.praat.call(segment, "To Intensity", 75, 0.0, True)
            times = intensity_obj.xs()
            intensity_values = intensity_obj.values.T.flatten()
            valid = intensity_values[intensity_values > -40]
            if len(valid) > 0:
                intensity_mean = float(valid.mean())
                intensity_range = float(valid.max() - valid.min())
            else:
                intensity_mean = math.nan
                intensity_range = math.nan
        except Exception:
            intensity_mean = math.nan
            intensity_range = math.nan
            times = []
            intensity_values = []

        if hasattr(pitch_values, "shape"):
            try:
                voiced = pitch_values > 0
                speaking_rate = float(voiced.sum() / duration)
            except Exception:
                speaking_rate = math.nan
        else:
            speaking_rate = math.nan

        silence_threshold = -30
        min_pause = 0.15
        pause_count = 0
        pause_total = 0.0
        in_pause = False
        pause_start = None

        if len(times) and len(intensity_values):
            for t, val in zip(times, intensity_values):
                if val < silence_threshold:
                    if not in_pause:
                        in_pause = True
                        pause_start = t
                else:
                    if in_pause:
                        pause_dur = t - pause_start
                        if pause_dur >= min_pause:
                            pause_count += 1
                            pause_total += pause_dur
                        in_pause = False
            if in_pause and pause_start is not None:
                pause_dur = times[-1] - pause_start
                if pause_dur >= min_pause:
                    pause_count += 1
                    pause_total += pause_dur

        pause_ratio = float(pause_total / duration) if duration > 0 else math.nan

        try:
            harmonicity = self._parselmouth.praat.call(
                segment, "To Harmonicity (cc)", 75, 0.1, 1.0
            )
            hnr = float(self._parselmouth.praat.call(harmonicity, "Get mean", 0, 0))
        except Exception:
            hnr = math.nan

        return {
            "pitch_mean_hz": pitch_mean,
            "pitch_range_hz": pitch_range,
            "jitter_local_pct": jitter,
            "intensity_db": intensity_mean,
            "intensity_range_db": intensity_range,
            "speaking_rate_syl_per_sec": speaking_rate,
            "pause_ratio": pause_ratio,
            "pause_count": float(pause_count),
            "hnr_db": hnr,
        }

    def _compute_global_baseline(
        self,
        sound: "parselmouth.Sound",  # type: ignore[name-defined]
        window: float = 3.0,
        step: float = 1.5,
    ) -> Dict[str, float]:
        duration = sound.duration
        stats_list: List[Dict[str, float]] = []

        t = 0.0
        while t + window <= duration:
            stats = self._analyze_segment(sound, t, t + window)
            if any(math.isnan(v) for v in stats.values()):
                t += step
                continue
            stats_list.append(stats)
            t += step

        if not stats_list:
            return self._analyze_segment(sound, 0.0, min(window, duration))

        baseline: Dict[str, float] = {}
        keys = stats_list[0].keys()
        for key in keys:
            values = sorted(s[key] for s in stats_list)
            n = len(values)
            if n % 2 == 1:
                baseline[key] = float(values[n // 2])
            else:
                baseline[key] = float((values[n // 2 - 1] + values[n // 2]) / 2)
        return baseline

    @staticmethod
    def _compute_linguistic_score(markers_str: str) -> int:
        if not markers_str:
            return 0
        markers = [m.strip().lower() for m in markers_str.split(",") if m.strip()]
        if not markers:
            return 0

        mild = {"filler", "pause", "hesitation"}
        moderate = {"repetition", "self-correction", "fragment", "hedging"}
        strong = {"speech_block", "disorganized", "racing_speech"}

        score = 0
        if any(m in mild for m in markers):
            score = max(score, 1)
        if any(m in moderate for m in markers):
            score = max(score, 2)
        if any(m in strong for m in markers):
            score = max(score, 3)
        if len(markers) >= 4 and score >= 2:
            score = min(score + 1, 4)

        return score

    @staticmethod
    def _compute_content_score(reason: str) -> int:
        if not reason:
            return 0
        r = reason.lower()
        if any(word in r for word in ["calm", "relaxed", "comfortable"]):
            return 0
        if any(word in r for word in ["slightly nervous", "a bit nervous", "mildly", "a little worried"]):
            return 1
        if any(word in r for word in ["stressed", "anxious", "overwhelmed", "under pressure", "worried"]):
            return 2
        if any(word in r for word in ["very anxious", "panic", "breaking down", "distress"]):
            return 3
        if any(word in r for word in ["crisis", "cannot cope", "hopeless", "desperate"]):
            return 4
        return 2

    @staticmethod
    def _compute_acoustic_score_from_deltas(deltas: Dict[str, float]) -> int:
        pitch_delta = abs(deltas.get("pitch_mean_hz_delta", 0.0))
        jitter_delta = deltas.get("jitter_local_pct_delta", 0.0)
        intensity_delta = abs(deltas.get("intensity_db_delta", 0.0))
        pitch_range_delta = abs(deltas.get("pitch_range_hz_delta", 0.0))
        intensity_range_delta = abs(deltas.get("intensity_range_db_delta", 0.0))
        speaking_delta = abs(deltas.get("speaking_rate_syl_per_sec_delta", 0.0))
        pause_ratio_delta = deltas.get("pause_ratio_delta", 0.0)
        pause_count_delta = deltas.get("pause_count_delta", 0.0)
        hnr_delta = deltas.get("hnr_db_delta", 0.0)

        score = 0
        if pitch_delta > 30:
            score = max(score, 1)
        if pitch_delta > 60:
            score = max(score, 2)
        if pitch_range_delta > 40:
            score = max(score, 1)
        if pitch_range_delta > 80:
            score = max(score, 2)
        if jitter_delta > 0.5:
            score = max(score, 2)
        if jitter_delta > 1.0:
            score = max(score, 3)
        if intensity_delta > 3:
            score = max(score, 1)
        if intensity_delta > 6:
            score = max(score, 2)
        if intensity_delta > 10:
            score = max(score, 3)
        if intensity_range_delta > 4:
            score = max(score, 1)
        if intensity_range_delta > 8:
            score = max(score, 2)
        if speaking_delta > 1.0:
            score = max(score, 1)
        if speaking_delta > 2.0:
            score = max(score, 2)
        if pause_ratio_delta > 0.15 or pause_count_delta > 2:
            score = max(score, 2)
        if pause_ratio_delta > 0.30 or pause_count_delta > 4:
            score = max(score, 3)
        if hnr_delta < -3:
            score = max(score, 2)
        if hnr_delta < -6:
            score = max(score, 3)

        big_anomalies = sum(
            [
                pitch_delta > 60,
                jitter_delta > 1.0,
                intensity_delta > 10,
                pause_ratio_delta > 0.3,
                hnr_delta < -6,
            ]
        )
        if big_anomalies >= 3:
            score = 4

        return min(int(score), 4)

    @staticmethod
    def _combine_scores(
        linguistic_score: int,
        acoustic_score: int,
        content_score: int,
    ) -> int:
        raw = 0.4 * linguistic_score + 0.35 * acoustic_score + 0.25 * content_score
        level = int(round(raw))
        return max(0, min(4, level))

    @staticmethod
    def _describe_acoustic_reason(deltas: Dict[str, float]) -> str:
        reasons: List[str] = []
        if abs(deltas.get("pitch_mean_hz_delta", 0.0)) > 40:
            reasons.append("pitch spiked sharply")
        if deltas.get("jitter_local_pct_delta", 0.0) > 0.8:
            reasons.append("voice stability dropped (jitter)")
        if abs(deltas.get("intensity_db_delta", 0.0)) > 6:
            reasons.append("volume swung dramatically")
        if deltas.get("pause_ratio_delta", 0.0) > 0.2:
            reasons.append("extra pauses crept in")
        if deltas.get("hnr_db_delta", 0.0) < -4:
            reasons.append("voice became noticeably breathy")

        if not reasons:
            return "Audio-based stress spike relative to baseline cadence."
        return "Audio-based stress spike: " + ", ".join(reasons) + "."
