from typing import List, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import json


# Define speaker personas
SPEAKER_PERSONAS = {
    "research_analyst": {
        "name": "Dr. Sarah Chen",
        "style": "analytical, data-driven, uses statistics and research findings",
        "voice": "professional and measured"
    },
    "business_enthusiast": {
        "name": "Marcus Johnson",
        "style": "practical, strategic, focuses on real-world applications",
        "voice": "energetic and engaging"
    },
    "tech_analyst": {
        "name": "Alex Rivera",
        "style": "technical, innovative, explains complex concepts simply",
        "voice": "curious and enthusiastic"
    },
    "industry_expert": {
        "name": "Diana Foster",
        "style": "experienced, insightful, provides context and perspective",
        "voice": "warm and authoritative"
    }
}

# Map speaker display names to Kokoro voice IDs available in the voices/ directory
# These voice IDs should match filenames (without .pt) in the `voices/` folder.
SPEAKER_TO_VOICE = {
    "Dr. Sarah Chen": "af_sarah",
    "Marcus Johnson": "am_michael",
    "Alex Rivera": "am_adam",
    "Diana Foster": "af_nicole"
}


class PodcastState(TypedDict):
    """State for the podcast generation graph"""
    document_text: str
    speakers: List[str]
    key_points: List[str]
    script_outline: List[Dict]
    full_script: List[Dict]
    current_speaker_index: int


class PodcastScriptGenerator:
    def __init__(self, model_name: str = "phi3:latest"):
        """
        Initialize the podcast script generator with LangGraph agents
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.llm = Ollama(model=model_name, temperature=0.7)
        self.workflow = self._build_workflow()

    def _sanitize_document_text(self, text: str, aggressive: bool = False) -> str:
        """Sanitize extracted PDF text to remove bookmarks and noisy boilerplate."""
        if not text:
            return text

        # Normalize whitespace
        cleaned = text.replace('\r', '\n')
        lines = [ln for ln in cleaned.splitlines()]

        def keep_line(ln: str) -> bool:
            low = ln.lower()
            if 'pdf bookmark' in low or 'bookmark' in low:
                return False
            if 'page' in low and 'of' in low and len(ln.split()) <= 6:
                return False
            return True

        if aggressive:
            lines = [ln for ln in lines if keep_line(ln) and len(ln) < 300]
        else:
            lines = [ln for ln in lines if keep_line(ln)]

        # Collapse multiple blank lines
        out_lines = []
        prev_blank = False
        for ln in lines:
            if not ln.strip():
                if not prev_blank:
                    out_lines.append('')
                prev_blank = True
            else:
                out_lines.append(ln.strip())
                prev_blank = False

        result = '\n'.join(out_lines).strip()

        # Trim length to avoid overwhelming the model
        MAX_CHARS = 20000
        if len(result) > MAX_CHARS:
            result = result[:MAX_CHARS]

        return result
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for podcast generation"""
        
        workflow = StateGraph(PodcastState)
        
        # Add nodes for each agent
        workflow.add_node("extract_key_points", self._extract_key_points)
        workflow.add_node("create_outline", self._create_outline)
        workflow.add_node("generate_dialogue", self._generate_dialogue)
        workflow.add_node("add_transitions", self._add_transitions)
        
        # Define the workflow edges
        workflow.set_entry_point("extract_key_points")
        workflow.add_edge("extract_key_points", "create_outline")
        workflow.add_edge("create_outline", "generate_dialogue")
        workflow.add_edge("generate_dialogue", "add_transitions")
        workflow.add_edge("add_transitions", END)
        
        return workflow.compile()
    
    def _extract_key_points(self, state: PodcastState) -> Dict:
        """Agent 1: Extract key points from the document"""
        
        prompt = PromptTemplate(
            template="""You are an expert content analyzer. Extract the 5-7 most important 
            and interesting points from this document that would make for engaging podcast discussion.
            
            Document:
            {document}
            
            Return only a JSON list of key points, no other text.
            Format: ["point1", "point2", ...]
            """,
            input_variables=["document"]
        )
        
        response = self.llm.invoke(prompt.format(document=state["document_text"][:3000]))
        
        try:
            # Extract JSON block if present
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            key_points = json.loads(response)
        except Exception:
            # Fallback: split by newlines
            key_points = [line.strip() for line in response.split('\n') if line.strip()][:7]

        # Ensure we always return a list of short key points
        if isinstance(key_points, str):
            key_points = [kp.strip() for kp in key_points.split('\n') if kp.strip()][:7]

        # If output looks suspicious, fall back to simple extraction
        suspicious = any(len(k) > 500 or 'PDF Bookmark' in k or 'Bookmark' in k for k in key_points)
        if not key_points or suspicious:
            print("[podcast_generator] Warning: key_points extraction produced suspicious output")
            doc_lines = [line.strip() for line in state["document_text"].splitlines() if line.strip()]
            key_points = doc_lines[:7]

        # Final safety: make sure key_points is a list of strings
        key_points = [str(k)[:200] for k in key_points][:7]

        return {"key_points": key_points}
    
    def _create_outline(self, state: PodcastState) -> Dict:
        """Agent 2: Create a structured outline for the podcast

        Produce an outline that alternates between the host (first speaker) and other speakers
        so that the generated dialogue becomes conversational and engaging. Each key point will
        become a short mini-dialogue: Host introduces the point, one or more guests respond, and
        Host may add a short follow-up. Return segments as a list of {"speaker": name, "topic": text}.
        """
        
        speakers_info = [SPEAKER_PERSONAS[s] for s in state["speakers"]]
        speaker_names = [s["name"] for s in speakers_info]
        host = speaker_names[0]
        guests = speaker_names[1:] if len(speaker_names) > 1 else [speaker_names[0]]

        outline = []

        # Intro segment
        if state.get("key_points"):
            outline.append({"speaker": host, "topic": "introduction and brief welcome"})

        # For each key point create host intro + guest response (+ host follow-up optionally)
        for i, point in enumerate(state.get("key_points", [])):
            guest = guests[i % len(guests)]
            outline.append({"speaker": host, "topic": f"Introduce: {point}"})
            outline.append({"speaker": guest, "topic": f"React to: {point}"})
            # Optionally add a short host follow-up to keep it conversational
            outline.append({"speaker": host, "topic": f"Follow-up on: {point}"})

        # Closing segment
        outline.append({"speaker": host, "topic": "conclusion and sign-off"})

        return {"script_outline": outline}
    
    def _generate_dialogue(self, state: PodcastState) -> Dict:
        """Agent 3: Generate short, conversational turns for each outline segment"""

        full_script = []
        speakers_info = {SPEAKER_PERSONAS[s]["name"]: SPEAKER_PERSONAS[s] for s in state["speakers"]}

        for segment in state["script_outline"]:
            speaker_name = segment["speaker"]
            speaker_info = speakers_info.get(speaker_name, list(speakers_info.values())[0])

            # Request concise, 1-2 sentence responses to keep episodes short and punchy
            prompt = PromptTemplate(
                template="""You are {speaker_name} with style: {style}.

                Produce a short, natural, conversational line about: {topic}.
                Guidelines:
                - Keep it to 1-2 short sentences (max ~40 words)
                - Be authentic and engaging
                - Use the tone: {voice}
                - Avoid starting with 'Hello' or long introductions

                Return only the line of dialogue.
                """,
                input_variables=["speaker_name", "style", "topic", "voice"]
            )

            try:
                dialogue = self.llm.invoke(
                    prompt.format(
                        speaker_name=speaker_name,
                        style=speaker_info["style"],
                        topic=segment["topic"],
                        voice=speaker_info["voice"]
                    )
                )
            except Exception as e:
                # On LLM failure, fallback to a short template
                print(f"[podcast_generator] LLM error for {speaker_name}: {e}")
                dialogue = f"Here's a quick thought on {segment['topic']}."

            text = dialogue.strip()
            if not text:
                text = f"A quick take on: {segment['topic']}"

            full_script.append({
                "speaker": speaker_name,
                "text": text,
                "emotion": self._detect_emotion(text),
                # Include explicit voice_id so downstream audio generator can pick the right Kokoro voice
                "voice_id": SPEAKER_TO_VOICE.get(speaker_name, "af")
            })

        return {"full_script": full_script}
    
    def _add_transitions(self, state: PodcastState) -> Dict:
        """Agent 4: Add natural transitions and polish the script"""
        
        enhanced_script = []
        
        for i, segment in enumerate(state["full_script"]):
            # Add intro at the beginning
            if i == 0:
                intro = {
                    "speaker": segment["speaker"],
                    "text": f"Welcome to today's podcast! I'm {segment['speaker']}, and we're diving into some fascinating insights.",
                    "emotion": "excited"
                }
                enhanced_script.append(intro)
            
            enhanced_script.append(segment)
            
            # Add outro at the end
            if i == len(state["full_script"]) - 1:
                outro = {
                    "speaker": segment["speaker"],
                    "text": "That's all for today! Thanks for listening, and we'll catch you next time.",
                    "emotion": "warm"
                }
                enhanced_script.append(outro)
        
        return {"full_script": enhanced_script}

    def _deterministic_fallback_script(self, state: PodcastState) -> List[Dict]:
        """Create a deterministic, concise multi-speaker script when LLM workflow fails."""
        kp_out = self._extract_key_points(state)
        key_points = kp_out.get("key_points", [])[:5]

        speakers_info = [SPEAKER_PERSONAS[s] for s in state["speakers"]]
        speaker_names = [s["name"] for s in speakers_info]
        host = speaker_names[0]
        guests = speaker_names[1:] if len(speaker_names) > 1 else [speaker_names[0]]

        script = []
        script.append({"speaker": host, "text": f"Welcome back! I'm {host}. Let's dive into a few key ideas.", "emotion": "excited", "voice_id": SPEAKER_TO_VOICE.get(host, "af")})

        for i, kp in enumerate(key_points):
            guest = guests[i % len(guests)]
            script.append({"speaker": host, "text": f"Quick question — {kp}", "emotion": "curious", "voice_id": SPEAKER_TO_VOICE.get(host, "af")})
            script.append({"speaker": guest, "text": f"In short: {kp}", "emotion": "neutral", "voice_id": SPEAKER_TO_VOICE.get(guest, "af")})
            script.append({"speaker": host, "text": "Great — thanks for that perspective.", "emotion": "warm", "voice_id": SPEAKER_TO_VOICE.get(host, "af")})

        script.append({"speaker": host, "text": "That's all for today. Thanks for listening!", "emotion": "warm", "voice_id": SPEAKER_TO_VOICE.get(host, "af")})
        return script
    
    def _detect_emotion(self, text: str) -> str:
        """Simple emotion detection based on text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["exciting", "amazing", "wow", "incredible"]):
            return "excited"
        elif any(word in text_lower for word in ["concerning", "worried", "problem"]):
            return "concerned"
        elif any(word in text_lower for word in ["interesting", "fascinating", "curious"]):
            return "curious"
        elif any(word in text_lower for word in ["hello", "welcome", "thanks"]):
            return "warm"
        else:
            return "neutral"
    
    async def generate_script(self, document_text: str, speakers: List[str]) -> List[Dict]:
        """
        Generate a complete podcast script from document text
        
        Args:
            document_text: The extracted text from the PDF
            speakers: List of speaker role IDs
            
        Returns:
            List of script segments with speaker, text, and emotion
        """
        # Sanitize document text before sending to agents
        sanitized = self._sanitize_document_text(document_text)

        # Initialize state as a proper dict matching PodcastState TypedDict
        initial_state: PodcastState = {
            "document_text": sanitized,
            "speakers": speakers,
            "key_points": [],
            "script_outline": [],
            "full_script": [],
            "current_speaker_index": 0
        }

        try:
            result = self.workflow.invoke(initial_state)
        except Exception as e:
            msg = str(e)
            print(f"[podcast_generator] workflow.invoke error: {msg}")

            # Retry with aggressive sanitization if needed
            if 'Invalid state update' in msg or '__start__' in msg:
                print('[podcast_generator] Retrying with aggressive sanitization')
                sanitized2 = self._sanitize_document_text(document_text, aggressive=True)
                initial_state["document_text"] = sanitized2
                try:
                    result = self.workflow.invoke(initial_state)
                except Exception as e2:
                    print(f"[podcast_generator] Retry failed: {e2}")
                    # Manual fallback
                    return self._manual_fallback(initial_state)
            else:
                return self._manual_fallback(initial_state)

        # Validate result
        if not isinstance(result, dict) or "full_script" not in result:
            print("[podcast_generator] Invalid workflow result, using manual fallback")
            return self._manual_fallback(initial_state)
            
        return result["full_script"]
    
    def _manual_fallback(self, state: PodcastState) -> List[Dict]:
        """Manual fallback when workflow fails"""
        try:
            # Prefer deterministic short script when fallback is needed
            return self._deterministic_fallback_script(state)
            # Note: earlier we collected key points and attempted to re-run the workflow; the deterministic
            # script ensures concise, multi-speaker output even when LLM steps fail.
            
            # Step 1: extract key points
            kp_out = self._extract_key_points(state)
            state["key_points"] = kp_out.get("key_points", [])
            
            # Step 2: create outline
            outline_out = self._create_outline(state)
            state["script_outline"] = outline_out.get("script_outline", [])
            
            # Step 3: generate dialogue
            dialog_out = self._generate_dialogue(state)
            state["full_script"] = dialog_out.get("full_script", [])
            
            # Step 4: add transitions
            trans_out = self._add_transitions(state)
            
            return trans_out.get("full_script", state["full_script"])
            
        except Exception as e_manual:
            print(f"[podcast_generator] Manual fallback failed: {e_manual}")
            # Return a minimal script
            speakers_info = [SPEAKER_PERSONAS[s] for s in state["speakers"]]
            return [{
                "speaker": speakers_info[0]["name"],
                "text": "Welcome to our podcast. Today we're discussing the content from the document provided.",
                "emotion": "warm",
                "voice_id": SPEAKER_TO_VOICE.get(speakers_info[0]["name"], "af")
            }, {
                "speaker": speakers_info[0]["name"],
                "text": "Thank you for listening!",
                "emotion": "warm"
            }]