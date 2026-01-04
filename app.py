"""
Kokoro-TTS Local Generator - Fixed for Japanese and Multi-language Support
"""

import gradio as gr
import os
import sys
import platform
from datetime import datetime
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np
import argparse
from typing import Union, List, Optional, Tuple, Dict, Any
import asyncio
from document_processor import extract_text_from_pdf
from podcast_generator import PodcastScriptGenerator, SPEAKER_PERSONAS, SPEAKER_TO_VOICE
from models import (
    list_available_voices, build_model,
    generate_speech, download_voice_files, EnhancedKPipeline,
    get_language_code_from_voice
)
import speed_dial

# Constants
MAX_TEXT_LENGTH = 5000
DEFAULT_SAMPLE_RATE = 24000
MIN_SPEED = 0.1
MAX_SPEED = 3.0
DEFAULT_SPEED = 1.0

# Define path type for consistent handling
PathLike = Union[str, Path]

# Configuration validation
def validate_sample_rate(rate: int) -> int:
    """Validate sample rate is within acceptable range"""
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        print(f"Warning: Unusual sample rate {rate}. Valid rates are {valid_rates}")
        return 24000  # Default to safe value
    return rate

# Global configuration
CONFIG_FILE = Path("tts_config.json")
DEFAULT_OUTPUT_DIR = Path("outputs")
SAMPLE_RATE = validate_sample_rate(24000)
# Additional directories
UPLOAD_DIR = Path("uploads")
SCRIPTS_DIR = Path("scripts")
PODCASTS_DIR = DEFAULT_OUTPUT_DIR / "podcasts"
for d in (UPLOAD_DIR, SCRIPTS_DIR, PODCASTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Initialize model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None

# FIXED: Updated language mapping with proper 3-character prefixes
LANG_MAP = {
    "af_": "a", "am_": "a",  # American English
    "bf_": "b", "bm_": "b",  # British English
    "jf_": "j", "jm_": "j",  # Japanese
    "zf_": "z", "zm_": "z",  # Chinese
    "ef_": "e", "em_": "e",  # Spanish
    "ff_": "f",              # French
    "hf_": "h", "hm_": "h",  # Hindi
    "if_": "i", "im_": "i",  # Italian
    "pf_": "p", "pm_": "p",  # Portuguese
}

# Store pipelines per language
pipelines = {}

def get_available_voices():
    """Get list of available voice models."""
    try:
        # Initialize model to trigger voice downloads
        global model
        if model is None:
            print("Initializing model and downloading voices...")
            model = build_model(None, device)

        voices = list_available_voices()
        if not voices:
            print("No voices found after initialization. Attempting to download...")
            download_voice_files()
            voices = list_available_voices()

        print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error getting voices: {e}")
        return []

def get_pipeline_for_voice(voice_name: str) -> EnhancedKPipeline:
    """
    Determine the language code from the voice prefix and return the associated pipeline.
    FIXED: Now properly detects Japanese voices
    """
    # Get the 3-character prefix (e.g., "jf_" from "jf_alpha")
    prefix = voice_name[:3].lower()
    
    # Get language code from mapping
    lang_code = LANG_MAP.get(prefix, "a")
    
    print(f"[INFO] Voice '{voice_name}' mapped to language code '{lang_code}'")
    
    # Create pipeline for this language if it doesn't exist
    if lang_code not in pipelines:
        print(f"[INFO] Creating new pipeline for language code '{lang_code}'")
        try:
            pipelines[lang_code] = build_model(None, device, lang_code=lang_code)
            print(f"[INFO] Pipeline created successfully for '{lang_code}'")
        except Exception as e:
            print(f"[ERROR] Failed to create pipeline for '{lang_code}': {e}")
            raise
    
    return pipelines[lang_code]

def convert_audio(input_path: PathLike, output_path: PathLike, format: str) -> Optional[PathLike]:
    """Convert audio to specified format."""
    try:
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if format.lower() == "wav":
            return input_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio = AudioSegment.from_wav(str(input_path))

        if format.lower() == "mp3":
            audio.export(str(output_path), format="mp3", bitrate="192k")
        elif format.lower() == "aac":
            audio.export(str(output_path), format="aac", bitrate="192k")
        else:
            raise ValueError(f"Unsupported format: {format}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise IOError(f"Failed to create {format} file")

        return output_path

    except Exception as e:
        print(f"Error converting audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_tts_with_logs(voice_name: str, text: str, format: str, speed: float = 1.0) -> Optional[PathLike]:
    """
    Generate TTS audio with progress logging and memory management.
    FIXED: Now properly handles Japanese and other languages
    """
    import psutil
    import gc

    try:
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 1.0:
            print(f"Warning: Low memory available ({available_gb:.1f}GB)")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create output directory
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Validate input text
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Dynamic text length limit based on memory
        MAX_CHARS = MAX_TEXT_LENGTH
        if available_gb < 2.0:
            MAX_CHARS = min(MAX_CHARS, 2000)
            print(f"Reduced text limit to {MAX_CHARS} characters due to low memory")
        
        if len(text) > MAX_CHARS:
            print(f"Warning: Text exceeds {MAX_CHARS} characters. Truncating.")
            text = text[:MAX_CHARS] + "..."

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{timestamp}"
        wav_path = DEFAULT_OUTPUT_DIR / f"{base_name}.wav"

        # FIXED: Get the appropriate pipeline for this voice's language
        print(f"\nGenerating speech for: '{text}'")
        print(f"Using voice: {voice_name}")
        
        # Get language-specific pipeline
        pipeline = get_pipeline_for_voice(voice_name)
        
        # Validate voice path
        voice_path = Path("voices").resolve() / f"{voice_name}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        # Generate speech using the correct pipeline
        try:
            print(f"[INFO] Generating with pipeline for language: {pipeline.lang_code}")
            generator = pipeline(text, voice=str(voice_path), speed=speed, split_pattern=r'\n+')

            all_audio = []
            max_segments = 100
            segment_count = 0

            for gs, ps, audio in generator:
                segment_count += 1
                if segment_count > max_segments:
                    print(f"Warning: Reached maximum segment limit ({max_segments})")
                    break

                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    print(f"Generated segment: {gs}")
                    if ps:
                        print(f"Phonemes: {ps}")

            if not all_audio:
                raise Exception("No audio generated")
                
        except Exception as e:
            print(f"Error in speech generation: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error in speech generation: {e}")

        # Combine and save audio
        if not all_audio:
            raise Exception("No audio segments were generated")

        if len(all_audio) == 1:
            final_audio = all_audio[0]
        else:
            try:
                final_audio = torch.cat(all_audio, dim=0)
            except RuntimeError as e:
                raise Exception(f"Failed to concatenate audio segments: {e}")

        # Save audio file
        try:
            sf.write(wav_path, final_audio.numpy(), SAMPLE_RATE)
            print(f"[SUCCESS] Audio saved to: {wav_path}")
        except Exception as e:
            raise Exception(f"Failed to save audio file: {e}")

        # Convert format if needed
        if format.lower() != "wav":
            output_path = DEFAULT_OUTPUT_DIR / f"{base_name}.{format.lower()}"
            result = convert_audio(wav_path, output_path, format.lower())
            if result:
                print(f"[SUCCESS] Converted to {format}: {result}")
            return result

        return wav_path

    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_podcast_from_file(uploaded_file, num_speakers, language, output_format, speed,
                             speaker1_name=None, speaker1_voice=None,
                             speaker2_name=None, speaker2_voice=None,
                             speaker3_name=None, speaker3_voice=None,
                             speaker4_name=None, speaker4_voice=None):
    statuses = []  # collect progress messages
    def log(msg: str):
        print(msg)
        statuses.append(msg)
    """Take an uploaded PDF, create a podcast script, TTS each segment, stitch into a final audio file.

    Returns: path to final audio file or None, and the generated script text for preview.
    """
    try:
        tmp = None
        if not uploaded_file:
            log("No file uploaded")
            return None, "No file uploaded", "No file uploaded"

        # Determine uploaded file path
        file_path = None
        if isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
            file_path = uploaded_file
        elif hasattr(uploaded_file, 'name') and os.path.exists(uploaded_file.name):
            file_path = uploaded_file.name
        elif isinstance(uploaded_file, dict) and 'name' in uploaded_file and os.path.exists(uploaded_file['name']):
            file_path = uploaded_file['name']
        else:
            # Try to write bytes to temp file
            try:
                tmp = DEFAULT_OUTPUT_DIR / f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(tmp, 'wb') as f:
                    # uploaded_file likely has a file-like object in uploaded_file.file
                    if hasattr(uploaded_file, 'file'):
                        data = uploaded_file.file.read()
                    elif isinstance(uploaded_file, (bytes, bytearray)):
                        data = uploaded_file
                    else:
                        data = uploaded_file.read()
                    f.write(data)
                file_path = str(tmp)
            except Exception as e:
                return None, f"Failed to save uploaded file: {e}"

        # Extract text
        log(f"Extracting text from: {file_path}")
        doc_text = extract_text_from_pdf(file_path)
        if not doc_text:
            log("Failed to extract text from the uploaded document")
            # Clean up temp file if created
            try:
                if tmp and tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return None, "Failed to extract text from the uploaded document", "Failed to extract text"

        # Save original uploaded document into uploads/ with timestamp
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            uploaded_name = Path(file_path).name
            saved_doc = UPLOAD_DIR / f"{timestamp}_{uploaded_name}"
            shutil.copy(file_path, saved_doc)
            log(f"Saved uploaded document to: {saved_doc}")
        except Exception as e:
            log(f"Warning: failed to save uploaded document: {e}")
        # Build speaker list (use first N personas)
        persona_keys = list(SPEAKER_PERSONAS.keys())
        n = int(max(1, min(int(num_speakers), len(persona_keys))))
        selected_personas = persona_keys[:n]

        # Build initial user-provided name mapping for use before full voice mapping
        user_names = [speaker1_name, speaker2_name, speaker3_name, speaker4_name]
        persona_to_user_name_pre = {}
        for i, pkey in enumerate(selected_personas):
            persona_name = SPEAKER_PERSONAS[pkey]['name']
            if i < len(user_names) and user_names[i]:
                persona_to_user_name_pre[persona_name] = user_names[i]
            else:
                persona_to_user_name_pre[persona_name] = persona_name

        # Generate script via PodcastScriptGenerator
        generator = PodcastScriptGenerator()
        try:
            log("Generating script from document text")
            script_list = asyncio.run(generator.generate_script(doc_text, selected_personas))
        except Exception as e:
            log(f"Script generation failed: {e}")
            return None, f"Script generation failed: {e}", "Script generation failed"

        if not script_list:
            log("No script was generated by the script generator")
            return None, "No script was generated", "No script generated"

        # Post-process script to make it short and engaging (greetings, short lines, <=5 minutes)
        import re

        def first_short_sentence(text: str, max_words: int = 20) -> str:
            # Split into sentences and return the first short one
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            if not sents:
                return ''
            first = sents[0]
            words = first.split()
            return ' '.join(words[:max_words]).strip()

        def short_text(text: str, max_words: int = 40) -> str:
            words = text.split()
            return ' '.join(words[:max_words]).strip()

        # Create a hook from the document first sentence
        doc_hook = first_short_sentence(doc_text, max_words=25)
        if not doc_hook:
            doc_hook = 'Let\'s walk through the key ideas.'

        # Ensure host greeting as the first segment
        host_persona_key = selected_personas[0]
        host_name = persona_to_user_name_pre.get(SPEAKER_PERSONAS[host_persona_key]['name'], SPEAKER_PERSONAS[host_persona_key]['name'])
        greeting_text = f"Welcome back! I'm {host_name}. Quick question—{doc_hook}"

        # Insert greeting at the start (keep persona reference)
        host_persona_name = SPEAKER_PERSONAS[host_persona_key]['name']
        script_processed = [{ 'persona': host_persona_name, 'speaker': host_name, 'text': greeting_text }]

        # Now process each generated segment: trust generator to produce concise lines
        for seg in script_list:
            sp = seg.get('speaker')  # original persona name
            t = seg.get('text', '').strip()
            if not t:
                continue
            # Do not perform heavy trimming here; the script generator returns concise lines
            script_processed.append({ 'persona': sp, 'speaker': persona_to_user_name_pre.get(sp, sp), 'text': t })

        # Ensure the script is not excessively long by limiting number of segments (no string cutting)
        MAX_SEGMENTS = 12
        final_script = script_processed
        if len(final_script) > MAX_SEGMENTS:
            log(f"Script has {len(final_script)} segments; truncating to first {MAX_SEGMENTS} segments to keep episode short")
            final_script = final_script[:MAX_SEGMENTS]

        # Replace script_list with final_script (each segment has 'persona','speaker','text')
        script_list = final_script
        # Assign voices for speakers based on requested language
        # Map language label to prefixes
        lang_to_prefixes = {
            'English': ['af_', 'am_', 'bf_', 'bm_'],
            'Japanese': ['jf_', 'jm_'],
            'Chinese': ['zf_', 'zm_'],
            'Spanish': ['ef_', 'em_'],
            'French': ['ff_'],
            'Hindi': ['hf_', 'hm_']
        }
        prefixes = lang_to_prefixes.get(language, ['af_', 'am_'])
        available = get_available_voices()

        # Create a pool of voices matching language prefixes
        lang_voices = [v for v in available if any(v.startswith(p) for p in prefixes)]
        if not lang_voices:
            lang_voices = available[:3]

        # Build user-provided speaker names and voices (first N entries will be used)
        user_names = [speaker1_name, speaker2_name, speaker3_name, speaker4_name]
        user_voices = [speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]

        # Map persona names to voices and user names
        persona_to_voice = {}
        persona_to_user_name = {}
        for i, pkey in enumerate(selected_personas):
            persona_name = SPEAKER_PERSONAS[pkey]['name']

            # Preferred voice: user-specified (if valid), else existing mapping, else language pool
            chosen_voice = None
            if i < len(user_voices) and user_voices[i]:
                # verify the voice exists in available voices
                if user_voices[i] in available:
                    chosen_voice = user_voices[i]
                else:
                    # allow partial matches (prefix)
                    matches = [v for v in available if v.startswith(user_voices[i])]
                    if matches:
                        chosen_voice = matches[0]

            if not chosen_voice:
                default_voice = SPEAKER_TO_VOICE.get(persona_name)
                if default_voice and any(default_voice == v or v.startswith(default_voice) for v in available):
                    chosen_voice = default_voice
                else:
                    chosen_voice = lang_voices[i % len(lang_voices)]

            persona_to_voice[persona_name] = chosen_voice

            # Map persona to user-provided name if given
            if i < len(user_names) and user_names[i]:
                persona_to_user_name[persona_name] = user_names[i]
            else:
                persona_to_user_name[persona_name] = persona_name

        # Generate audio segments
        segment_paths = []
        for i, seg in enumerate(script_list):
            persona_name = seg.get('persona') or seg.get('speaker')
            display_name = seg.get('speaker')
            text = seg.get('text', '')

            # Map persona to user-provided name & voice
            mapped_name = persona_to_user_name.get(persona_name, display_name)
            voice_id = persona_to_voice.get(persona_name) or seg.get('voice_id') or (lang_voices[0] if lang_voices else (available[0] if available else None))

            # Update segment for preview consistency
            seg['speaker'] = mapped_name
            seg['voice_id'] = voice_id

            if not voice_id:
                log(f"No voice available for segment {i} ({mapped_name}), skipping")
                continue

            log(f"Generating audio for segment {i} - speaker: {mapped_name} voice: {voice_id}")

            try:
                wav_path = generate_tts_with_logs(voice_id, text, 'wav', speed)
            except Exception as e:
                log(f"Error during TTS for segment {i}: {e}")
                wav_path = None

            if not wav_path:
                log(f"Failed to generate audio for segment {i} ({mapped_name})")
                continue

            segment_paths.append(str(wav_path))

        if not segment_paths:
            log("No audio segments were generated")
            return None, "No audio segments were generated", "No audio segments generated"

        # Stitch segments via pydub
        final_audio = AudioSegment.silent(duration=0)
        for p in segment_paths:
            try:
                seg_audio = AudioSegment.from_wav(p)
                final_audio += seg_audio
            except Exception as e:
                print(f"[podcast] Failed to append {p}: {e}")

        # Save final podcast in outputs/podcasts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"podcast_{timestamp}.{output_format.lower()}"
        PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PODCASTS_DIR / out_name
        if output_format.lower() == 'wav':
            final_audio.export(str(out_path), format='wav')
        else:
            final_audio.export(str(out_path), format=output_format.lower(), bitrate='192k')

        # Save script to scripts/ folder
        script_text = "\n\n".join([f"{s['speaker']}: {s['text']}" for s in script_list])
        try:
            script_file = SCRIPTS_DIR / f"script_{timestamp}.txt"
            with open(script_file, 'w', encoding='utf-8') as sfp:
                sfp.write("# Podcast Script\n")
                sfp.write(f"Source Document: {saved_doc}\n\n")
                for s in script_list:
                    sfp.write(f"{s['speaker']}: {s['text']}\n\n")
            log(f"Saved script to: {script_file}")
        except Exception as e:
            log(f"Warning: failed to save script: {e}")

        log(f"Podcast created: {out_path}")

        # Clean up temp file if created
        try:
            if tmp and tmp.exists():
                tmp.unlink()
        except Exception:
            pass

        return str(out_path), script_text, "\n".join(statuses)

    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"Error while creating podcast: {e}")
        return None, f"Error while creating podcast: {e}", f"Error: {e}"


    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"Error while creating podcast: {e}")
        return None, f"Error while creating podcast: {e}", f"Error: {e}"


def create_interface(server_name="127.0.0.1", server_port=7860):
    """Create and launch the Gradio interface."""

    voices = get_available_voices()
    if not voices:
        print("No voices found! Please check the voices directory.")
        return

    preset_names = speed_dial.get_preset_names()

    with gr.Blocks(title="Kokoro TTS Generator", fill_height=True) as interface:
        gr.Markdown("# Kokoro TTS Generator")
        gr.Markdown("**Now with full Japanese support!** 🇯🇵")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## TTS Controls")
            
            with gr.Column(scale=1):
                gr.Markdown("## Speed Dial")
                
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("## Podcast Generator")
                file_upload = gr.File(file_types=['.pdf'], label="Upload Document (PDF)")
                num_speakers = gr.Dropdown(choices=['1','2','3','4'], value='2', label="Number of Speakers")
                language = gr.Dropdown(choices=['English','Japanese','Chinese','Spanish','French','Hindi'], value='English', label="Podcast Language (applies to all speakers)")

                # Speaker configuration (up to 4)
                speaker1_name = gr.Textbox(label="Speaker 1 Name (Host)", value="Host")
                speaker1_voice = gr.Dropdown(choices=voices, value=voices[0] if voices else None, label="Speaker 1 Voice")

                speaker2_name = gr.Textbox(label="Speaker 2 Name", value="Speaker 1")
                speaker2_voice = gr.Dropdown(choices=voices, value=voices[1] if len(voices) > 1 else (voices[0] if voices else None), label="Speaker 2 Voice")

                speaker3_name = gr.Textbox(label="Speaker 3 Name", value="Speaker 2")
                speaker3_voice = gr.Dropdown(choices=voices, value=voices[2] if len(voices) > 2 else (voices[0] if voices else None), label="Speaker 3 Voice")

                speaker4_name = gr.Textbox(label="Speaker 4 Name", value="Speaker 3")
                speaker4_voice = gr.Dropdown(choices=voices, value=voices[3] if len(voices) > 3 else (voices[0] if voices else None), label="Speaker 4 Voice")

            with gr.Column(scale=1):
                # Podcast outputs
                podcast_output_audio = gr.Audio(label="Final Podcast")
                podcast_script_preview = gr.Textbox(lines=10, label="Generated Script", interactive=False)
                podcast_status = gr.Textbox(lines=4, label="Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    format = gr.Radio(
                        choices=["wav", "mp3", "aac"],
                        value="wav",
                        label="Output Format"
                    )
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )

            with gr.Column(scale=1):
                create_podcast_btn = gr.Button("Create Podcast")

        # Podcast button wiring
        create_podcast_btn.click(
            fn=create_podcast_from_file,
            inputs=[file_upload, num_speakers, language, format, speed,
                    speaker1_name, speaker1_voice,
                    speaker2_name, speaker2_voice,
                    speaker3_name, speaker3_voice,
                    speaker4_name, speaker4_voice],
            outputs=[podcast_output_audio, podcast_script_preview, podcast_status]
        )

    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=False
    )

def cleanup_resources():
    """Properly clean up resources when the application exits"""
    global model, pipelines

    try:
        print("Cleaning up resources...")

        # Clean up all language-specific pipelines
        if pipelines:
            for lang_code, pipeline in pipelines.items():
                print(f"Cleaning up pipeline for language: {lang_code}")
                if hasattr(pipeline, 'voices') and pipeline.voices:
                    pipeline.voices.clear()
            pipelines.clear()

        # Clean up main model
        if model is not None:
            print("Releasing model resources...")
            if hasattr(model, 'voices') and model.voices is not None:
                try:
                    model.voices.clear()
                except:
                    pass
            try:
                del model
                model = None
            except:
                pass

        # Clear CUDA memory
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass

        # Garbage collection
        import gc
        gc.collect()
        print("Cleanup completed")

    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup
import atexit
atexit.register(cleanup_resources)

import signal
import sys

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    cleanup_resources()
    sys.exit(0)

for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, signal_handler)
    except (ValueError, AttributeError):
        pass

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kokoro TTS Local Generator - Now with Japanese support!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the server on"
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        create_interface(server_name=args.host, server_port=args.port)
    finally:
        cleanup_resources()