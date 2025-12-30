import os
import shutil
import subprocess
import tempfile
import hashlib
import logging
from typing import Dict, Tuple, List

from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# For verbose debug messages, uncomment the line below:
# logging.getLogger().setLevel(logging.DEBUG)

logging.info("Stability seed set to 0 for langdetect.")
DetectorFactory.seed = 0

# --- CONFIGURATION (Candidate for externalization to a file like config.yaml or config.json) ---
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str = os.path.join(BASE_DIR, "data", "output")

CONFIG: Dict[str, any] = {
    "LANGUAGES_DIR": os.path.join(OUTPUT_DIR, "Languages"),
    "BGM_DIR": os.path.join(OUTPUT_DIR, "bgm"),
    "REMIX_DIR": os.path.join(OUTPUT_DIR, "remix"),
    "DUPLICATES_DIR": os.path.join(OUTPUT_DIR, "duplicates"), # New directory for duplicates

    "WHISPER_MODEL_SIZE": "small",
    "WHISPER_DEVICE": "cpu",
    "WHISPER_COMPUTE_TYPE": "int8",

    "LANG_MAP": {
        "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
        "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
        "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
        "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
    },
    "MIN_SECOND_LANG_SHARE_FOR_REMIX": 0.20, # Heuristic for remix classification
    "MIN_TOTAL_LANG_SHARE_FOR_REMIX": 0.10,
    "MIN_LANGDETECT_CONFIDENCE": 0.5, # Minimum confidence for langdetect to count a language
    "MIN_WORD_COUNT_FOR_VOCALS": 10, # Minimum words to classify as having vocals
}

# Ensure output directories exist
os.makedirs(CONFIG["LANGUAGES_DIR"], exist_ok=True)
os.makedirs(CONFIG["BGM_DIR"], exist_ok=True)
os.makedirs(CONFIG["REMIX_DIR"], exist_ok=True)
os.makedirs(CONFIG["DUPLICATES_DIR"], exist_ok=True) # Create duplicates dir
logging.info(f"Output directories setup: Languages={CONFIG['LANGUAGES_DIR']}, BGM={CONFIG['BGM_DIR']}, REMIX={CONFIG['REMIX_DIR']}, Duplicates={CONFIG['DUPLICATES_DIR']}")

# Global variable to store hashes of processed files for duplicate detection
PROCESSED_FILE_HASHES: set[str] = set()

# Load Whisper model (global instance to avoid reloading)
logging.info("📥 Loading Whisper model...")
_whisper_model: WhisperModel = WhisperModel(
    CONFIG["WHISPER_MODEL_SIZE"],
    device=CONFIG["WHISPER_DEVICE"],
    compute_type=CONFIG["WHISPER_COMPUTE_TYPE"]
)
logging.info(f"✅ Whisper model loaded (Model: {CONFIG['WHISPER_MODEL_SIZE']}, Device: {CONFIG['WHISPER_DEVICE']}, Compute: {CONFIG['WHISPER_COMPUTE_TYPE']})")


# --- Helper Functions ---

def _get_file_hash(file_path: str) -> str | None:
    """
    Generates an MD5 hash for a file, handling large files efficiently.
    Returns the hash string or None if an error occurs.
    """
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536): # Read in 64KB chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"❌ Could not get hash for {file_path}. Error: {e}", exc_info=True)
        return None

def _move_file(source_path: str, destination_dir: str, target_filename: str | None = None) -> str | None:
    """
    Handles moving a file to a destination directory, renaming if a file with the same name exists.
    Returns the final destination path or None if move fails.
    """
    if target_filename is None:
        target_filename = os.path.basename(source_path)

    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, target_filename)

    base, ext = os.path.splitext(destination)
    i = 1
    # Handle filename conflicts by adding _copyX
    while os.path.exists(destination):
        destination = f"{base}_copy{i}{ext}"
        i += 1
    
    logging.debug(f"Attempting to move '{source_path}' to '{destination}'")
    try:
        shutil.move(source_path, destination)
        logging.info(f"✅ Moved '{os.path.basename(source_path)}' → '{destination}'")
        return destination
    except Exception as e:
        logging.error(f"❌ Failed to move '{source_path}' to '{destination}'. Error: {e}", exc_info=True)
        return None

def _extract_audio(video_path: str) -> str:
    """
    Extracts mono 16kHz audio from video using ffmpeg, creating a temporary file.
    Returns the path to the temporary audio file.
    Raises RuntimeError if ffmpeg extraction fails or the output file is invalid.
    """
    # Create a temporary file that will be deleted after use
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path: str = temp_audio_file.name
    temp_audio_file.close() # Close so ffmpeg can write to it

    logging.info(f"\n🎬 Extracting audio from video: '{os.path.basename(video_path)}' to temporary file '{os.path.basename(temp_audio_path)}'")
    command: List[str] = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", temp_audio_path, "-y"]
    logging.debug(f"FFmpeg command: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"❌ FFmpeg audio extraction failed for '{video_path}'.")
        logging.error(f"   Command: {' '.join(command)}")
        logging.error(f"   Stderr: {result.stderr.strip()}")
        # Ensure temp file is cleaned up even on error
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logging.debug(f"Cleaned up failed temporary audio file: {temp_audio_path}")
        raise RuntimeError(f"FFmpeg failed to extract audio: {result.stderr.strip()}")
    
    if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
        if os.path.exists(temp_audio_path): 
            os.remove(temp_audio_path)
            logging.debug(f"Cleaned up empty temporary audio file: {temp_audio_path}")
        raise RuntimeError(f"FFmpeg extraction succeeded but output file {temp_audio_path} is missing or empty.")

    logging.info(f"✅ Extracted audio → '{os.path.basename(temp_audio_path)}'")
    return temp_audio_path

def _transcribe_audio(audio_path: str) -> Tuple[str, Dict[str, float]]:
    """
    Transcribes audio using faster-whisper. Detects language per segment using langdetect
    for robust multi-language weighting for remix classification.
    Returns the full transcribed text and a dictionary of language weights.
    """
    logging.info(f"--- Starting transcription for: '{os.path.basename(audio_path)}'")
    
    segments, info = _whisper_model.transcribe(audio_path, beam_size=5, vad_filter=True)

    logging.debug(f"Overall Whisper model detected language for entire file: {info.language} (probability: {info.language_probability:.2f})")

    langs_detected: Dict[str, float] = {} # Storing accumulated text length * confidence per language
    all_segment_texts: List[str] = []
    segment_count: int = 0
    
    for segment in segments:
        segment_count += 1
        seg_text: str = segment.text.strip()
        
        if seg_text:
            all_segment_texts.append(seg_text)
            try:
                # Use langdetect for segment-level language detection
                detected_segment_langs = detect_langs(seg_text)
                log_segment_langs = [f'{d.lang}:{d.prob:.2f}' for d in detected_segment_langs]
                logging.debug(f"   🎧 Segment {segment_count}: Text '{seg_text[:50].replace('\n', ' ')}...', LangDetect: {log_segment_langs}")
                
                # Aggregate language weights by text length * confidence
                for d in detected_segment_langs:
                    lang_code: str = d.lang.split("-")[0]
                    # Only consider languages with reasonable confidence
                    if d.prob > CONFIG["MIN_LANGDETECT_CONFIDENCE"]:
                        langs_detected[lang_code] = langs_detected.get(lang_code, 0.0) + (len(seg_text) * d.prob)
            except LangDetectException:
                # langdetect might fail on very short or non-linguistic segments
                logging.debug(f"   🎧 Segment {segment_count}: LangDetect failed for segment '{seg_text[:50].replace('\n', ' ')}...'")
                # Fallback to overall Whisper detected language for weighting
                fallback_lang: str = info.language if info.language else "unknown"
                langs_detected[fallback_lang] = langs_detected.get(fallback_lang, 0.0) + len(seg_text)
            except Exception as e:
                logging.warning(f"   ⚠️ Unexpected error during segment language detection: {e}. Segment: '{seg_text[:50].replace('\n', ' ')}...'")
                fallback_lang = info.language if info.language else "unknown"
                langs_detected[fallback_lang] = langs_detected.get(fallback_lang, 0.0) + len(seg_text)
        else:
            logging.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    logging.debug(f"Total segments processed: {segment_count}")
    return " ".join(all_segment_texts), langs_detected

def _classify_file(text: str, lang_weights: Dict[str, float]) -> str:
    """
    Classifies the file based on transcription text and language weights.
    Returns the target base directory path.
    """
    cleaned_text: str = text.replace("♪", "").replace("♫", "").strip()
    word_count: int = len(cleaned_text.split())
    
    logging.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
    logging.info(f"🔍 Word count: {word_count}")

    # Sort languages by their accumulated weight (text length * confidence)
    langs_sorted: List[Tuple[str, float]] = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
    # Filter to only include recognized languages from LANG_MAP
    detected_langs: List[str] = [l for l, _ in langs_sorted if l in CONFIG['LANG_MAP']]
    
    logging.debug(f"Sorted language weights (Text Length * Confidence): {langs_sorted}")
    logging.info(f"🌍 Languages detected with weights: {lang_weights}")

    total_weight: float = sum(lang_weights.values())
    
    base_folder: str = "" # Initialize base_folder

    # Heuristic for Remix classification
    is_remix: bool = False
    if len(detected_langs) >= 2 and total_weight > 0:
        dominant_lang_weight: float = langs_sorted[0][1]
        second_lang_weight: float = langs_sorted[1][1] if len(langs_sorted) > 1 else 0

        # Remix if the second language contributes significantly compared to the dominant or total
        if (second_lang_weight >= CONFIG["MIN_SECOND_LANG_SHARE_FOR_REMIX"] * dominant_lang_weight or
            second_lang_weight >= CONFIG["MIN_TOTAL_LANG_SHARE_FOR_REMIX"] * total_weight):
            is_remix = True

    if word_count <= CONFIG["MIN_WORD_COUNT_FOR_VOCALS"] or total_weight == 0:
        base_folder = CONFIG["BGM_DIR"]
        logging.info("➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= %d or no significant speech detected.", CONFIG["MIN_WORD_COUNT_FOR_VOCALS"])
    elif is_remix:
        # Form a descriptive name for the remix folder
        pretty_langs: List[str] = [CONFIG['LANG_MAP'].get(l, l) for l, _ in langs_sorted[:min(len(langs_sorted), 3)]]
        remix_folder_name: str = f"Remix ({' + '.join(pretty_langs)})" if pretty_langs else "Remix (Multi-language)"
        base_folder = os.path.join(CONFIG["REMIX_DIR"], remix_folder_name)
        logging.info(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple prominent languages detected ({', '.join(detected_langs)}).")
    else:
        primary_lang_code: str = detected_langs[0] if detected_langs else "unknown"
        language_name: str = CONFIG['LANG_MAP'].get(primary_lang_code, primary_lang_code)
        base_folder = os.path.join(CONFIG["LANGUAGES_DIR"], language_name, "vocals")
        logging.info(f"➡️ CLASSIFICATION: VOCALS ({language_name}). Reason: Primary language detected.")

    return base_folder

# --- Main Processing Function ---

def process_file(file_path: str) -> None:
    """
    Processes a single audio/video file:
    1. Checks for duplicates.
    2. Classifies as remix if filename contains " X ".
    3. Extracts audio if it's a video.
    4. Transcribes audio and detects languages per segment.
    5. Classifies the file into BGM, Remix, or Language/Vocals.
    6. Moves the original file to its classified destination.
    7. Cleans up temporary audio files.
    """
    logging.info(f"\n=======================================================")
    logging.info(f"🚀 Processing file: '{os.path.basename(file_path)}'")
    logging.info(f"   FULL PATH: '{file_path}'")
    logging.info(f"=======================================================")
    
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        return

    # 1. Check for duplicates and move if found
    file_hash: str | None = _get_file_hash(file_path)
    if file_hash:
        logging.debug(f"File hash generated: {file_hash}")
        if file_hash in PROCESSED_FILE_HASHES:
            logging.warning("⚠️ DUPLICATE DETECTED! File with this hash has already been processed.")
            _move_file(file_path, CONFIG['DUPLICATES_DIR'], os.path.basename(file_path) + "_duplicate")
            return # Exit the function immediately, duplicate handled
        else:
            PROCESSED_FILE_HASHES.add(file_hash)
            logging.info("✅ File is unique. Proceeding with processing.")
    else:
        logging.warning(f"⚠️ Skipping duplicate check for '{file_path}' due to hashing error.")

    # 2. Filename-based Remix Check (Pre-transcription for quick classification)
    file_name_no_ext: str = os.path.splitext(os.path.basename(file_path))[0]
    if " X " in file_name_no_ext.upper():
        logging.info("✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains ' X ' pattern.")
        _move_file(file_path, CONFIG['REMIX_DIR'])
        return # Exit after moving pre-classified remix

    # 3. Prepare Audio Source (extract if video, manage temp file)
    audio_path: str = file_path
    is_temp_audio: bool = False
    file_ext: str = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
            logging.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
            audio_path = _extract_audio(file_path)
            is_temp_audio = True
        else:
            logging.debug(f"File recognized as AUDIO ({file_ext} or unknown). Using file directly.")

        # 4. Transcribe Audio
        text, lang_weights = _transcribe_audio(audio_path)
        
        # 5. Classify File Based on Transcription
        target_base_dir: str = _classify_file(text, lang_weights)

        # 6. Move Original File to its final classified location
        _move_file(file_path, target_base_dir)

    except RuntimeError as e: # Catch errors specifically from _extract_audio or ffmpeg
        logging.error(f"❌ Processing failed for '{file_path}' due to a runtime error: {e}")
    except Exception as e: # Catch any other unexpected errors during transcription or classification
        logging.error(f"❌ An unexpected error occurred during processing of '{file_path}': {e}", exc_info=True)
    finally:
        # 7. Clean up temporary audio file if created
        if is_temp_audio and os.path.exists(audio_path):
            os.remove(audio_path)
            logging.debug(f"Cleaned up temporary audio file: {audio_path}")