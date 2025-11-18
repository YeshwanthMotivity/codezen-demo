import os
import shutil
import subprocess
import tempfile
import logging
import whisper
from langdetect import detect, LangDetectException

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.ts', '.3gp'}

class LanguageDetector:
    """
    A class to detect the language of audio in video files and sort them accordingly.
    It uses Whisper for transcription and langdetect for language identification.
    """

    def __init__(self, output_base_dir: str = "Output", whisper_model_size: str = "small"):
        """
        Initializes the LanguageDetector with a specified output directory and Whisper model size.

        Args:
            output_base_dir (str): The base directory where language-sorted folders will be created.
            whisper_model_size (str): The size of the Whisper model to load (e.g., 'tiny', 'base', 'small').
        """
        self.output_base_dir = output_base_dir
        self.whisper_model_size = whisper_model_size
        self.model = None # Whisper model instance
        self._load_whisper_model()

        # Create the base output directory if it doesn't exist
        os.makedirs(self.output_base_dir, exist_ok=True)
        logging.info(f"LanguageDetector initialized. Output base directory: {self.output_base_dir}")

    def _load_whisper_model(self):
        """
        Loads the Whisper ASR model. Handles potential errors during model loading.
        """
        try:
            logging.info(f"Attempting to load Whisper model '{self.whisper_model_size}'...")
            # Setting fp16=False can prevent issues on CPUs or non-CUDA GPUs
            self.model = whisper.load_model(self.whisper_model_size)
            logging.info(f"Whisper model '{self.whisper_model_size}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Whisper model '{self.whisper_model_size}': {e}")
            self.model = None # Ensure model is None if loading fails

    def _extract_audio(self, video_path: str) -> str | None:
        """
        Extracts audio from a video file using FFmpeg and saves it to a temporary WAV file.

        Args:
            video_path (str): The path to the input video file.

        Returns:
            str | None: The path to the temporary audio file if successful, otherwise None.
        """
        audio_path = None
        try:
            # Create a temporary WAV file that will be cleaned up automatically by the caller
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                audio_path = temp_audio_file.name
            
            logging.debug(f"Extracting audio from '{os.path.basename(video_path)}' to temporary file '{os.path.basename(audio_path)}'")
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",              # No video
                "-acodec", "pcm_s16le", # PCM 16-bit signed little-endian
                "-ar", "16000",      # 16 kHz sample rate (Whisper recommended)
                "-ac", "1",         # Mono audio
                audio_path
            ]
            
            # Execute ffmpeg command, capture output for logging, and check for errors
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.debug(f"FFmpeg stdout: {process.stdout.strip()}")
            if process.stderr.strip():
                logging.debug(f"FFmpeg stderr: {process.stderr.strip()}")
            logging.info(f"Audio extraction successful for '{os.path.basename(video_path)}'.")
            return audio_path
        except FileNotFoundError:
            logging.error("FFmpeg not found. Please install FFmpeg to enable audio extraction.")
            return None
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg command failed for '{os.path.basename(video_path)}'. Error: {e.stderr.strip()}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during audio extraction for '{os.path.basename(video_path)}': {e}")
            return None

    def _transcribe_audio(self, audio_path: str) -> str | None:
        """
        Transcribes an audio file using the loaded Whisper model.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            str | None: The transcribed text if successful, otherwise None.
        """
        if not self.model:
            logging.error("Whisper model not loaded. Cannot transcribe audio.")
            return None
        
        try:
            logging.debug(f"Transcribing audio from '{os.path.basename(audio_path)}' using Whisper model '{self.whisper_model_size}'...")
            # fp16=False for better compatibility on CPU/non-CUDA environments
            # no_speech_threshold=0.6 can help with segments with low speech presence
            result = self.model.transcribe(audio_path, verbose=False, fp16=False, no_speech_threshold=0.6)
            transcription_text = result.get("text")
            if transcription_text:
                logging.debug(f"Transcription successful for '{os.path.basename(audio_path)}': {transcription_text[:100]}...")
            else:
                logging.warning(f"Transcription resulted in empty text for '{os.path.basename(audio_path)}'.")
            return transcription_text
        except RuntimeError as e:
            logging.error(f"Whisper transcription failed for '{os.path.basename(audio_path)}' due to a runtime error: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during transcription for '{os.path.basename(audio_path)}': {e}")
            return None

    def _detect_language(self, text: str) -> str:
        """
        Detects the language of a given text string using langdetect.

        Args:
            text (str): The text to analyze.

        Returns:
            str: The detected language code (e.g., 'en', 'es'), or 'unknown' if detection fails.
        """
        if not text or len(text.strip()) < 20: # Langdetect needs sufficient text for reliable detection
            logging.warning("Text too short or empty for reliable language detection. Assigning 'unknown'.")
            return "unknown"
        try:
            # langdetect returns ISO 639-1 language codes
            lang = detect(text)
            logging.debug(f"Detected language for text snippet: '{text[:50]}...' -> {lang}")
            return lang
        except LangDetectException as e:
            logging.warning(f"Could not detect language for text: '{text[:100]}...'. Error: {e}. Assigning 'unknown'.")
            return "unknown"
        except Exception as e:
            logging.error(f"An unexpected error occurred during language detection: {e}. Assigning 'unknown'.")
            return "unknown"

    def process_file(self, file_path: str) -> tuple[bool, str]:
        """
        Processes a single video file to extract audio, transcribe it, detect language,
        and move the file to a language-specific output directory.

        Args:
            file_path (str): The absolute path to the video file.

        Returns:
            tuple[bool, str]: A tuple where the first element indicates success (True/False)
                              and the second element is either the detected language or an error message.
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False, "File not found"
        if not os.path.isfile(file_path):
            logging.error(f"Path is not a file: {file_path}")
            return False, "Path is not a file"

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in VIDEO_EXTENSIONS:
            logging.info(f"Skipping '{os.path.basename(file_path)}': Not a supported video file type.")
            return False, "Not a supported video file"

        audio_file = None
        try:
            # 1. Extract Audio
            audio_file = self._extract_audio(file_path)
            if not audio_file:
                return False, "Audio extraction failed"

            # 2. Transcribe Audio
            transcription = self._transcribe_audio(audio_file)
            if not transcription:
                return False, "Audio transcription failed or was empty"

            # 3. Detect Language
            detected_language = self._detect_language(transcription)
            
            # Determine target directory
            target_dir_name = detected_language if detected_language != "unknown" else "unknown_language"
            target_dir = os.path.join(self.output_base_dir, target_dir_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # 4. Move File
            destination_path = os.path.join(target_dir, os.path.basename(file_path))
            shutil.move(file_path, destination_path)
            logging.info(f"Successfully moved '{os.path.basename(file_path)}' to '{os.path.relpath(destination_path, self.output_base_dir)}'. Detected language: {detected_language}")
            
            return True, detected_language

        except Exception as e:
            logging.error(f"Error processing file '{file_path}': {e}", exc_info=True)
            return False, f"An unexpected error occurred: {e}"
        finally:
            # Ensure the temporary audio file is deleted
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logging.debug(f"Temporary audio file '{os.path.basename(audio_file)}' removed.")
                except OSError as e:
                    logging.warning(f"Could not remove temporary audio file '{os.path.basename(audio_file)}': {e}")

