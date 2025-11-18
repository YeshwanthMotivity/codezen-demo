import os
import subprocess
import tempfile
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageProcessor:
    def __init__(self, whisper_model_size="base", output_dir="processed_files"):
        self.whisper_model_size = whisper_model_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Ensure output directory exists
        self._whisper_model = None # To load model lazily

    def _load_whisper_model(self):
        """Loads the Whisper model if not already loaded."""
        if self._whisper_model is None:
            try:
                import whisper # Import here to avoid global import issues if not installed
                logger.info(f"Loading Whisper model: {self.whisper_model_size}")
                self._whisper_model = whisper.load_model(self.whisper_model_size)
                logger.info("Whisper model loaded successfully.")
            except ImportError:
                logger.error("Whisper library not found. Please install it with 'pip install openai-whisper'")
                raise
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise
        return self._whisper_model

    def _convert_audio_to_wav(self, input_path):
        """Converts any audio file to WAV format using ffmpeg and returns path to temp WAV."""
        # Use tempfile to create a temporary WAV file
        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close() # Close immediately to allow subprocess to write to it

        logger.info(f"Converting '{input_path}' to WAV at '{temp_wav_path}'...")
        command = [
            'ffmpeg', '-y', # Overwrite output files without asking
            '-i', input_path,
            '-ar', '16000', # Sample rate 16kHz
            '-ac', '1',    # Mono channel
            '-c:a', 'pcm_s16le', # PCM 16-bit little-endian
            temp_wav_path
        ]
        try:
            process = subprocess.run(command, check=False, capture_output=True, text=True)
            if process.returncode != 0:
                logger.error(f"FFmpeg conversion failed for '{input_path}'.")
                logger.error(f"FFmpeg stdout: {process.stdout}")
                logger.error(f"FFmpeg stderr: {process.stderr}")
                raise RuntimeError(f"FFmpeg conversion failed: {process.stderr}")
            logger.info(f"FFmpeg conversion successful for '{input_path}'.")
            logger.debug(f"FFmpeg stdout: {process.stdout}")
            return temp_wav_path
        except FileNotFoundError:
            logger.error("ffmpeg command not found. Please install ffmpeg.")
            raise FileNotFoundError("ffmpeg not found. Please install it to convert audio.")
        except Exception as e:
            logger.error(f"An error occurred during FFmpeg conversion: {e}")
            raise

    def process_audio_file(self, audio_file_path):
        """
        Transcribes audio and detects language using OpenAI's Whisper.
        Saves transcription and language detection results to a JSON file.
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        model = self._load_whisper_model()
        temp_wav_path = None
        try:
            # Step 1: Convert audio to WAV (Whisper prefers 16kHz mono WAV)
            temp_wav_path = self._convert_audio_to_wav(audio_file_path)

            # Step 2: Transcribe and detect language using Whisper
            logger.info(f"Transcribing and detecting language for '{audio_file_path}' using Whisper...")
            result = model.transcribe(temp_wav_path, fp16=False) # fp16=False for CPU or compatibility
            detected_language = result.get("language")
            transcription = result.get("text", "").strip()

            logger.info(f"Language detected: {detected_language}")
            logger.info(f"Transcription: {transcription[:100]}...") # Log first 100 chars

            # Step 3: Save results to a JSON file
            original_file_name = os.path.basename(audio_file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_json_name = f"{os.path.splitext(original_file_name)[0]}_lang_detection_{timestamp}.json"
            output_json_path = os.path.join(self.output_dir, output_json_name)

            results_data = {
                "original_file": audio_file_path,
                "processed_timestamp": timestamp,
                "detected_language": detected_language,
                "transcription": transcription
            }

            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Results saved to '{output_json_path}'")

            return results_data

        except RuntimeError as e:
            logger.error(f"Audio processing failed for '{audio_file_path}': {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"An unexpected error occurred during audio processing for '{audio_file_path}': {e}")
            return {"error": str(e)}
        finally:
            # Ensure temporary WAV file is cleaned up
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
                logger.info(f"Cleaned up temporary file: {temp_wav_path}")

    def process_language(self, file_path):
        """
        Main function to detect language for a given file.
        Determines if it's an audio file and processes accordingly.
        """
        # A simple check for common audio extensions. This could be more robust, e.g., by checking MIME type.
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma']
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in audio_extensions:
            logger.info(f"Processing audio file: {file_path}")
            return self.process_audio_file(file_path)
        else:
            # Integrate langdetect for text files if needed
            try:
                from langdetect import detect
                logger.info(f"Processing non-audio file (assuming text): {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip(): # Only detect if there's actual text
                        detected_language = detect(content)
                        logger.info(f"Language detected for text file '{file_path}': {detected_language}")
                        return {
                            "original_file": file_path,
                            "processed_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "detected_language": detected_language,
                            "content_preview": content[:200]
                        }
                    else:
                        logger.warning(f"File '{file_path}' is empty or contains no detectable text.")
                        return {"error": "File is empty or contains no detectable text."}
                except Exception as e:
                    logger.error(f"Error reading or detecting language for text file '{file_path}': {e}")
                    return {"error": f"Error processing text file: {e}"}
            except ImportError:
                logger.warning("langdetect library not found. Skipping text-based language detection for non-audio files.")
                return {"error": "Unsupported file type and langdetect not installed for text processing."}
