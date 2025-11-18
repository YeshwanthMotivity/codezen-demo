import argparse
import os
import logging
from .language_detector import LanguageDetector

# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define supported media file extensions (should align with LanguageDetector's capabilities)
SUPPORTED_MEDIA_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm'}

def _process_single_file(file_path: str, detector: LanguageDetector):
    """
    Processes a single media file using the provided LanguageDetector.
    """
    logging.info(f"Attempting to process single file: {file_path}")
    try:
        success, message = detector.process_file(file_path)
        if success:
            logging.info(f"Successfully processed '{file_path}'. Detected language: {message}")
        else:
            logging.warning(f"Failed to process '{file_path}': {message}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing '{file_path}': {e}")

def _process_folder(folder_path: str, detector: LanguageDetector):
    """
    Walks through a folder, identifies supported media files, and processes each.
    """
    logging.info(f"Starting to process folder: {folder_path}")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1].lower()
            full_file_path = os.path.join(root, file_name)

            if file_extension in SUPPORTED_MEDIA_EXTENSIONS:
                logging.debug(f"Found supported media file: {full_file_path}")
                try:
                    success, _ = detector.process_file(full_file_path)
                    if success:
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logging.error(f"Error processing file '{full_file_path}': {e}")
                    error_count += 1
            else:
                logging.debug(f"Skipping non-supported file in folder: {full_file_path}")
                skipped_count += 1

    logging.info(f"Finished processing folder '{folder_path}'. Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")

def main():
    """
    Main function to parse arguments and orchestrate file/folder processing.
    """
    parser = argparse.ArgumentParser(
        description="Process media files to detect language and sort them into language-specific folders."
    )
    parser.add_argument(
        "input_path", 
        type=str,
        help="Path to a single media file or a folder containing media files to be processed."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="Output",
        help="Base directory where sorted files will be saved. Subfolders for languages will be created here. Default is 'Output'."
    )
    parser.add_argument(
        "--whisper_model", 
        type=str, 
        default="small",
        help="Whisper model size to use for transcription (e.g., 'tiny', 'base', 'small', 'medium', 'large'). Default is 'small'."
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    whisper_model = args.whisper_model

    if not os.path.exists(input_path):
        logging.error(f"Error: The specified input path '{input_path}' does not exist. Please provide a valid path.")
        exit(1)

    # Initialize the LanguageDetector once at the start of the application
    logging.info(f"Initializing LanguageDetector with output directory: '{output_dir}' and Whisper model: '{whisper_model}'")
    detector = LanguageDetector(output_base_dir=output_dir, whisper_model_size=whisper_model)
    
    # Check if the Whisper model was loaded successfully
    if detector.model is None:
        logging.critical("LanguageDetector could not be initialized due to Whisper model loading failure. Exiting.")
        exit(1) # Exit if the core component (Whisper model) is not ready

    if os.path.isfile(input_path):
        logging.info(f"Input is a single file. Processing '{input_path}'.")
        _process_single_file(input_path, detector)
    elif os.path.isdir(input_path):
        logging.info(f"Input is a directory. Processing contents of '{input_path}'.")
        _process_folder(input_path, detector)
    else:
        logging.error(f"Error: Path '{input_path}' is neither a file nor a directory. Please check the path.")
        exit(1)

    logging.info("All processing tasks completed successfully.")

if __name__ == "__main__":
    main()
