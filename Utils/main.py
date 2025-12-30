import argparse
import logging
import os
import configparser
from language_detector import process_file # process_file is the core function

# --- Configuration Setup ---
CONFIG_FILE = 'config.ini'

def load_config():
    """Loads configuration from config.ini or provides defaults."""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    # Define default settings
    default_media_extensions = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')
    default_folder_path = None # No default path hardcoded, must be provided or derived

    # Get settings from config, using fallbacks
    media_extensions_str = config.get('DEFAULT', 'media_extensions', fallback=','.join(default_media_extensions))
    folder_path = config.get('DEFAULT', 'default_folder_path', fallback=default_folder_path)

    settings = {
        'media_extensions': tuple(ext.strip() for ext in media_extensions_str.split(',')),
        'default_folder_path': folder_path
    }
    
    return settings

# --- Logging Setup ---
# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Output to console
        # logging.FileHandler("processing.log") # Optional: Output to a file
    ]
)

def main(folder_path_to_process, media_extensions):
    """
    Iterates through all media files in the given folder and processes them
    one by one using the language detection and segregation logic.
    """
    logging.info(f"\n--- Starting batch processing in: {folder_path_to_process} ---")

    # Robust path validation
    if not os.path.isdir(folder_path_to_process):
        logging.error(f"Error: Path is not a valid folder: {folder_path_to_process}")
        return
    if not os.access(folder_path_to_process, os.R_OK):
        logging.error(f"Error: Folder path is not readable: {folder_path_to_process}")
        return

    try:
        # Iterate through all files in the directory
        for file_name in os.listdir(folder_path_to_process):
            full_file_path = os.path.join(folder_path_to_process, file_name)

            # Check if the item is a file and has a supported media extension
            if os.path.isfile(full_file_path) and full_file_path.lower().endswith(media_extensions):
                try:
                    # Step 1: Detect & segregate language (Sequential Processing)
                    # The loop waits here until process_file completes the move operation.
                    logging.info(f"Processing file: {file_name}")
                    process_file(full_file_path)
                    logging.info(f"Successfully processed: {file_name}")

                except FileNotFoundError:
                    logging.error(f"❌ File not found during processing: {file_name}")
                except PermissionError:
                    logging.error(f"❌ Permission denied while processing: {file_name}. Check file permissions.")
                except IOError as e:
                    logging.error(f"❌ I/O error while processing {file_name}: {e}")
                except Exception as e:
                    logging.error(f"❌ An unexpected error occurred while processing {file_name}: {e}", exc_info=True)
            else:
                logging.info(f"⏭️ Skipping non-file or unsupported file: {file_name}")
    except PermissionError:
        logging.error(f"Error: Permission denied when trying to list directory: {folder_path_to_process}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during directory traversal: {e}", exc_info=True)

    logging.info("\n--- Batch processing complete! All media files have been classified and moved. ---")


if __name__ == "__main__":
    config_settings = load_config()

    parser = argparse.ArgumentParser(
        description="Process media files in a folder for language detection and segregation."
    )
    parser.add_argument(
        "folder_path",
        nargs='?', # Makes it optional
        default=config_settings['default_folder_path'],
        help="Path to the folder containing media files to process."
    )

    args = parser.parse_args()

    # If no folder_path is provided via CLI and no default in config, raise an error
    if args.folder_path is None:
        logging.error("Error: No folder path provided. Please specify a folder using the command-line argument or in config.ini.")
        parser.print_help()
    else:
        # Ensure media_extensions from config are passed
        main(args.folder_path, config_settings['media_extensions'])