import LLMDialogController
import argparse
from SupervisorAgent import SupervisorAgent
import logging
from datetime import datetime

DESKTOP_LOG_NAME = "DESKTOP-ASSISTANT"

def setup_logging(model_name):
    now = datetime.now()
    formatted = now.strftime("%Y_%m_%d_%S")

    # Create a logger for each model
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs debug and higher level messages
    log_filename = f"supervisor_results_{formatted}.txt"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler for output to the console (optional)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def main(dialog_pattern_file = None):
    base_dialog_path = "."
    dialog_controller = LLMDialogController.LLMDialogController(dialog_index_path=base_dialog_path)
    logger = setup_logging(DESKTOP_LOG_NAME)
    supervisor = SupervisorAgent(logger = logger)
    if dialog_pattern_file:
        print(f"Attempting to initialize dialog with pattern file: {dialog_pattern_file}\n\n*********************\n")
        successful_initialization = supervisor.initializeDialog(dialog_pattern_file, dialog_controller)
        if successful_initialization:
            print(f"\n************************\nSuccessfully initialized dialog from {dialog_pattern_file}\n"
                  f"Resuming user dialog"
                  f"***************\n")
        else:
            print(f"Unexpected responses received from LLM when initializing dialog from: {dialog_pattern_file}")
    else:
        print("Starting desktop chat REPL")
    print("Press [Esc] followed by [Enter] to accept input.")
    dialog_controller.chat("", True)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the chat interface with optional dialog pattern file.")

    # Add optional --dialog-pattern-file argument
    parser.add_argument(
        '--dialog-pattern-file',
        type=str,
        help='Path to the dialog pattern file'
    )


    # Parse the arguments
    args = parser.parse_args()

    # Access the dialog pattern file path
    dialog_pattern_file = args.dialog_pattern_file
    main(dialog_pattern_file)






