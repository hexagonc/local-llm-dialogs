import LLMDialogController
import argparse
from SupervisorAgent import SupervisorAgent
import logging
from datetime import datetime

DESKTOP_LOG_NAME = "DESKTOP-ASSISTANT"


def setup_logging(model_name, debug=False, verbose=False):
    now = datetime.now()
    formatted = now.strftime("%Y_%m_%d__%H_%M_%S")

    # Create a logger for each model
    logger = logging.getLogger(model_name)

    # Set the logger level based on the debug flag
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create a file handler that logs debug and higher level messages
    log_filename = f"supervisor_results_{model_name}_{formatted}.txt"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create a console handler for output to the console (optional)
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if verbose:
        console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    if verbose:
        logger.addHandler(console_handler)

    return logger


def main(dialog_pattern_file=None, debug=False, verbose=False):
    base_dialog_path = "."
    dialog_controller = LLMDialogController.LLMDialogController(dialog_index_path=base_dialog_path)

    # Pass the debug and verbose flags to setup_logging
    if debug:
        logger = setup_logging("deepseek-small", debug=debug, verbose=verbose)
    else:
        logger = None

    supervisor = SupervisorAgent(logger=logger)
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

    # Add optional --debug and --verbose flags
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode for more detailed logging'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose mode for console output'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the dialog pattern file path and debug/verbose flags
    dialog_pattern_file = args.dialog_pattern_file
    debug = args.debug
    verbose = args.verbose
    main(dialog_pattern_file, debug=debug, verbose=verbose)






