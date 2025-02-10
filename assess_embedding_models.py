
import argparse
import logging
import os
import csv
import numpy



def setup_logging(model_name):
    # Create a logger for each model
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs debug and higher level messages
    log_filename = f"assessment_results_{model_name}.txt"
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


def get_commands(command_csv) -> dict:
    pass

def get_command_test_inputs(test_inputs_csv) -> dict:
    pass


def assess_model(model, url, api_key, command_config_dic, command_exp_dic):
    pass

def main(models, url, api_key):
    # Your code to test the models goes here


    for model in models:
        logger = setup_logging(model)
        logger.info(f"Starting assessment for model: {model}")
        logger.debug(f"Using URL: {url}")
        logger.debug(f"Using API Key: {api_key}")

        # Simulate some operations
        try:
            # Your model assessment logic here
            logger.info(f"Successfully assessed model: {model}")
        except Exception as e:
            logger.error(f"Error assessing model {model}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assess embedding models.")
    parser.add_argument('--models', nargs='+', help='A list of LLM model names to be tested', required=True)
    parser.add_argument('--url', help='The URL for the API endpoint', required=True)
    parser.add_argument('--api_key', help='The API key for authentication', required=True)

    args = parser.parse_args()
    main(args.models, args.url, args.api_key)
