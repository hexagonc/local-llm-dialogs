
import argparse
import logging
import os
import csv
import numpy

from ModelAssessor import ModelAssessor


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

def main(models, url, api_key):
    # Your code to test the models goes here


    for model_name in models:
        logger = setup_logging(model_name)
        min_pass_fraction = 1
        assessor = ModelAssessor(model_name, url, api_key, min_pass_fraction)
        results = assessor.assessEmbeddingModel()

        report_message = f"Assessment for model [{model_name}]: {results}"
        logger.info(report_message)
        print(report_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assess embedding models.")
    parser.add_argument('--models', nargs='+', help='A list of LLM model names to be tested', required=True)
    parser.add_argument('--url', help='The URL for the API endpoint', required=True)
    parser.add_argument('--api_key', help='The API key for authentication', required=True)

    args = parser.parse_args()
    main(args.models, args.url, args.api_key)
