import argparse
from logging import getLogger

logger = getLogger(__name__)


# TODO: Add argparse arguments to __init__.py
def get_parser(available_models=None, modules=None):
    """
    Returns an argument parser for the Review Intention CLI.

    Args:
        available_models (list, optional): List of available model choices.
        modules (list, optional): List of available prompt modules.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    try:
        available_models = available_models or ["gpt-4-turbo", "gpt-3.5-turbo"]
        modules = modules or ["v1", "v2"]

        parser = argparse.ArgumentParser(description="Review Intention CLI")

        parser.add_argument(
            "--review_file",
            type=str,
            required=True,
            help="Path to the JSON file containing reviews",
        )

        parser.add_argument(
            "--output_file",
            type=str,
            required=True,
            help="Path to the output file",
        )

        parser.add_argument(
            "--model",
            type=str,
            required=False,
            choices=available_models,
            default="gpt-4-turbo",
            help="Model to use for extraction",
        )

        parser.add_argument(
            "--max_tokens",
            type=int,
            required=False,
            default=20,
            help="Max tokens for the model",
        )

        parser.add_argument(
            "--temperature",
            type=float,
            required=False,
            default=0.3,
            help="Temperature for the model",
        )

        parser.add_argument(
            "--prompt",
            type=str,
            required=False,
            choices=modules,
            default="v1",
            help="Prompt version to use",
        )

        parser.add_argument(
            "--limit",
            type=int,
            required=False,
            default=10,
            help="Limit the number of reviews to process",
        )
        logger.debug("Successfully created parser")
        return parser
    except Exception as e:
        logger.error(f"Error creating parser: {e}")
        return None
