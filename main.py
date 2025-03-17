import os
import json
import pkgutil
import logging
import argparse
import importlib
import pandas as pd
from dotenv import load_dotenv

# local Imports
import agent_query_template
from agents.agent_flow import CustomerIntentionAgent

load_dotenv()

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/cli.log"),
        logging.StreamHandler(),
    ],
)

modules = [
    module.name for module in pkgutil.iter_modules(agent_query_template.__path__)
]

# Predefined model versions
available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # TODO: Add argparse arguments to __init__.py
    parser = argparse.ArgumentParser(description="Review Intention CLI")
    parser.add_argument(
        "--review_file",
        type=str,
        required=True,
        help="Path to the Json file containing reviews",
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
        help="Model to use for the extraction",
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
        default=0.5,
        help="Temperature for the model",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        choices=modules,
        default="v1",
        help="Prompt to use for the model",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=10,
        help="Limit the number of reviews to process",
    )

    args = parser.parse_args()

    try:
        with open(args.review_file, "r", encoding="utf-8") as file:
            reviews = json.load(file)
        df = pd.DataFrame(reviews)
        logger.info(f"Successfully loaded {len(df)} reviews from {args.review_file}")
    except FileExistsError as e:
        logger.error(f"Error loading reviews: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading reviews: {e}")
        exit(1)

    # Print settings
    print(f"\nUsing Model: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Review File: {args.review_file}")
    print(f"Output File: {args.output_file}")

    importlib.import_module(f"agent_query_template.{args.prompt}")
    prompt = getattr(agent_query_template, args.prompt).prompt
    system_content = getattr(agent_query_template, args.prompt).system_content

    print(f"Prompt: {prompt}")
    print(f"System Content: {system_content}")

    response_df = CustomerIntentionAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        prompt=prompt,
        system_content=system_content,
        model_config={
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    ).return_df(df, limit=args.limit)

    response_df.to_csv(args.output_file, index=False)
    logger.info(f"Successfully saved output to {args.output_file}")
