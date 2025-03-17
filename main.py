import os
import json
import pkgutil
import logging
import importlib
import pandas as pd
from dotenv import load_dotenv

# local Imports
import agent_query_template
from utils.evaluation import Evaluation
from utils.arg_parser import get_parser
from agents.agent_flow import CustomerIntentionAgent

# Load environment variables
load_dotenv()

# Setup logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/cli.log"),
        # logging.StreamHandler(),
    ],
)

modules = [
    module.name for module in pkgutil.iter_modules(agent_query_template.__path__)
]

# Predefined model versions
available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    args = get_parser(available_models=available_models, modules=modules).parse_args()

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

    if not args.output_file.lower().endswith(".json"):
        logger.error("Output file must have a .json extension.")
        raise ValueError("Output file must have a .json extension.")

    # Print settings
    print(f"\nUsing Model: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Review File: {args.review_file}")
    print(f"Output File: {args.output_file}")

    # Importing prompt and system content from the selected module
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

    # Convert DataFrame to correct JSON format
    reviews_json = {"reviews": response_df.to_dict(orient="records")}

    # Save the properly formatted JSON
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(reviews_json, outfile, indent=4, ensure_ascii=False)

    logger.info(f"Successfully saved output to {args.output_file}")

    # Evalutation
    evaluator = Evaluation()
    output_csv_file = args.output_file.rsplit(".", 1)[0] + ".csv"
    evaluator.evaluate(response_df, output_csv_file, column="delight_attribute")
    logger.info(f"Successfully saved evaluation output to {output_csv_file}")
    print(f"Successfully saved evaluation output to {output_csv_file}")
