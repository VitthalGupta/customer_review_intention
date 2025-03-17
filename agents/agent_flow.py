import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from logging import getLogger

logger = getLogger(__name__)


class CustomerIntentionAgent:
    def __init__(
        self,
        api_key,
        prompt,
        system_content,
        model_config={"model": "gpt-3.5-turbo", "max_tokens": 20, "temperature": 0.5},
    ):
        self.client = OpenAI(api_key=api_key)
        self.prompt = prompt
        self.system_content = system_content
        self.model_config = model_config

    def extract_attributes(self, review):
        try:
            formatted_prompt = self.prompt.format(review=self.clean_reviews(review))
            response = self.client.chat.completions.create(
                model=self.model_config["model"],
                messages=[
                    {
                        "role": "system",
                        "content": self.system_content,
                    },  # System message
                    {"role": "user", "content": formatted_prompt},  # User query
                ],
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"],
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
            return None

    def clean_reviews(self, reviews):
        try:
            reviews = re.sub(r"[^a-zA-Z0-9\s:]", "", reviews)
            return reviews.lower().strip()

        except Exception as e:
            logger.error(f"Error cleaning reviews: {e}")
            return None

    def post_process(self, text):
        """Formats extracted attributes: removes newlines, numbers, and extra spaces."""
        try:
            attributes = re.split(r"[\n,]+", text)
            attributes = [
                re.sub(r"^\d+\.\s*|-", "", attr).strip() for attr in attributes
            ]
            attributes = [attr for attr in attributes if attr]  # Remove empty strings

            return ", ".join(attributes) if attributes else "Unknown"

        except Exception as e:
            logger.error(f"Error post-processing text: {e}")
            return "Unknown"

    def return_df(self, df, limit=10):
        try:
            processed_reviews = []
            for row in tqdm(
                df.values[:limit] if limit else df.values,
                desc="Processing Reviews",
                total=limit if limit else len(df),
            ):
                if row[3] and row[3] != "N/A":
                    query = f"review title : {row[2]} :: review body : {row[3]}"
                    processed_reviews.append(
                        {
                            "review_id": row[0],
                            "author": row[1],
                            "body": row[3],
                            "delight_attribute": self.post_process(
                                self.extract_attributes(query)
                            ),
                        }
                    )

            return pd.DataFrame(processed_reviews)

        except Exception as e:
            logger.error(f"Error cleaning invalid dataframe: {e}")
            return None
