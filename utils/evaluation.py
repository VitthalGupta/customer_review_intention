import csv
import nltk
from logging import getLogger
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

logger = getLogger(__name__)

class Evaluation:
    '''
    This class is used to evaluate the performance of the delight attribute extraction model.
    '''

    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        nltk.download("punkt")
        self.stemmer = PorterStemmer()
        self.model = SentenceTransformer(model_name)

    def process_delight_attributes(self,reviews):
        """Extracts and processes delight attributes from reviews."""
        try:
            attribute_counter = Counter()

            for review in reviews:
                if "delight_attribute" in review and review["delight_attribute"]:
                    attributes = [
                        self.stemmer.stem(word.strip().lower())
                        for word in review["delight_attribute"].split(",")
                    ]
                    attribute_counter.update(attributes)

            return attribute_counter
        
        except Exception as e:
            logger.error(f"Error processing delight attributes: {e}")
            return None
    
    def save_to_csv(self,attribute_counter, output_file):
        """Saves the ranked delight attributes to a CSV file."""
        try:
            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Delight Attribute", "Frequency"])
                for attribute, frequency in attribute_counter.most_common():
                    writer.writerow([attribute, frequency])
            logger.info(f"Output saved to {output_file}")
            print(f"Output saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving output to CSV: {e}")
            print(f"Error saving output to CSV: {e}")
        
    def semantic_match(self, expected, extracted, threshold=0.8):
        """Checks if extracted attribute is semantically similar to expected using embeddings."""
        try:
            expected_embedding = self.model.encode(expected, convert_to_tensor=True)
            extracted_embedding = self.model.encode(extracted, convert_to_tensor=True)
            
            similarity = util.pytorch_cos_sim(expected_embedding, extracted_embedding).item()
            return similarity >= threshold
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return False
    # TODO: Implement the evaluate_performance method
    def evaluate_performance(self, expected_reviews, extracted_reviews):
        """Evaluates performance by comparing expected and extracted attributes."""
        try:
            correct = 0
            incorrect = 0
            total = 0

            for expected_review, extracted_review in zip(expected_reviews, extracted_reviews):
                expected_attributes = [attr.strip().lower() for attr in expected_review["delight_attribute"].split(",")]
                extracted_attributes = [attr.strip().lower() for attr in extracted_review["delight_attribute"].split(",")]

                for extracted_attr in extracted_attributes:
                    total += 1
                    if any(self.semantic_match(expected_attr, extracted_attr) for expected_attr in expected_attributes):
                        correct += 1
                    else:
                        incorrect += 1

            accuracy = (correct / total) * 100 if total > 0 else 0

            return {
                "Correct Extractions": correct,
                "Incorrect Extractions": incorrect,
                "Accuracy (%)": round(accuracy, 2),
            }
        
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return None
    
    def evaluate(self, df, output_file, column="delight_attribute"):
        """Evaluates performance and saves results to a CSV file."""
        try:
            expected_reviews = df.to_dict(orient="records")
            extracted_reviews = self.process_delight_attributes(expected_reviews)
            results = self.evaluate_performance(expected_reviews, extracted_reviews)
            self.save_to_csv(extracted_reviews, output_file)
            return results
        
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return None
