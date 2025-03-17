# Customer Review Intention

## Introduction

Customer reviews are a valuable source of feedback for businesses, providing insights into customer satisfaction, product performance, and service quality. This project aims to analyze customer reviews and extract key attributes that indicate customer sentiment and experience.

## Problem Statement

Manually analyzing customer reviews is time-consuming and inefficient. Extracting relevant attributes such as "Fragrance," "Quality," and "Packaging" from free-text reviews is challenging due to variations in wording and semantics. This project aims to automate this process using OpenAI's GPT models and NLP.

## Data Source

The project processes JSON files containing customer reviews. Each review consists of attributes such as `review_id`, `author`, `body` (the review text), and `delight_attribute` (expected attributes).

## How to Run the Code

### Prerequisites

- **Python 3.8+** is required.
- **Install `uv` and create a virtual environment**:

  ```sh
  pip install uv
  uv sync
    ```

- **Activate the virtual environment**:

  ```sh
  source .venv/bin/activate  # On macOS/Linux
  .venv\Scripts\activate     # On Windows
  ```

- Set up OpenAI API key in a `.env` file:

  ```sh
  OPENAI_API_KEY=your_api_key_here
  ```

### Running the CLI Tool

```sh
python main.py --review_file input.json --output_file output.json --model gpt-4-turbo --temperature 0.5 --max_tokens 20 --prompt v1 --limit 10
```

### Output Formats

1. **JSON File:** Extracted attributes are saved in the specified output file.
2. **CSV File:** Attribute frequencies are saved in `output.csv` in the following format:

   ```csv
   Delight Attribute,Frequency
   Fragrance,5
   Sustainable Packaging,3
   ```

3. **Evaluation Metrics:** (To be implemented)
   - Number of correct attribute extractions
   - Number of incorrect attribute extractions
   - Overall accuracy percentage

## Attribute Extraction Approach

- **Pre-processing:** Uses both title and body of the review for attribute extraction after cleaning.
- **GPT Model:** Utilizes OpenAI's GPT models to generate attribute predictions.
- **Post-processing:** Normalizes and deduplicates extracted attributes.
  
## Clustering and Deduplication Methodology

- **Semantic Matching**: Uses NLP techniques to cluster similar attributes.
- **Normalization**: Converts attributes to lowercase and removes duplicates.
- **Synonym Handling** (Ongoing): Maps synonyms to a common attribute (e.g., "smell" → "fragrance").

## Limitations & Improvements

- Use embeddings from already classified reviews for better attribute classification and to reduce API usage.
- Enhance semantic matching by incorporating additional NLP techniques such as TF-IDF similarity and BERT embeddings.
- Create a documentation that extracts doctring from code.

## Known Issues

- The spelling of the words may be wrong due to stemmer.
- Need a fixed pool of attributes to map with the extracted attributes. That would have been possible, if we could have known the product for which the review is given. (**ASSUMPTION: The product is something related to cosmetics**)
