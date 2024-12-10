import csv
import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# Input and output filenames
INPUT_CSV = "dataset.csv"
OUTPUT_CSV = "expanded_dataset.csv"
BATCH_SIZE = 5

# Define our Pydantic models for structured output
class ParaphraseItem(BaseModel):
    original_title: str
    paraphrase1: str
    paraphrase2: str

class ParaphraseResponse(BaseModel):
    items: List[ParaphraseItem]

def process_batch(batch):
    """
    Given a batch of items (each item a dict from the CSV), call the OpenAI API
    to produce paraphrased versions of their titles.
    """
    print("Preparing prompt for the model...")
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Our goal is to expand the dataset by generating paraphrased versions of the titles which contain the same topic but with different wording.\n"
            "The data are social media posts with a user preference (like/dislike). We only provide you with the post titles.\n"
            "Your job:\n"
            "1. You will receive 5 titles.\n"
            "2. For each title, return:\n"
            "   - original_title: the same title you received\n"
            "   - paraphrase1: a paraphrased sentence of the title\n"
            "   - paraphrase2: another paraphrased sentence of the title\n\n"
            "3. Make sure the paraphrases convey the same idea/topic but with different wording.\n"
            "4. Output must be JSON and must conform to the provided schema.\n"
            "5. Do not add extra commentary, just return the structure requested.\n"
        )
    }

    user_msg_content = "Here are the 5 titles:\n\n"
    for item in batch:
        user_msg_content += f"title: {item['title']}\n\n"
    user_msg_content += (
        "Please return a JSON object with an 'items' field containing these 5 objects. "
        "Each object must have 'original_title', 'paraphrase1', and 'paraphrase2'."
    )

    print("Sending request to the model...")
    try:
        # Use the parse method to get a parsed object directly
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[system_msg, {"role": "user", "content": user_msg_content}],
            response_format=ParaphraseResponse,  # Pass our Pydantic model here
            temperature=0.7
        )
        print("Received and parsed response from the model.")

        # Extract parsed data
        parsed = completion.choices[0].message.parsed
        items = parsed.items

        # items should correspond one-to-one with the batch titles
        expanded_rows = []
        for model_item, original_row in zip(items, batch):
            # Original row
            expanded_rows.append({
                "Index": original_row["Index"],
                "title": model_item.original_title,
                "Links": original_row["Links"],
                "file_name": original_row["file_name"],
                "Preference": original_row["Preference"]
            })
            # Paraphrase 1
            expanded_rows.append({
                "Index": original_row["Index"],
                "title": model_item.paraphrase1,
                "Links": original_row["Links"],
                "file_name": original_row["file_name"],
                "Preference": original_row["Preference"]
            })
            # Paraphrase 2
            expanded_rows.append({
                "Index": original_row["Index"],
                "title": model_item.paraphrase2,
                "Links": original_row["Links"],
                "file_name": original_row["file_name"],
                "Preference": original_row["Preference"]
            })

        print("Batch processed successfully.")
        return expanded_rows

    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        raise


def main():
    print("Reading input CSV...")
    # Read all data points
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["Index"] = int(r["Index"])
            r["Preference"] = int(r["Preference"])
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {INPUT_CSV}.")

    # This will hold all expanded rows across all batches
    all_expanded_rows = []

    # Process in batches of 5
    print("Starting batch processing...")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        batch_size = len(batch)
        if batch_size < BATCH_SIZE:
            print(f"Warning: Processing partial batch of size {batch_size}")

        try:
            print(f"Processing batch starting at index {i} with size {batch_size}...")
            expanded = process_batch(batch)
            all_expanded_rows.extend(expanded)
            print(f"Successfully processed batch starting at index {i}")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            continue

    print("All batches processed. Converting results to DataFrame...")
    # Convert to DataFrame
    df = pd.DataFrame(all_expanded_rows)

    print("Saving results to CSV...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Expansion complete. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()