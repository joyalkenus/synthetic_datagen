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
OUTPUT_CSV = "expanded_categorized_dataset.csv"
BATCH_SIZE = 5

# Define Pydantic models for structured output
class ExpandedItem(BaseModel):
    original_title: str
    paraphrase1: str
    paraphrase2: str
    topic: str
    subtopic: str

class BatchResponse(BaseModel):
    items: List[ExpandedItem]

def process_batch(batch):
    """
    Given a batch of items, call the OpenAI API to:
    1. Generate two paraphrases for each title
    2. Determine topic and subtopic for the original content
    """
    print("Preparing prompt for the model...")
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant which is given a list of social media post text that user has interacted with. For each social media post title, you need to:\n"
            "1. Generate two paraphrased versions that maintain the same meaning\n"
            "2. Identify a general topic and specific subtopic which shows the user's preferences\n\n"
            "For each title, return:\n"
            "- original_title: exactly as provided\n"
            "- paraphrase1: first paraphrased version\n"
            "- paraphrase2: second paraphrased version\n"
            "- topic: general category (e.g., 'Technology', 'Politics', 'Health')\n"
            "- subtopic: specific subcategory of the topic\n\n"
            "Ensure paraphrases maintain the original meaning but use different wording.\n"
            "Topics should reflect the user's interests/preferences shown in the post.\n"
            "Output must be JSON conforming to the schema.\n"
            "No additional commentary.\n"
        )
    }

    user_msg_content = "Here are the titles to process:\n\n"
    for item in batch:
        user_msg_content += f"title: {item['title']}\n\n"
    user_msg_content += (
        "Return a JSON object with an 'items' array. Each item must have "
        "'original_title', 'paraphrase1', 'paraphrase2', 'topic', and 'subtopic'."
    )

    print("Sending request to the model...")
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[system_msg, {"role": "user", "content": user_msg_content}],
            response_format=BatchResponse,
            temperature=0.7
        )
        print("Received and parsed response from the model.")

        # Extract parsed data
        parsed = completion.choices[0].message.parsed
        items = parsed.items

        # Create expanded dataset with all versions
        expanded_rows = []
        for model_item, original_row in zip(items, batch):
            # Original version
            expanded_rows.append({
                "Index": original_row["Index"],
                "title": model_item.original_title,
                "topic": model_item.topic,
                "subtopic": model_item.subtopic,
                "Links": original_row["Links"],
                "file_name": original_row["file_name"],
                "Preference": original_row["Preference"]
            })
            # Paraphrase 1
            expanded_rows.append({
                "Index": original_row["Index"],
                "title": model_item.paraphrase1,
                "topic": model_item.topic,
                "subtopic": model_item.subtopic,
                "Links": original_row["Links"],
                "file_name": original_row["file_name"],
                "Preference": original_row["Preference"]
            })
            # Paraphrase 2
            expanded_rows.append({
                "Index": original_row["Index"],
                "title": model_item.paraphrase2,
                "topic": model_item.topic,
                "subtopic": model_item.subtopic,
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

    # This will hold all expanded and categorized rows
    all_processed_rows = []

    # Process in batches
    print("Starting batch processing...")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        batch_size = len(batch)
        if batch_size == 0:
            continue
        if batch_size < BATCH_SIZE:
            print(f"Warning: Processing partial batch of size {batch_size}")

        try:
            print(f"Processing batch starting at index {i} with size {batch_size}...")
            processed = process_batch(batch)
            all_processed_rows.extend(processed)
            print(f"Successfully processed batch starting at index {i}")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            continue

    # If no rows were processed successfully
    if not all_processed_rows:
        print("No data processed successfully. Exiting.")
        return

    print("All batches processed. Converting results to DataFrame...")
    df = pd.DataFrame(all_processed_rows)

    # Ensure columns are in the desired order
    column_order = ['Index', 'title', 'topic', 'subtopic', 'Links', 'file_name', 'Preference']
    df = df[column_order]

    print("Saving results to CSV...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Processing complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 