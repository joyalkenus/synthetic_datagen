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
OUTPUT_CSV = "categorized_dataset.csv"
BATCH_SIZE = 5

# Define Pydantic models to represent the structured output
class TopicItem(BaseModel):
    original_title: str
    topic: str
    subtopic: str

    class Config:
        extra = "forbid"  # Disallow extra fields

class TopicResponse(BaseModel):
    items: List[TopicItem]

    class Config:
        extra = "forbid"  # Disallow extra fields

def process_batch(batch):
    """
    Given a batch of items, call the OpenAI API to determine the topic and subtopic.
    """
    print("Preparing prompt for the model...")
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. We have social media post titles. "
            "For each given title, identify a general 'topic' and a more specific 'subtopic'. Remember that these are posts that user interacted with, so topic and subtopic must show the user's preferences. "
            "Return the data as JSON conforming to the provided schema. "
            "If unsure, pick the best guess, but always provide a topic and subtopic.\n\n"
            "You will receive 5 titles.\n"
            "For each title, return:\n"
            "- original_title: exactly the given title\n"
            "- topic: a short, general category (e.g., 'Technology', 'Health', 'Politics')\n"
            "- subtopic: a more specific category related to the topic\n\n"
            "Output must be JSON and must conform to the schema.\n"
            "Noextra commentary.\n"
        )
    }

    user_msg_content = "Here are the 5 titles:\n\n"
    for item in batch:
        user_msg_content += f"title: {item['title']}\n\n"
    user_msg_content += (
        "Please return a JSON object with an 'items' field containing these 5 objects. "
        "Each object must have 'original_title', 'topic', and 'subtopic'."
    )

    print("Sending request to the model...")
    try:
        # Use the parse method to get a parsed object directly
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[system_msg, {"role": "user", "content": user_msg_content}],
            response_format=TopicResponse,
            temperature=0.7
        )
        print("Received and parsed response from the model.")

        # Extract parsed data
        parsed = completion.choices[0].message.parsed
        items = parsed.items

        # items should correspond one-to-one with the batch titles
        expanded_rows = []
        for model_item, original_row in zip(items, batch):
            expanded_rows.append({
                "Index": original_row["Index"],
                "title": model_item.original_title,
                "topic": model_item.topic,
                "subtopic": model_item.subtopic,
                "file_name": original_row["file_name"],
                "Links": original_row["Links"],
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

    # This will hold all categorized rows
    all_expanded_rows = []

    # Process in batches of 5
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
            expanded = process_batch(batch)
            all_expanded_rows.extend(expanded)
            print(f"Successfully processed batch starting at index {i}")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            continue

    # If no rows were processed successfully
    if not all_expanded_rows:
        print("No data processed successfully. Exiting.")
        return

    print("All batches processed. Converting results to DataFrame...")
    df = pd.DataFrame(all_expanded_rows)

    # Ensure columns are in the desired order
    desired_columns = ["Index", "title", "topic", "subtopic", "file_name", "Links", "Preference"]
    if all(col in df.columns for col in desired_columns):
        df = df[desired_columns]

    print("Saving results to CSV...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Categorization complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()