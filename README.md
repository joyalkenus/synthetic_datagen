# Synthetic Data Generator/augmenter.

A set of python scripts that expands dataset entries by generating paraphrased versions of titles using OpenAI's GPT-4o-mini model. This tool is particularly useful for data augmentation in natural language processing tasks.

## Overview

This tool takes a CSV dataset containing titles and related metadata, and for each title, it generates two additional paraphrased versions while preserving the original metadata. The result is a 3x larger dataset that maintains semantic meaning while introducing linguistic variety.

## Features

- Processes data in batches of 5 entries for efficient API usage
- Generates two unique paraphrased versions for each title
- Maintains all original metadata (Index, Links, file_name, Preference)
- Structured JSON output validation
- Error handling for API responses
- UTF-8 encoding support

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages:  ```bash
  pip install pandas openai  ```

## Setup

1. Clone this repository
2. Set up your OpenAI API key as an environment variable:   ```bash
   export OPENAI_API_KEY='your-api-key-here'   ```
3. Prepare your input CSV file with the following columns:
   - Index (integer)
   - title (string)
   - Links (string)
   - file_name (string)
   - Preference (integer)

## Usage

1. Place your input CSV file named `dataset.csv` in the same directory as the script
2. Run the script:   ```bash
   python expander.py   ```
3. The expanded dataset will be saved as `expanded_dataset.csv`

## Input CSV Format

Your input CSV should follow this structure.