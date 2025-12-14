# classify_reviews_agent.py
import os
import json
import pandas as pd
import time
import cohere

# ----------------------------
# Configuration
# ----------------------------
API_KEY = os.environ.get("COHERE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set COHERE_API_KEY environment variable before running.")

MODEL = "command-a-03-2025"
INPUT_CSV = "reviews.csv"    
OUTPUT_CSV = "classified_reviews.csv"
SLEEP_BETWEEN_CALLS = 0.2

# ----------------------------
# Initialize Cohere client
# ----------------------------
co = cohere.ClientV2(api_key=API_KEY)

# ----------------------------
# Cohere-only Feedback Classifier
# ----------------------------
def classify_feedback_with_ai(review_text: str) -> dict:
    """
    Uses Cohere Command-A to perform:
    - Category classification
    - Sentiment
    - Severity rating
    - PM-friendly summary
    """

    prompt = f"""
You are an expert Product Manager assistant. Analyze the following App Store review and produce:

1. Category (choose one): 
   - Praise
   - Bug/Crash
   - Subscription/Price
   - Feature Request
   - Usability
   - Other

2. Sentiment: Positive, Neutral, or Negative.

3. Severity (1-5):  
   1 = trivial  
   3 = moderate  
   5 = critical issue requiring immediate attention.

4. Summary for a PM (1-2 clear sentences).

Respond strictly in valid JSON with keys:
category, sentiment, severity, summary

Review:
\"\"\"{review_text}\"\"\"
"""

    response = co.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2
    )

    ai_text = response.message.content[0].text

    # Parse JSON safely
    try:
        parsed = json.loads(ai_text)
    except json.JSONDecodeError:
        parsed = {
            "category": "Other",
            "sentiment": "Neutral",
            "severity": 3,
            "summary": ai_text
        }

    return parsed

# ----------------------------
# Main processing function
# ----------------------------
def classify_reviews_from_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    if "review" not in df.columns:
        raise RuntimeError("Input CSV must have a 'review' column")

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        review_text = str(row["review"]).strip()
        print(f"[{idx+1}/{total}] Processing: {review_text[:80]}{'...' if len(review_text) > 80 else ''}")

        ai_output = classify_feedback_with_ai(review_text)

        results.append({
            "review": review_text,
            "category": ai_output.get("category"),
            "sentiment": ai_output.get("sentiment"),
            "severity": ai_output.get("severity"),
            "summary": ai_output.get("summary"),
            "raw_ai_response": json.dumps(ai_output)
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\nDone! Classified {len(results)} reviews → {output_csv}")

# ----------------------------
# Run script
# ----------------------------
if __name__ == "__main__":
    classify_reviews_from_csv(INPUT_CSV, OUTPUT_CSV)
