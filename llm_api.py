import os
import json
import requests
import pandas as pd
import sys

# -----------------------
# CONFIG
# -----------------------

# OPENROUTER_API_KEY = ...
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "qwen": "qwen/qwen3-235b-a22b",
    "llama": "meta-llama/llama-3.1-70b-instruct",
    "mistral": "mistralai/mistral-large",
    "gpt5": "openai/gpt-5.1",
    "claude": "anthropic/claude-4.5-sonnet",
}

GEN_CONFIG = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 12500,
    "seed": 42,
}

REQUIRED_COLUMNS = [
    "#ID", "FORM", "REVERTED", "LEMMA",
    "UPOS", "FEATS", "PPM", "TRANSLIT",
    "ANNOTATION", "COMMENT"
]

DELIMITER = "+"

# -----------------------
# LOAD GUIDELINES
# -----------------------

def load_guidelines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------
# BUILD BATCH PROMPT
# -----------------------

def build_batch_prompt(guidelines, rows, language, examples=None):
    prompt = f"""
Context:
You are a trained linguist with expertise in morphological segmentation.

Strictly follow these guidelines:
{guidelines}

Task:
Segment ALL input word forms.

Input:
- Language: {language}
- Data (JSON list):
{json.dumps(rows, ensure_ascii=False)}

Output format:
Return ONLY valid JSON as a LIST of objects:
[
  {{"#ID": "...", "ANNOTATION": "morph1{DELIMITER}morph2...", "COMMENT": "optional"}},
  ...
]

Rules:
- Output MUST align 1-to-1 with input rows using #ID
- Do NOT skip or reorder rows
- Do NOT change the IDs of the rows
- Do NOT include any explanations outside of the "comment" field in JSON.
- As delimiter, use ONLY {DELIMITER}. Do NOT add spaces.
- Concatenation of morphs MUST produce the form.
"""
    if len(examples) > 0:
        prompt += "Example output: \n" + str(examples)
    return prompt
# -----------------------
# API CALL
# -----------------------

def query_model(model_id, prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        **GEN_CONFIG,
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("API error:", response.text)
        return None

    return response.json()["choices"][0]["message"]["content"]

# -----------------------
# PARSE BATCH OUTPUT
# -----------------------

def extract_json(output):
    match = re.search(r'\[.*\]', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return None
    else:
        print("No JSON found")
        return None

def parse_batch_output(output):
    try:
        data = json.loads(output.replace("```json", "").replace("```", ""))
        result = {}
        for item in data:
            _id = str(item.get("#ID", "")).strip()
            annotation = str(item.get("ANNOTATION", ""))
            comment = str(item.get("COMMENT", ""))

            if " " in annotation:
                annotation = ""
                comment = "INVALID_FORMAT"

            result[_id] = (annotation, comment)

        return result

    except Exception:
        #print(Exception)
        return {}

# -----------------------
# VALIDATE TSV
# -----------------------

def validate_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

# -----------------------
# PROCESS TSV (BATCH)
# -----------------------

def annotate_tsv(file_path, guidelines, model_name, model_id, language, guidelines_type="base", n_examples=0):
    df = pd.read_csv(file_path, sep="\t", dtype=str).fillna("")
    validate_columns(df)
    df["FORM_CHARS"] = df["FORM"].apply(lambda x: "|".join(list(x)) if isinstance(x, str) else "")
    INPUT_COLUMNS = ["#ID", "FORM", "LEMMA", "UPOS", "FEATS", "TRANSLIT", "FORM_CHARS"]
    
    print(f"\nProcessing {file_path} with {model_name} (BATCH)...")
    examples = []
    if n_examples > 0:
        example_df = df.sample(n=min(n_examples, len(df)), random_state=42)
        examples = example_df[["#ID", "ANNOTATION", "COMMENT"]].to_dict(orient="records")
        df = df.drop(example_df.index).reset_index(drop=True)

    rows = df[INPUT_COLUMNS].to_dict(orient="records")

    prompt = build_batch_prompt(guidelines, rows, language, examples=examples)
    print(model_id, model_name)
    output = query_model(model_id, prompt)
    print(output)

    if output is None:
        df["COMMENT"] = "API_ERROR"
    else:
        parsed = parse_batch_output(output)

        for i in range(len(df)):
            _id = str(df.at[i, "#ID"]).strip()
            annotation, comment = parsed.get(_id, ("", "MISSING"))

            df.at[i, "ANNOTATION"] = annotation
            df.at[i, "COMMENT"] = comment

    out_path = file_path.replace(".tsv", f".{model_name}.{guidelines_type}.{n_examples}.tsv").replace("metamorphosis-test","results")
    df.to_csv(out_path, sep="\t", index=False)

    print(f"Saved: {out_path}")

# -----------------------
# MAIN
# -----------------------

def main():
    guidelines_path = sys.argv[1] #"guidelines-no-examples.txt" # "guidelines.txt"
    data_dir = "metamorphosis-test"
    guidelines_type = guidelines_path.split("/")[-1].replace(".txt", "")
    guidelines = load_guidelines(guidelines_path)
    if len(sys.argv) > 2 and sys.argv[2] == "examples":
        n_examples = 50
    else:
        n_examples = 0
 

    tsv_files = [f for f in os.listdir(data_dir) if f.endswith(".tsv")]

    print(tsv_files)
    for model_name, model_id in MODELS.items():
        for file in tsv_files:
            file_path = os.path.join(data_dir, file)
            language = file.replace(".tsv", "")
            try:
                annotate_tsv(file_path, guidelines, model_name, model_id, language, guidelines_type, n_examples)
            except Exception:
                print("ERROR: Error in " + language + " " + guidelines_type)

if __name__ == "__main__":
    main()
