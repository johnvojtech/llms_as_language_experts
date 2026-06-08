import os
import json
import requests
import pandas as pd
import sys

# -----------------------
# CONFIG
# -----------------------

OPENROUTER_API_KEY = "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"#"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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "qwen": "qwen/qwen3-235b-a22b", #"qwen/qwen-2.5-72b-instruct",
    "llama": "meta-llama/llama-3.1-70b-instruct",
    #"qwen": "qwen/qwen3-235b-a22b",
    "mistral": "mistralai/mistral-large",
    "gpt5": "openai/gpt-5.1",
    "claude": "anthropic/claude-4.5-sonnet",
    #"gemini": "google/gemini-2.5-pro",
}

GEN_CONFIG = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 12500,  # IMPORTANT: increase for batch output
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
    INPUT_COLUMNS = ["#ID", "FORM", "TRANSLIT", "FORM_CHARS"]
    
    print(f"\nProcessing {file_path} with {model_name} (BATCH)...")
    examples = []
    
    #fixed_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 18, 21, 22, 23, 24, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 62, 63, 64, 65, 71, 72, 73, 74, 75, 78, 81, 82, 84, 86, 88, 89, 90, 91, 92, 93, 95, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 124, 127, 128, 130, 131, 132, 134, 135, 137, 139, 140, 141, 142, 143, 145, 146, 147, 148, 150, 152, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 167, 168, 169, 170, 172, 173, 174, 176, 177, 179, 180, 181, 182, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 197, 199, 200]
    excluded_ids = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0011', '0012', '0013', '0014', '0015', '0018', '0021', '0022', '0023', '0024', '0026', '0027', '0028', '0029', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0040', '0041', '0042', '0043', '0044', '0045', '0047', '0048', '0049', '0050', '0051', '0052', '0053', '0054', '0055', '0058', '0059', '0060', '0062', '0063', '0064', '0065', '0071', '0072', '0073', '0074', '0075', '0078', '0081', '0082', '0084', '0086', '0088', '0089', '0090', '0091', '0092', '0093', '0095', '0097', '0098', '0099', '0100', '0101', '0102', '0103', '0104', '0106', '0107', '0108', '0109', '0110', '0111', '0112', '0113', '0114', '0115', '0117', '0118', '0119', '0120', '0121', '0122', '0124', '0127', '0128', '0130', '0131', '0132', '0134', '0135', '0137', '0139', '0140', '0141', '0142', '0143', '0145', '0146', '0147', '0148', '0150', '0152', '0154', '0155', '0156', '0157', '0158', '0160', '0161', '0162', '0163', '0164', '0167', '0168', '0169', '0170', '0172', '0173', '0174', '0176', '0177', '0179', '0180', '0181', '0182', '0184', '0185', '0186', '0188', '0189', '0190', '0191', '0192', '0193', '0194', '0195', '0197', '0199', '0200']

    if n_examples > 0:
    # take only rows NOT in excluded_ids
        candidate_df = df[~df["#ID"].isin(excluded_ids)]
        example_df = candidate_df.head(min(n_examples, len(candidate_df)))
        examples = example_df[["#ID", "ANNOTATION", "COMMENT"]].to_dict(orient="records")
        # remove them from main data
        df = df.drop(example_df.index).reset_index(drop=True)

#    if n_examples > 0:
#        example_df = df.sample(n=min(n_examples, len(df)), random_state=42)
#        examples = example_df[["#ID", "ANNOTATION", "COMMENT"]].to_dict(orient="records")
        # remove them from main data
#        df = df.drop(example_df.index).reset_index(drop=True)

    # Convert entire table to list of dicts
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

    out_path = file_path.replace(".tsv", f".{model_name}.{guidelines_type}.{n_examples}_no_ud.tsv").replace("metamorphosis-test","results_2")
    #df.drop("FORM_CHARS")
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
 

    tsv_files = [f for f in os.listdir(data_dir) if f.endswith(".tsv") and sys.argv[3] in f]#("zho" not in f) and ("hin" not in f) and ("mar" not in f) and "jpn" not in f and "tel" not in f] # or "ita" in f or "lat" in f or "pol" in f)]

    print(tsv_files)
    for model_name, model_id in MODELS.items():
        #if model_name != "mistral":
        #    continue
        for file in tsv_files:
            file_path = os.path.join(data_dir, file)
            language = file.replace(".tsv", "")
            try:
                annotate_tsv(file_path, guidelines, model_name, model_id, language, guidelines_type, n_examples)
            except Exception:
                print("ERROR: Error in " + language + " " + guidelines_type)

if __name__ == "__main__":
    main()
