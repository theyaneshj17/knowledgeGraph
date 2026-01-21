import pandas as pd
import anthropic
import time
import re
from tqdm import tqdm

# 1️⃣ Claude API client
client = anthropic.Anthropic(
    api_key="A"  # Replace with your actual Claude API key
)

# 2️⃣ Load data and group by PMID
df = pd.read_excel("C:/Users/theya/Downloads/processed_Alzheimers_Dementia.xlsx")
grouped = df.groupby("PMID")["CH_TXT"].apply(lambda x: "\n".join(x.astype(str))).reset_index()

# ✅ Optional: limit for testing
grouped = grouped.head(2)

# 3️⃣ Prompt builder
def build_prompt(text_block):
    return f"""
You are a biomedical knowledge extraction assistant.

Your task is to extract all relevant information from the provided biomedical abstract.

1. Named Entities:
Extract all distinct entities mentioned in the text. 

2. Triples:
Extract all subject–predicate–object (SPO) relationships between those entities.

---

FORMAT:

#### ENTITIES:
List each entity once, using double quotes.
["entity A", "entity B", "entity C"]

#### TRIPLES:
Each fact must be represented in this format:
("subject", "predicate", "object")

---

TEXT:
{text_block}

#### ENTITIES:
"""

# 4️⃣ Claude call
def query_claude(prompt):
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"❌ Error for prompt:\n{prompt[:200]}...\n{e}")
        return None

# 5️⃣ Extract triples from Claude response
def parse_claude_output(pmid, text):
    triples = []
    triple_pattern = r'\("([^"]+)", "([^"]+)", "([^"]+)"\)'
    for match in re.findall(triple_pattern, text):
        triples.append({
            "PMID": pmid,
            "Subject": match[0].strip(),
            "Predicate": match[1].strip(),
            "Object": match[2].strip()
        })
    return triples

# 6️⃣ Process and collect results
structured_results = []

for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Processing by PMID"):
    pmid = row["PMID"]
    full_text = row["CH_TXT"]
    prompt = build_prompt(full_text)
    output = query_claude(prompt)
    if output:
        structured_results += parse_claude_output(pmid, output)
    time.sleep(1.2)

# 7️⃣ Save as structured CSV
pd.DataFrame(structured_results).to_csv("C:/Users/theya/Downloads/parsed_triples_by_pmid.csv", index=False)
print("✅ Done! Extracted triples saved.")


# 8️⃣ Save as formatted tuples with PMID
with open("C:/Users/theya/Downloads/pmid_triples.txt", "w", encoding="utf-8") as f:
    for row in structured_results:
        triple = f'("{row["PMID"]}", "{row["Subject"]}", "{row["Predicate"]}", "{row["Object"]}")'
        f.write(triple + "\n")

print("✅ Formatted triples saved to pmid_triples.txt")
