import spacy

nlp = spacy.load("en_core_web_md")

with open("data/jci_standards.txt", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

# Add entity ruler
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Process document first
doc = nlp(text)

# Extract clauses using regex on the raw text
import re
clause_pattern = r'([A-Z]{3}\.\d+(?:\.\d+)*)\s+([^\n]+)'
matches = re.findall(clause_pattern, text)

# Combine code and text
clauses = [f"{code} {clause_text.strip()}" for code, clause_text in matches]

# Save to file
with open("extracted_clauses.txt", "w", encoding="utf-8") as f:
    for clause in clauses:
        f.write(clause + "\n")

# Print count
print(f"Total clauses extracted: {len(clauses)}")
print(f"Clauses saved to: extracted_clauses.txt")