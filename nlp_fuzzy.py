import spacy
from typing import List, Dict, Any
from rapidfuzz import process, fuzz
from symptom_db import plant_disease_db


# ---------------------------------------------------------
# NLP SETUP
# ---------------------------------------------------------
try:
    # Load the small English model for NLP tasks
    nlp = spacy.load("en_core_web_sm") 
except OSError:
    print("Warning: SpaCy model 'en_core_web_sm' not found. NLP functionality will be limited.")
    # Fallback dummy class if SpaCy model is missing
    class DummyDoc:
        def __init__(self):
            self.noun_chunks = []
            self.tokens = []
        def __iter__(self):
            return iter([])
    nlp = lambda text: DummyDoc()

# ---------------------------------------------------------
# TEXT PROCESSING UTILITIES
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """Cleans user input for better matching."""
    return text.lower().strip()

def candidate_phrases(text: str) -> List[str]:
    """
    Extracts key phrases, nouns, and adjectives from the user's text 
    to compare against the disease database.
    """
    doc = nlp(text)
    cands = set()

    # 1. Include the full sentence
    cands.add(text)

    # 2. Include Noun Chunks (e.g., "brown spots", "yellow leaves")
    if hasattr(doc, 'noun_chunks'):
        for chunk in doc.noun_chunks:
            cands.add(chunk.text.strip())

    # 3. Include individual keywords (Lemmatized nouns and adjectives)
    # This loop is safer for both real SpaCy and the dummy fallback
    try:
        for token in doc:
            if token.pos_ in ("NOUN", "ADJ", "PROPN"):
                cands.add(token.lemma_.strip())
            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                pair = f"{token.text} {token.head.text}"
                cands.add(pair.strip())
    except AttributeError:
        pass

    # Filter out very short strings or pure numbers
    return [c for c in cands if len(c) >= 2 and not c.isnumeric()]

# ---------------------------------------------------------
# CORE FUZZY MATCHING LOGIC
# ---------------------------------------------------------

def fuzzy_match_one(candidate: str, symptom_keys: List[str], limit: int = 3):
    if not symptom_keys:
        return []
    return process.extract(candidate, symptom_keys, scorer=fuzz.token_sort_ratio, limit=limit)


def extract_symptoms(user_text: str, plant: str, score_threshold=60):
    text = normalize_text(user_text)

    # plant_disease_db["Tomato"] → dict
    plant_records = plant_disease_db.get(plant, {})
    if not plant_records:
        return []

    symptom_keys = list(plant_records.keys())

    matches = []
    for cand in candidate_phrases(text):
        res = fuzzy_match_one(cand, symptom_keys)
        for key, score, _ in res:
            if score >= score_threshold:
                record = plant_records[key]
                matches.append({
                    "symptom_key": key,
                    "score": score,
                    "matched_on": cand,
                    "cause": record["cause"],
                    "treatment": record["treatment"]
                })

    # keep best match per disease
    best = {}
    for m in matches:
        k = m["symptom_key"]
        if k not in best or m["score"] > best[k]["score"]:
            best[k] = m

    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    print("NLP Database-Ready Component Loaded.")