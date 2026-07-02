"""
Create embeddings for the curated Citizens Advice legal concepts.
Reads ca_terms_all.csv and outputs ca_terms_final.pkl with embeddings.

Usage:
    python create_ca_embeddings.py
"""

import pickle
import csv
from sentence_transformers import SentenceTransformer

INPUT_FILE = "ca_terms_all.csv"
OUTPUT_FILE = "ca_terms_all.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def parse_csv_file(filepath):
    """Parse the CSV file into structured data."""
    terms = []
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        
        for row in reader:
            # Skip rows with empty title or definition
            if not row.get('Title') or not row.get('Definition'):
                continue
            
            terms.append({
                'term': row.get('Title', '').strip(),
                'definition': row.get('Definition', '').strip(),
                'source': row.get('Source', '').strip()
            })
    
    return terms

def create_embeddings(terms, model_name):
    """Generate embeddings for each term's definition."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(terms)} terms...")
    
    for i, term in enumerate(terms, 1):
        if i % 10 == 0:
            print(f"  {i}/{len(terms)}...")
        
        # Embed the definition (plain text, no context)
        embedding = model.encode(term['definition'], show_progress_bar=False)
        term['embedding'] = embedding.tolist()
    
    print("✓ Embeddings complete")
    return terms

def main():
    print("="*80)
    print("Creating embeddings for Citizens Advice concepts (all areas)")
    print("="*80 + "\n")
    
    # Parse CSV
    print(f"Reading {INPUT_FILE}...")
    terms = parse_csv_file(INPUT_FILE)
    print(f"Loaded {len(terms)} terms\n")
    
    # Create embeddings
    terms_with_embeddings = create_embeddings(terms, MODEL_NAME)
    
    # Save to pickle
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(terms_with_embeddings, f)
    
    print(f"✓ Saved {len(terms_with_embeddings)} terms with embeddings\n")
    
    
    print("\n" + "="*80)
    print(f"✓ Complete. Use {OUTPUT_FILE} for tagging.")
    print("="*80)

if __name__ == "__main__":
    main()