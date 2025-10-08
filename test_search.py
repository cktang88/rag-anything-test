#!/usr/bin/env python3

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

# Import the functions from the main script
from lightrag_lotr import (
    extract_text_from_html, 
    parse_and_chunk_fountain_script, 
    load_or_create_embeddings,
    search_similar_chunks
)

def test_search():
    """Test the improved search function with 'you have my sword' query"""
    
    # Load existing data
    WORKING_DIR = "./rag_storage"
    embeddings_file = os.path.join(WORKING_DIR, "chunk_embeddings.json")
    
    if not os.path.exists(embeddings_file):
        print("No embeddings found. Please run the main script first to create embeddings.")
        return
    
    print("Loading existing embeddings...")
    with open(embeddings_file, "r") as f:
        data = json.load(f)
        chunks = data["chunks"]
        embeddings = np.array(data["embeddings"])
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Test the search
    query = "you have my sword"
    print(f"\nTesting search for: '{query}'")
    print("=" * 50)
    
    results = search_similar_chunks(query, chunks, embeddings, top_k=5)
    
    print("\nTop 5 most similar chunks:")
    print("=" * 50)
    
    for result in results:
        chunk = result["chunk"]
        print(f"\nRank {result['rank']} (Similarity: {result['similarity']:.4f})")
        print("-" * 30)
        if chunk.get("scene_heading"):
            print(f"Scene: {chunk['scene_heading']}")
        if chunk.get("character"):
            print(f"Character: {chunk['character']}")
        print(f"Content:\n{chunk['full_chunk']}")
        print()

if __name__ == "__main__":
    test_search()
