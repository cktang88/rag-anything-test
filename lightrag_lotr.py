import os
import sys
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity


# Using fountain_objects.json for accurate element classification


WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize the embedding model
print("Loading SentenceTransformer model...")
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")  # "all-MiniLM-L6-v2"
# BAAI/bge-base-en-v1.5 --> fails miserably...
# Qwen/Qwen3-Embedding-0.6B -> needs time to download
# Alibaba-NLP/gte-multilingual-base - needs time to download, doesn't perform better than minilm
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_html(html_file_path):
    """Extract clean text content from HTML file, focusing on screenplay content"""
    with open(html_file_path, "r", encoding="utf-8", errors="ignore") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Find the screenplay content - look for the <pre> tag that contains the script
    pre_tags = soup.find_all("pre")
    screenplay_text = ""

    for pre in pre_tags:
        pre_text = pre.get_text()
        # Check if this looks like screenplay content
        markers = ["INT.", "EXT.", "BLACK SCREEN", "SUPER:"]
        if any(marker in pre_text.upper() for marker in markers):
            screenplay_text = pre_text
            break

    # If no <pre> tag found, fall back to full text extraction
    if not screenplay_text:
        screenplay_text = soup.get_text()

    # Convert HTML screenplay format to proper Fountain format
    lines = screenplay_text.split("\n")
    fountain_lines = []

    for line in lines:
        # Don't strip the line - preserve original whitespace and newlines
        if not line.strip():
            # Empty line - preserve it
            fountain_lines.append("")
            continue

        # Convert scene headings (INT., EXT., SUPER:)
        markers = ["INT.", "EXT.", "SUPER:"]
        if any(marker in line.upper() for marker in markers):
            # Remove HTML tags but preserve whitespace
            clean_line = line.replace("<b>", "").replace("</b>", "")
            fountain_lines.append(clean_line)

        # Convert character names (they're in <b> tags and should be all caps)
        elif "<b>" in line and "</b>" in line:
            char_name = line.replace("<b>", "").replace("</b>", "")
            # Check if it looks like a character name
            exclude_terms = [
                "IMAGE:",
                "SUPER:",
                "INT.",
                "EXT.",
                "FADE",
                "CUT",
                "DISSOLVE",
            ]
            is_upper = char_name.strip().isupper()
            is_short = len(char_name.strip()) < 50
            no_exclude = not any(x in char_name.upper() for x in exclude_terms)
            no_parens = not char_name.strip().startswith(
                "("
            ) and not char_name.strip().endswith(")")
            if is_upper and is_short and no_exclude and no_parens:
                # This is a character name - format it properly for Fountain
                fountain_lines.append(char_name.strip())
            else:
                # It's probably action or dialogue, clean it up but preserve whitespace
                clean_line = line.replace("<b>", "").replace("</b>", "")
                fountain_lines.append(clean_line)

        # Also check for character names that are indented (not in <b> tags)
        elif line.strip() != line and len(line.strip()) > 0:
            # This is an indented line - check if it's a character name
            stripped = line.strip()
            exclude_terms = [
                "IMAGE:",
                "SUPER:",
                "INT.",
                "EXT.",
                "FADE",
                "CUT",
                "DISSOLVE",
            ]
            # Check if it looks like a character name (all caps, reasonable length)
            is_upper = stripped.isupper()
            is_short = len(stripped) < 50
            no_exclude = not any(x in stripped for x in exclude_terms)
            # Allow character names with extensions like (O.S.), (V.O.), etc.
            has_parens = not stripped.startswith("(") or (
                stripped.startswith("(") and ")" in stripped
            )
            if is_upper and is_short and no_exclude and has_parens:
                # This is a character name - format it properly for Fountain
                fountain_lines.append(stripped)
            else:
                fountain_lines.append(line)

        # Convert parentheticals (lines in parentheses)
        elif line.strip().startswith("(") and line.strip().endswith(")"):
            fountain_lines.append(line)

        # Convert dialogue (lines that are not scene headings, character names, or parentheticals)
        else:
            clean_line = line.replace("<b>", "").replace("</b>", "")
            fountain_lines.append(clean_line)

    # Join lines preserving all original newlines
    text = "\n".join(fountain_lines)
    return text


def parse_and_chunk_fountain_script(fountain_objects_file):
    """
    Parse screenplay using fountain_objects JSON file for accurate element classification
    This creates focused chunks around individual dialogue exchanges with overlap
    """
    # Load fountain objects from JSON file
    if not os.path.exists(fountain_objects_file):
        raise FileNotFoundError(
            f"Fountain objects file not found: {fountain_objects_file}. "
            f"Please run extract_fountain_objects.py first to generate it."
        )

    with open(fountain_objects_file, "r") as f:
        fountain_data = json.load(f)

    elements_data = fountain_data["elements"]
    print(f"Loaded {len(elements_data)} elements from fountain_objects.json")

    # Extract speech segments and create chunks with overlap
    chunks = []
    elements = elements_data

    for i, element in enumerate(elements):
        if element["type"] == "CHARACTER":
            # Found a character - create a chunk around this speech segment
            chunk_parts = []
            current_scene_heading = None

            # Look backwards for scene heading and action context
            lookback_start = max(0, i - 3)  # Look back up to 3 elements
            for j in range(lookback_start, i):
                prev_element = elements[j]
                if prev_element["type"] == "HEADING":
                    current_scene_heading = prev_element["text"].strip()
                elif prev_element["type"] == "ACTION" and current_scene_heading:
                    chunk_parts.append(
                        f"[{prev_element['type']}] {prev_element['text'].strip()}"
                    )

            # Add scene heading if found
            if current_scene_heading:
                chunk_parts.append(f"[HEADING] {current_scene_heading}")

            # Add the character name (use name property for CHARACTER elements)
            char_name = element["name"] if element["name"] else element["text"].strip()
            if element["extension"]:
                char_name += f" ({element['extension']})"
            chunk_parts.append(f"[CHARACTER] {char_name}")

            # Look ahead for dialogue, parentheticals, and action
            j = i + 1
            while j < len(elements) and j < i + 5:  # Look ahead up to 5 elements
                next_element = elements[j]

                if next_element["type"] == "PARENTHETICAL":
                    chunk_parts.append(
                        f"[PARENTHETICAL] ({next_element['text'].strip()})"
                    )
                elif next_element["type"] == "DIALOGUE":
                    chunk_parts.append(f"[DIALOGUE] {next_element['text'].strip()}")
                elif next_element["type"] == "ACTION":
                    chunk_parts.append(f"[ACTION] {next_element['text'].strip()}")
                elif next_element["type"] == "CHARACTER":
                    # Hit another character, stop here
                    break
                elif next_element["type"] == "HEADING":
                    # Hit a new scene, stop here
                    break

                j += 1

            # Look ahead for additional context (action after dialogue)
            if (
                j < len(elements) and j < i + 8
            ):  # Look a bit further for post-dialogue action
                for k in range(j, min(j + 3, len(elements))):
                    if elements[k]["type"] == "ACTION":
                        chunk_parts.append(f"[ACTION] {elements[k]['text'].strip()}")
                    elif elements[k]["type"] in ["CHARACTER", "HEADING"]:
                        break

            # Create the chunk with metadata
            if chunk_parts and len("\n".join(chunk_parts).strip()) > 30:
                chunk_text = "\n".join(chunk_parts).strip()

                # Create chunk metadata
                chunk_metadata = {
                    "scene_heading": current_scene_heading,
                    "character": char_name,
                    "full_chunk": chunk_text,
                }

                chunks.append(chunk_metadata)

    print(f"Created {len(chunks)} character speech chunks")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks as preview
        print(f"Chunk {i+1} preview: {chunk['full_chunk'][:100]}...")

    return chunks


def load_or_create_embeddings(chunks, fountain_objects_file):
    """Load existing embeddings or create new ones"""
    # Use fountain objects filename for embedding file naming
    base_name = os.path.splitext(os.path.basename(fountain_objects_file))[0]
    embeddings_file = os.path.join(WORKING_DIR, f"{base_name}_embeddings.json")

    if os.path.exists(embeddings_file):
        print("Loading existing embeddings...")
        with open(embeddings_file, "r") as f:
            data = json.load(f)
            return data["chunks"], np.array(data["embeddings"])
    else:
        print("Creating new embeddings...")
        print(f"Embedding {len(chunks)} chunks...")

        # Extract dialogue parts for embedding
        dialogue_parts = []
        for chunk in chunks:
            lines = chunk["full_chunk"].split("\n")
            dialogue_lines = [line for line in lines if line.startswith("[DIALOGUE]")]
            dialogue_text = "\n".join(dialogue_lines)
            dialogue_parts.append(dialogue_text)

        # Create embeddings for dialogue parts only
        embeddings = model.encode(dialogue_parts, show_progress_bar=True)

        # Save embeddings for future use
        data = {"chunks": chunks, "embeddings": embeddings.tolist()}
        with open(embeddings_file, "w") as f:
            json.dump(data, f)

        return chunks, embeddings


def max_window_cosine(query_vec, doc_token_vecs, window_size=5):
    """Calculate maximum cosine similarity using sliding window approach"""
    from numpy import dot
    from numpy.linalg import norm

    def cos(a, b): 
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    best = -1.0
    for i in range(0, len(doc_token_vecs) - window_size + 1):
        win = doc_token_vecs[i:i+window_size]
        win_vec = sum(win) / len(win)          # mean within the small window
        best = max(best, cos(query_vec, win_vec))
    return best


def search_similar_chunks(query, chunks, embeddings, top_k=5, window_size=5):
    """Search for similar chunks using window sliding cosine similarity"""
    # Extract dialogue parts from query for embedding
    query_lines = query.split("\n")
    query_dialogue_lines = [
        line for line in query_lines if line.startswith("[DIALOGUE]")
    ]
    query_dialogue = "\n".join(query_dialogue_lines)

    # If no dialogue found in query, use the whole query
    if not query_dialogue.strip():
        query_dialogue = query

    # Encode the query dialogue
    query_embedding = model.encode([query_dialogue])[0]

    # For each chunk, create sentence-level embeddings and use window sliding
    similarities = []
    
    for i, chunk in enumerate(chunks):
        # Split chunk into sentences for window sliding
        chunk_text = chunk["full_chunk"]
        
        # Extract dialogue parts for better matching
        lines = chunk_text.split("\n")
        dialogue_lines = [line for line in lines if line.startswith("[DIALOGUE]")]
        if dialogue_lines:
            dialogue_text = "\n".join(dialogue_lines)
        else:
            dialogue_text = chunk_text
        
        # Split into sentences (simple approach - split on periods, exclamation, question marks)
        import re
        sentences = re.split(r'[.!?]+', dialogue_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            # Fallback to original embedding if no sentences found
            similarities.append(cosine_similarity([query_embedding], [embeddings[i]])[0][0])
            continue
            
        # Create embeddings for each sentence
        sentence_embeddings = model.encode(sentences)
        
        # Use window sliding to find best match
        if len(sentence_embeddings) < window_size:
            # If not enough sentences, use all of them
            window_vec = np.mean(sentence_embeddings, axis=0)
            from numpy import dot
            from numpy.linalg import norm
            similarity = float(dot(query_embedding, window_vec) / (norm(query_embedding) * norm(window_vec) + 1e-9))
        else:
            # Use window sliding
            similarity = max_window_cosine(query_embedding, sentence_embeddings, window_size)
        
        similarities.append(similarity)

    # Get top-k most similar chunks
    similarities = np.array(similarities)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i, idx in enumerate(top_indices):
        results.append(
            {"rank": i + 1, "similarity": similarities[idx], "chunk": chunks[idx]}
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Interactive screenplay quote finder using fountain objects")
    parser.add_argument("fountain_objects_file", help="Path to the fountain objects JSON file")
    
    args = parser.parse_args()
    fountain_objects_file = args.fountain_objects_file
    
    try:
        # Check if file exists
        if not os.path.exists(fountain_objects_file):
            print(f"Error: File '{fountain_objects_file}' not found.")
            return
        
        print(f"Loading fountain objects from {fountain_objects_file}...")

        # Parse screenplay with Fountain format
        print("Parsing screenplay with Fountain format...")
        try:
            scene_chunks = parse_and_chunk_fountain_script(fountain_objects_file)
        except Exception as e:
            print(f"Fountain parsing failed: {e}")
            raise e

        # Load or create embeddings
        chunks, embeddings = load_or_create_embeddings(scene_chunks, fountain_objects_file)

        print("Text successfully processed and embedded!")

        # Interactive quote finder
        print("\n" + "=" * 60)
        # Extract filename for display
        filename = os.path.basename(fountain_objects_file).replace('.json', '').replace('_', ' ').replace('fountain objects', '').strip().title()
        print(f"SCREENPLAY QUOTE FINDER - {filename}")
        print("=" * 60)
        print(f"Enter phrases to find similar quotes from {filename}")
        print("Type 'quit' to exit")
        print("=" * 60)

        while True:
            user_input = input("\nEnter a phrase to find similar quotes: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter a phrase.")
                continue

            print(f"\nSearching for quotes similar to: '{user_input}'")
            print("-" * 50)

            # Search for similar chunks
            results = search_similar_chunks(user_input, chunks, embeddings, top_k=5)

            print("\nTop 5 most similar chunks:")
            print("=" * 50)

            for result in results:
                chunk = result["chunk"]
                print(
                    f"\nRank {result['rank']} (Similarity: {result['similarity']:.4f})"
                )
                print("-" * 30)
                if chunk.get("scene_heading"):
                    print(f"Scene: {chunk['scene_heading']}")
                if chunk.get("character"):
                    print(f"Character: {chunk['character']}")
                print(f"Content:\n{chunk['full_chunk']}")
                print()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
