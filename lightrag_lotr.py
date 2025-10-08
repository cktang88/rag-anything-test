import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity


# Simple screenplay element parser
class ScreenplayElement:
    def __init__(self, element_type, text, name=None, extension=None):
        self.type = type("Type", (), {"value": element_type})()
        self.text = text
        self.name = name
        self.extension = extension


class SimpleParser:
    def __init__(self):
        self.script = None

    def add_text(self, text):
        self.script = self.parse_screenplay(text)

    def parse_screenplay(self, text):
        lines = text.split("\n")
        elements = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Scene headings (INT., EXT., etc.)
            if any(marker in line.upper() for marker in ["INT.", "EXT.", "SUPER:"]):
                elements.append(ScreenplayElement("HEADING", line))

            # Character names (all caps, not too long, not scene elements)
            elif (
                line.isupper()
                and len(line) < 50
                and not any(
                    x in line
                    for x in [
                        "IMAGE:",
                        "SUPER:",
                        "INT.",
                        "EXT.",
                        "FADE",
                        "CUT",
                        "DISSOLVE",
                    ]
                )
                and not line.startswith("(")
            ):
                # Check for character extensions like (O.S.), (V.O.)
                if "(" in line and ")" in line:
                    name, extension = line.split("(", 1)
                    name = name.strip()
                    extension = extension.rstrip(")")
                    elements.append(
                        ScreenplayElement("CHARACTER", line, name, extension)
                    )
                else:
                    elements.append(ScreenplayElement("CHARACTER", line, line))

            # Parentheticals (lines in parentheses)
            elif line.startswith("(") and line.endswith(")"):
                elements.append(ScreenplayElement("PARENTHETICAL", line))

            # Dialogue (everything else that's not empty)
            else:
                elements.append(ScreenplayElement("DIALOGUE", line))

        # Create a simple script object
        script = type("Script", (), {})()
        script.elements = elements
        return script


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


def parse_and_chunk_fountain_script(text):
    """
    Parse screenplay using fountain-tools library and chunk by character speech segments
    This creates focused chunks around individual dialogue exchanges with overlap
    """
    # Initialize parser
    parser = SimpleParser()

    # Parse the script
    parser.add_text(text)
    script = parser.script

    # Extract speech segments and create chunks with overlap
    chunks = []
    elements = list(script.elements)

    for i, element in enumerate(elements):
        if element.type.value == "CHARACTER":
            # Found a character - create a chunk around this speech segment
            chunk_parts = []
            current_scene_heading = None

            # Look backwards for scene heading and action context
            lookback_start = max(0, i - 3)  # Look back up to 3 elements
            for j in range(lookback_start, i):
                prev_element = elements[j]
                if prev_element.type.value == "HEADING":
                    current_scene_heading = prev_element.text.strip()
                elif prev_element.type.value == "ACTION" and current_scene_heading:
                    chunk_parts.append(
                        f"[{prev_element.type.value}] {prev_element.text.strip()}"
                    )

            # Add scene heading if found
            if current_scene_heading:
                chunk_parts.append(f"[HEADING] {current_scene_heading}")

            # Add the character name (use name property for CHARACTER elements)
            char_name = (
                element.name if hasattr(element, "name") else element.text.strip()
            )
            if hasattr(element, "extension") and element.extension:
                char_name += f" ({element.extension})"
            chunk_parts.append(f"[CHARACTER] {char_name}")

            # Look ahead for dialogue, parentheticals, and action
            j = i + 1
            while j < len(elements) and j < i + 5:  # Look ahead up to 5 elements
                next_element = elements[j]

                if next_element.type.value == "PARENTHETICAL":
                    chunk_parts.append(f"[PARENTHETICAL] ({next_element.text.strip()})")
                elif next_element.type.value == "DIALOGUE":
                    chunk_parts.append(f"[DIALOGUE] {next_element.text.strip()}")
                elif next_element.type.value == "ACTION":
                    chunk_parts.append(f"[ACTION] {next_element.text.strip()}")
                elif next_element.type.value == "CHARACTER":
                    # Hit another character, stop here
                    break
                elif next_element.type.value == "HEADING":
                    # Hit a new scene, stop here
                    break

                j += 1

            # Look ahead for additional context (action after dialogue)
            if (
                j < len(elements) and j < i + 8
            ):  # Look a bit further for post-dialogue action
                for k in range(j, min(j + 3, len(elements))):
                    if elements[k].type.value == "ACTION":
                        chunk_parts.append(f"[ACTION] {elements[k].text.strip()}")
                    elif elements[k].type.value in ["CHARACTER", "HEADING"]:
                        break

            # Create the chunk
            if chunk_parts and len("\n".join(chunk_parts).strip()) > 30:
                chunk_text = "\n".join(chunk_parts).strip()
                chunks.append(chunk_text)

    print(f"Created {len(chunks)} character speech chunks")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks as preview
        print(f"Chunk {i+1} preview: {chunk[:100]}...")

    return chunks


def load_or_create_embeddings(chunks):
    """Load existing embeddings or create new ones"""
    embeddings_file = os.path.join(WORKING_DIR, "chunk_embeddings.json")

    if os.path.exists(embeddings_file):
        print("Loading existing embeddings...")
        with open(embeddings_file, "r") as f:
            data = json.load(f)
            return data["chunks"], np.array(data["embeddings"])
    else:
        print("Creating new embeddings...")
        print(f"Embedding {len(chunks)} chunks...")

        # Create embeddings for all chunks
        embeddings = model.encode(chunks, show_progress_bar=True)

        # Save embeddings for future use
        data = {"chunks": chunks, "embeddings": embeddings.tolist()}
        with open(embeddings_file, "w") as f:
            json.dump(data, f)

        return chunks, embeddings


def search_similar_chunks(query, chunks, embeddings, top_k=5):
    """Search for similar chunks using cosine similarity"""
    # Encode the query
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top-k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i, idx in enumerate(top_indices):
        results.append(
            {"rank": i + 1, "similarity": similarities[idx], "chunk": chunks[idx]}
        )

    return results


def main():
    try:
        # Extract text from the Lord of the Rings HTML file
        html_file = "./input_files/Lord-of-the-Rings-Fellowship-of-the-Ring,-The.html"
        print(f"Extracting text from {html_file}...")
        text_content = extract_text_from_html(html_file)

        print(f"Extracted {len(text_content)} characters of text")

        # Parse screenplay with Fountain format
        print("Parsing screenplay with Fountain format...")
        try:
            scene_chunks = parse_and_chunk_fountain_script(text_content)
        except Exception as e:
            print(f"Fountain parsing failed: {e}")
            raise e

        # Load or create embeddings
        chunks, embeddings = load_or_create_embeddings(scene_chunks)

        print("Text successfully processed and embedded!")

        # Interactive quote finder
        print("\n" + "=" * 60)
        print("LOTR QUOTE FINDER")
        print("=" * 60)
        print("Enter phrases to find similar quotes from The Fellowship of the Ring")
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
                print(
                    f"\nRank {result['rank']} (Similarity: {result['similarity']:.4f})"
                )
                print("-" * 30)
                print(result["chunk"])
                print()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
