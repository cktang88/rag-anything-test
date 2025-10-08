import os
import sys
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from bs4 import BeautifulSoup

# Add fountain-tools to Python path
sys.path.insert(0, '/Users/kwuang/Documents/rag-anything-test/fountain-tools/python/src')
from fountain_tools.parser import Parser
from fountain_tools.writer import Writer

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

def extract_text_from_html(html_file_path):
    """Extract clean text content from HTML file, focusing on screenplay content"""
    with open(html_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Find the screenplay content - look for the <pre> tag that contains the script
    pre_tags = soup.find_all('pre')
    screenplay_text = ""
    
    for pre in pre_tags:
        pre_text = pre.get_text()
        # Check if this looks like screenplay content
        if any(marker in pre_text.upper() for marker in ['INT.', 'EXT.', 'BLACK SCREEN', 'SUPER:']):
            screenplay_text = pre_text
            break
    
    # If no <pre> tag found, fall back to full text extraction
    if not screenplay_text:
        screenplay_text = soup.get_text()
    
    # Convert HTML screenplay format to proper Fountain format
    # Preserve all newlines and whitespace structure
    lines = screenplay_text.split('\n')
    fountain_lines = []
    
    for line in lines:
        # Don't strip the line - preserve original whitespace and newlines
        if not line.strip():
            # Empty line - preserve it
            fountain_lines.append('')
            continue
            
        # Convert scene headings (INT., EXT., SUPER:)
        if any(marker in line.upper() for marker in ['INT.', 'EXT.', 'SUPER:']):
            # Remove HTML tags but preserve whitespace
            clean_line = line.replace('<b>', '').replace('</b>', '')
            fountain_lines.append(clean_line)
                
        # Convert character names (they're in <b> tags and should be all caps)
        elif '<b>' in line and '</b>' in line:
            char_name = line.replace('<b>', '').replace('</b>', '')
            # Check if it looks like a character name
            if (char_name.strip().isupper() and len(char_name.strip()) < 50 and 
                not any(x in char_name.upper() for x in ['IMAGE:', 'SUPER:', 'INT.', 'EXT.', 'FADE', 'CUT', 'DISSOLVE']) and
                not char_name.strip().startswith('(') and not char_name.strip().endswith(')')):
                # This is a character name - format it properly for Fountain (no indentation)
                fountain_lines.append(char_name.strip())
            else:
                # It's probably action or dialogue, clean it up but preserve whitespace
                clean_line = line.replace('<b>', '').replace('</b>', '')
                fountain_lines.append(clean_line)
        
        # Also check for character names that are indented (not in <b> tags)
        elif line.strip() != line and len(line.strip()) > 0:
            # This is an indented line - check if it's a character name
            stripped = line.strip()
            # Check if it looks like a character name (all caps, reasonable length, may have extensions like (O.S.))
            if (stripped.isupper() and len(stripped) < 50 and 
                not any(x in stripped for x in ['IMAGE:', 'SUPER:', 'INT.', 'EXT.', 'FADE', 'CUT', 'DISSOLVE']) and
                # Allow character names with extensions like (O.S.), (V.O.), etc.
                (not stripped.startswith('(') or stripped.startswith('(') and ')' in stripped)):
                # This is a character name - format it properly for Fountain (no indentation)
                fountain_lines.append(stripped)
            else:
                fountain_lines.append(line)
                    
        # Convert parentheticals (lines in parentheses)
        elif line.strip().startswith('(') and line.strip().endswith(')'):
            fountain_lines.append(line)
            
        # Convert dialogue (lines that are not scene headings, character names, or parentheticals)
        else:
            clean_line = line.replace('<b>', '').replace('</b>', '')
            fountain_lines.append(clean_line)
    
    # Join lines preserving all original newlines
    text = '\n'.join(fountain_lines)
    
    # Don't clean up multiple newlines - preserve the original structure
    
    return text

def parse_and_chunk_fountain_script(text):
    """
    Parse screenplay using fountain-tools library and chunk by character speech segments
    This creates focused chunks around individual dialogue exchanges with overlap
    """
    # Initialize parser and writer
    parser = Parser()
    writer = Writer()
    
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
                    chunk_parts.append(f"[{prev_element.type.value}] {prev_element.text.strip()}")
            
            # Add scene heading if found
            if current_scene_heading:
                chunk_parts.append(f"[HEADING] {current_scene_heading}")
            
            # Add the character name (use name property for CHARACTER elements)
            char_name = element.name if hasattr(element, 'name') else element.text.strip()
            if hasattr(element, 'extension') and element.extension:
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
            if j < len(elements) and j < i + 8:  # Look a bit further for post-dialogue action
                for k in range(j, min(j + 3, len(elements))):
                    if elements[k].type.value == "ACTION":
                        chunk_parts.append(f"[ACTION] {elements[k].text.strip()}")
                    elif elements[k].type.value in ["CHARACTER", "HEADING"]:
                        break
            
            # Create the chunk
            if chunk_parts and len('\n'.join(chunk_parts).strip()) > 30:
                chunk_text = '\n'.join(chunk_parts).strip()
                chunks.append(chunk_text)
    
    print(f"Created {len(chunks)} character speech chunks")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks as preview
        print(f"Chunk {i+1} preview: {chunk[:100]}...")
    
    return chunks

def search_text_directly(text, search_phrase):
    """Search for a phrase directly in the text and return context"""
    search_phrase_lower = search_phrase.lower()
    text_lower = text.lower()
    
    # Find the position of the phrase
    pos = text_lower.find(search_phrase_lower)
    if pos == -1:
        return None
    
    # Get context around the phrase (200 characters before and after)
    start = max(0, pos - 200)
    end = min(len(text), pos + len(search_phrase) + 200)
    context = text[start:end]
    
    return context

async def gpt_5_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    """Wrapper function to use GPT-5 model"""
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        from lightrag.llm.openai import GPTKeywordExtractionFormat
        kwargs["response_format"] = GPTKeywordExtractionFormat
    
    # Import the internal function
    from lightrag.llm.openai import openai_complete_if_cache
    
    return await openai_complete_if_cache(
        "gpt-5-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_5_mini_complete,
        # Custom chunking disabled - we're doing scene-based chunking manually
        # chunk_token_size=2000,  # Not needed with custom chunking
        # chunk_overlap=200,      # Not needed with custom chunking
        # max_chunk_num=50,       # Not needed with custom chunking
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag

async def main():
    rag = None
    try:
        # Clear existing storage to start fresh with new chunk sizes
        import shutil
        if os.path.exists(WORKING_DIR):
            print(f"Clearing existing storage at {WORKING_DIR}...")
            shutil.rmtree(WORKING_DIR)
            os.mkdir(WORKING_DIR)
        
        # Initialize RAG instance
        print("Initializing LightRAG...")
        rag = await initialize_rag()
        
        # Extract text from the Lord of the Rings HTML file
        html_file = "./input_files/Lord-of-the-Rings-Fellowship-of-the-Ring,-The.html"
        print(f"Extracting text from {html_file}...")
        text_content = extract_text_from_html(html_file)
        
        print(f"Extracted {len(text_content)} characters of text")
        
        # Try Fountain parsing first, fallback to scene markers
        print("Parsing screenplay with Fountain format...")
        try:
            scene_chunks = parse_and_chunk_fountain_script(text_content)
        except Exception as e:
            print(f"Fountain parsing failed: {e}")
            raise e
        
        print(f"Inserting {len(scene_chunks)} scene-based chunks into RAG...")
        
        # Insert each chunk separately to preserve scene boundaries
        for i, chunk in enumerate(scene_chunks):
            print(f"Inserting chunk {i+1}/{len(scene_chunks)}...")
            await rag.ainsert(chunk)
        
        # Store the text content for direct search fallback
        global original_text
        original_text = text_content
        
        print("Text successfully inserted into RAG!")
        
        # Try different search modes
        search_modes = ["hybrid", "local", "global"]
        # Interactive quote finder
        print("\n" + "="*60)
        print("LOTR QUOTE FINDER")
        print("="*60)
        print("Enter phrases to find similar quotes from The Fellowship of the Ring")
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            user_input = input("\nEnter a phrase to find similar quotes: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                print("Please enter a phrase.")
                continue
            
            print(f"\nSearching for quotes similar to: '{user_input}'")
            print("-" * 50)
            
            # Create a general search query for any input
            query = f"""Search through The Fellowship of the Ring for quotes that have the most similar sentence structure and phrasing pattern as: "{user_input}"

Look for quotes that:
- Use the same grammatical structure (subject + verb + object + prepositional phrase)
- Have similar word patterns and rhythm
- Match the emotional tone and delivery style
- Are actual dialogue from characters in the book

Please provide:
1. The exact quote from the book (word-for-word)
2. Who said it (character name)
3. Brief context of when/where it was said
4. Why this quote matches the input structure

Be very specific about finding quotes with identical sentence patterns, not just similar themes or characters."""
            
            # Try different search modes to find the best results
            best_result = None
            for search_mode in search_modes:
                try:
                    print(f"Trying {search_mode} search...")
                    result = await rag.aquery(
                        query,
                        param=QueryParam(
                            mode=search_mode,
                            top_k=30,  # Increase to get more chunks
                            enable_rerank=False
                        )
                    )
                    
                    # Check if this result is better (contains actual quotes)
                    if "quote" in result.lower() and "said" in result.lower():
                        best_result = result
                        print(f"Found good result with {search_mode} search!")
                        break
                    elif not best_result:
                        best_result = result
                        
                except Exception as e:
                    print(f"Error with {search_mode} search: {e}")
                    continue
            
            if best_result:
                print(f"\nBest result found:")
                print("-" * 50)
                print(best_result)
            else:
                print("No suitable quotes found. Try rephrasing your input.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            print("\nFinalizing storages...")
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
