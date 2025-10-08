#!/usr/bin/env python3
"""
Output the first 500 chunks to separate text files for examination
"""

import sys
import os
sys.path.insert(0, '/Users/kwuang/Documents/rag-anything-test/fountain-tools/python/src')

from fountain_tools.parser import Parser
from fountain_tools.writer import Writer
from bs4 import BeautifulSoup

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
    
    return chunks

def main():
    print("Extracting and chunking LOTR screenplay...")
    
    # Extract text from HTML
    html_file = "./input_files/Lord-of-the-Rings-Fellowship-of-the-Ring,-The.html"
    print(f"Processing: {html_file}")
    
    text_content = extract_text_from_html(html_file)
    print(f"Extracted {len(text_content)} characters")
    
    # Parse and chunk
    print("Parsing with Fountain and creating character speech chunks...")
    chunks = parse_and_chunk_fountain_script(text_content)
    print(f"Created {len(chunks)} character speech chunks")
    
    # Create output directory
    output_dir = "./chunks_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Output ALL chunks to separate files
    num_chunks = len(chunks)
    print(f"Outputting all {num_chunks} chunks to {output_dir}/")
    
    for i in range(num_chunks):
        chunk_file = os.path.join(output_dir, f"chunk_{i+1:03d}.txt")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(f"=== CHUNK {i+1} ===\n\n")
            f.write(chunks[i])
            f.write(f"\n\n=== END CHUNK {i+1} ===")
    
    print(f"Successfully created {num_chunks} chunk files in {output_dir}/")
    
    # Show preview of first few chunks
    print("\nPreview of first 3 chunks:")
    print("=" * 60)
    for i in range(min(3, len(chunks))):
        print(f"\n--- Chunk {i+1} ---")
        print(chunks[i][:200] + "..." if len(chunks[i]) > 200 else chunks[i])
        print("-" * 40)

if __name__ == "__main__":
    main()
