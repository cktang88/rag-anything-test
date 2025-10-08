#!/usr/bin/env python3
"""
Test the improved HTML to Fountain conversion
"""

import sys
import os
sys.path.insert(0, '/Users/kwuang/Documents/rag-anything-test/fountain-tools/python/src')

from fountain_tools.parser import Parser
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
    lines = screenplay_text.split('\n')
    fountain_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            fountain_lines.append('')
            continue
            
        # Convert scene headings (INT., EXT., SUPER:)
        if any(marker in line.upper() for marker in ['INT.', 'EXT.', 'SUPER:']):
            # Remove HTML tags and clean up
            clean_line = line.replace('<b>', '').replace('</b>', '').strip()
            # Ensure proper scene heading format
            if clean_line.startswith('INT.') or clean_line.startswith('EXT.'):
                fountain_lines.append(clean_line)
            elif clean_line.startswith('SUPER:'):
                fountain_lines.append(clean_line)
            else:
                fountain_lines.append(clean_line)
                
        # Convert character names (they're in <b> tags and should be all caps)
        elif '<b>' in line and '</b>' in line:
            char_name = line.replace('<b>', '').replace('</b>', '').strip()
            # Check if it looks like a character name
            if (char_name.isupper() and len(char_name) < 50 and 
                not any(x in char_name for x in ['IMAGE:', 'SUPER:', 'INT.', 'EXT.', 'FADE', 'CUT', 'DISSOLVE']) and
                not char_name.startswith('(') and not char_name.endswith(')')):
                # This is a character name - format it properly for Fountain
                fountain_lines.append(char_name)
            else:
                # It's probably action or dialogue, clean it up
                clean_line = line.replace('<b>', '').replace('</b>', '').strip()
                if clean_line:
                    fountain_lines.append(clean_line)
                    
        # Convert parentheticals (lines in parentheses)
        elif line.startswith('(') and line.endswith(')'):
            fountain_lines.append(line)
            
        # Convert dialogue (lines that are not scene headings, character names, or parentheticals)
        else:
            clean_line = line.replace('<b>', '').replace('</b>', '').strip()
            if clean_line:
                # Check if this looks like action (not dialogue)
                if (any(marker in clean_line.upper() for marker in ['IMAGE:', 'FADE', 'CUT', 'DISSOLVE', 'WIDE ON:', 'CLOSE ON:', 'SUPER:']) or
                    clean_line.isupper() and len(clean_line) > 20):
                    # This is action
                    fountain_lines.append(clean_line)
                else:
                    # This is likely dialogue
                    fountain_lines.append(clean_line)
    
    # Join lines and clean up
    text = '\n'.join(fountain_lines)
    
    # Clean up multiple newlines
    import re
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text

def test_fountain_parsing(text):
    """Test Fountain parsing on the converted text"""
    parser = Parser()
    parser.add_text(text)
    script = parser.script
    
    print(f"Total elements found: {len(script.elements)}")
    print("\nElement types found:")
    element_types = {}
    for element in script.elements:
        element_type = element.type.value
        element_types[element_type] = element_types.get(element_type, 0) + 1
    
    for elem_type, count in element_types.items():
        print(f"  {elem_type}: {count}")
    
    # Look for character elements specifically
    character_elements = [e for e in script.elements if e.type.value == "CHARACTER"]
    print(f"\nCharacter elements found: {len(character_elements)}")
    for i, char in enumerate(character_elements[:10]):
        print(f"  {i+1}. CHARACTER: '{char.text}'")
    
    # Show some dialogue examples
    dialogue_elements = [e for e in script.elements if e.type.value == "DIALOGUE"]
    print(f"\nDialogue elements found: {len(dialogue_elements)}")
    for i, dialogue in enumerate(dialogue_elements[:5]):
        print(f"  {i+1}. DIALOGUE: '{dialogue.text[:50]}...'")

if __name__ == "__main__":
    print("Testing improved HTML to Fountain conversion...")
    print("=" * 60)
    
    # Extract and convert text
    html_file = "./input_files/Lord-of-the-Rings-Fellowship-of-the-Ring,-The.html"
    print(f"Processing: {html_file}")
    
    text_content = extract_text_from_html(html_file)
    
    # Save the converted text
    with open('./lotr_fountain_format.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    print(f"Converted {len(text_content)} characters")
    print(f"Saved to: ./lotr_fountain_format.txt")
    print(f"First 1000 characters:")
    print("-" * 50)
    print(text_content[:1000])
    print("-" * 50)
    
    # Test Fountain parsing
    print("\nTesting Fountain parsing...")
    test_fountain_parsing(text_content)
