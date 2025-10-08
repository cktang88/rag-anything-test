#!/usr/bin/env python3
"""
Test character name property
"""

import sys
sys.path.insert(0, '/Users/kwuang/Documents/rag-anything-test/fountain-tools/python/src')
from fountain_tools.parser import Parser

def test_character_name():
    """Test character name property"""
    
    simple_text = """
INT. ROOM - DAY

BILBO
Hello there!

GANDALF
What do you mean?

GALADRIEL (V.O.)
The world is changed.
"""
    
    print("Testing character name property:")
    print(simple_text)
    print("-" * 50)
    
    parser = Parser()
    parser.add_text(simple_text)
    script = parser.script
    
    print(f"Total elements: {len(script.elements)}")
    for i, element in enumerate(script.elements):
        if element.type.value == "CHARACTER":
            print(f"  {i+1}. CHARACTER:")
            print(f"      name: '{element.name}'")
            print(f"      extension: '{getattr(element, 'extension', 'None')}'")
            print(f"      text: '{element.text}'")
            print(f"      text_raw: '{element.text_raw}'")
        else:
            print(f"  {i+1}. {element.type.value}: '{element.text}'")

if __name__ == "__main__":
    test_character_name()
