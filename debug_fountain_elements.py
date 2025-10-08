#!/usr/bin/env python3
"""
Debug Fountain elements to understand the text property issue
"""

import sys
sys.path.insert(0, '/Users/kwuang/Documents/rag-anything-test/fountain-tools/python/src')
from fountain_tools.parser import Parser

def debug_fountain_elements():
    """Debug Fountain elements to understand the text property issue"""
    
    simple_text = """
INT. ROOM - DAY

BILBO
Hello there!

GANDALF
What do you mean?
"""
    
    print("Testing Fountain element properties:")
    print(simple_text)
    print("-" * 50)
    
    parser = Parser()
    parser.add_text(simple_text)
    script = parser.script
    
    print(f"Total elements: {len(script.elements)}")
    for i, element in enumerate(script.elements):
        print(f"  {i+1}. {element.type.value}:")
        print(f"      text: '{element.text}'")
        print(f"      text_raw: '{element.text_raw}'")
        print(f"      _text: '{element._text}'")
        print(f"      dir: {[attr for attr in dir(element) if not attr.startswith('_')]}")
        print()

if __name__ == "__main__":
    debug_fountain_elements()
