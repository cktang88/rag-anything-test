#!/usr/bin/env python3
"""
Test simple Fountain format to understand what the parser expects
"""

import sys
sys.path.insert(0, '/Users/kwuang/Documents/rag-anything-test/fountain-tools/python/src')
from fountain_tools.parser import Parser

def test_simple_fountain():
    """Test with a simple Fountain format example"""
    
    # Simple Fountain format test
    simple_text = """
INT. ROOM - DAY

BILBO
Hello there!

GANDALF
What do you mean?

BILBO
I mean exactly what I said.
"""
    
    print("Testing simple Fountain format:")
    print(simple_text)
    print("-" * 50)
    
    parser = Parser()
    parser.add_text(simple_text)
    script = parser.script
    
    print(f"Total elements: {len(script.elements)}")
    for i, element in enumerate(script.elements):
        print(f"  {i+1}. {element.type.value}: '{element.text}'")
    
    # Test with character extensions
    extended_text = """
INT. ROOM - DAY

GALADRIEL (V.O.)
The world is changed.

BILBO
What's this?
"""
    
    print("\nTesting with character extensions:")
    print(extended_text)
    print("-" * 50)
    
    parser2 = Parser()
    parser2.add_text(extended_text)
    script2 = parser2.script
    
    print(f"Total elements: {len(script2.elements)}")
    for i, element in enumerate(script2.elements):
        print(f"  {i+1}. {element.type.value}: '{element.text}'")

if __name__ == "__main__":
    test_simple_fountain()
