#!/usr/bin/env python3
"""
Fix the Fountain format to ensure proper character name recognition
"""

def fix_fountain_format(text):
    """Fix the Fountain format to ensure proper character name recognition"""
    lines = text.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this looks like a character name (all caps, possibly with extensions)
        if (line.isupper() and len(line) > 2 and len(line) < 50 and 
            not any(x in line for x in ['IMAGE:', 'SUPER:', 'INT.', 'EXT.', 'FADE', 'CUT', 'DISSOLVE', 'WIDE ON:', 'CLOSE ON:']) and
            not line.startswith('(') and not line.endswith(')') and
            ('(' in line or line in ['GALADRIEL', 'BILBO', 'FRODO', 'GANDALF', 'SAM', 'ARAGORN', 'LEGOLAS', 'GIMLI', 'BOROMIR', 'GOLLUM', 'ELROND', 'ARWEN', 'ELENDIL', 'ISILDUR', 'SAURON'])):
            
            # This is a character name - add it
            fixed_lines.append(line)
            
            # Look ahead for dialogue or parentheticals
            j = i + 1
            while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                    
                # Check for parenthetical
                if next_line.startswith('(') and next_line.endswith(')'):
                    fixed_lines.append(next_line)
                    j += 1
                    continue
                    
                # Check for dialogue (not action, not character name)
                if (not next_line.isupper() or 
                    any(x in next_line for x in ['IMAGE:', 'FADE', 'CUT', 'DISSOLVE', 'WIDE ON:', 'CLOSE ON:', 'SUPER:']) or
                    next_line.startswith('"') or
                    (len(next_line) > 20 and not next_line.isupper())):
                    # This is dialogue
                    fixed_lines.append(next_line)
                    j += 1
                else:
                    # Hit another character or action, stop
                    break
            
            i = j
        else:
            # Regular line
            if line:
                fixed_lines.append(line)
            else:
                fixed_lines.append('')
            i += 1
    
    return '\n'.join(fixed_lines)

if __name__ == "__main__":
    print("Fixing Fountain format...")
    
    # Read the current format
    with open('./lotr_fountain_format.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Fix the format
    fixed_text = fix_fountain_format(text)
    
    # Save the fixed format
    with open('./lotr_fountain_fixed.txt', 'w', encoding='utf-8') as f:
        f.write(fixed_text)
    
    print(f"Fixed format saved to: ./lotr_fountain_fixed.txt")
    print(f"First 1000 characters:")
    print("-" * 50)
    print(fixed_text[:1000])
    print("-" * 50)
    
    # Test with Fountain parser
    import sys
    sys.path.insert(0, '/Users/kwuang/Documents/rag-anything-test/fountain-tools/python/src')
    from fountain_tools.parser import Parser
    
    parser = Parser()
    parser.add_text(fixed_text)
    script = parser.script
    
    print(f"\nFountain parsing results:")
    print(f"Total elements: {len(script.elements)}")
    
    character_elements = [e for e in script.elements if e.type.value == "CHARACTER"]
    print(f"Character elements: {len(character_elements)}")
    for i, char in enumerate(character_elements[:10]):
        print(f"  {i+1}. CHARACTER: '{char.text}'")
