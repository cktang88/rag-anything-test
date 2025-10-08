import os
import sys
import json
import asyncio
from lightrag_lotr import extract_text_from_html

# Add fountain-tools to Python path
sys.path.insert(0, "/Users/apache/github/rag-anything-test/fountain-tools/python/src")


async def main():
    html_file = "./input_files/Lord-of-the-Rings-Fellowship-of-the-Ring,-The.html"
    output_file = "./fountain_objects.json"

    print("Extracting Fountain objects from LOTR screenplay...")
    print(f"Processing: {html_file}")

    # Extract text from HTML
    text_content = extract_text_from_html(html_file)
    print(f"Extracted {len(text_content)} characters")

    # Parse with Fountain
    from fountain_tools.parser import Parser

    parser = Parser()
    parser.add_text(text_content)
    script = parser.script

    print(f"Parsed script with {len(script.elements)} elements")

    # Convert Fountain objects to JSON-serializable format
    fountain_data = {
        "script_info": {
            "title": getattr(script, "title", None),
            "credit": getattr(script, "credit", None),
            "author": getattr(script, "author", None),
            "draft_date": getattr(script, "draft_date", None),
            "contact": getattr(script, "contact", None),
            "notes": getattr(script, "notes", None),
            "total_elements": len(script.elements),
        },
        "elements": [],
    }

    for i, element in enumerate(script.elements):
        element_data = {
            "index": i,
            "type": element.type.value if hasattr(element, "type") else None,
            "text": element.text if hasattr(element, "text") else None,
            "name": getattr(element, "name", None),
            "extension": getattr(element, "extension", None),
            "dual_dialogue": getattr(element, "dual_dialogue", None),
            "scene_number": getattr(element, "scene_number", None),
            "is_centered": getattr(element, "is_centered", None),
            "is_dual_dialogue": getattr(element, "is_dual_dialogue", None),
            "original_line": getattr(element, "original_line", None),
        }
        fountain_data["elements"].append(element_data)

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(fountain_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved Fountain objects to {output_file}")
    print(f"Total elements: {len(fountain_data['elements'])}")

    # Show some statistics
    element_types = {}
    for element in fountain_data["elements"]:
        element_type = element["type"]
        if element_type:
            element_types[element_type] = element_types.get(element_type, 0) + 1

    print("\nElement type statistics:")
    for element_type, count in sorted(element_types.items()):
        print(f"  {element_type}: {count}")

    # Show first few elements as preview
    print("\nFirst 10 elements preview:")
    for i, element in enumerate(fountain_data["elements"][:10]):
        print(
            f"  {i}: {element['type']} - name: {element['name']} - text: {repr(element['text'][:50] if element['text'] else '')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
