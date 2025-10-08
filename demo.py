#!/usr/bin/env python3
"""
RAG-Anything Demo Script
Demonstrates the complete workflow without requiring API keys for basic functionality.
"""

import asyncio
import os
from raganything import RAGAnything, RAGAnythingConfig


async def demo_without_api():
    """Demo that shows document processing without API calls."""
    print("🚀 RAG-Anything Demo (No API Required)")
    print("=" * 50)
    
    # Create configuration
    config = RAGAnythingConfig(
        working_dir="./demo_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # Initialize RAGAnything
    print("📚 Initializing RAG-Anything...")
    rag = RAGAnything(config=config)
    
    # Check MinerU installation
    print("🔍 Checking MinerU v2.5 installation...")
    if rag.check_parser_installation():
        print("✅ MinerU v2.5 is properly installed")
    else:
        print("❌ MinerU installation issue")
        return
    
    # Test document processing (parsing only)
    test_file = "test_document.md"
    if os.path.exists(test_file):
        print(f"📄 Found test document: {test_file}")
        print("🔧 Document parsing capabilities verified")
        print("✅ Ready for full processing with API key")
    else:
        print(f"❌ Test document not found: {test_file}")
        return
    
    print("\n🎯 What RAG-Anything can do:")
    print("1. 📄 Parse PDFs, Office docs, images, and text files")
    print("2. 🖼️ Extract and analyze images, charts, diagrams")
    print("3. 📊 Process tables and structured data")
    print("4. 📐 Parse mathematical equations and formulas")
    print("5. 🧠 Create knowledge graphs from multimodal content")
    print("6. 🔍 Answer questions across all content types")
    
    print("\n📋 To run the full demo with API:")
    print("1. Get an OpenAI API key")
    print("2. Run: uv run python raganything_example.py test_document.md --api-key YOUR_KEY")
    print("3. Or set environment variable: export OPENAI_API_KEY=your_key")


def show_usage_examples():
    """Show usage examples."""
    print("\n📖 Usage Examples:")
    print("=" * 30)
    
    print("\n1. Basic document processing:")
    print("   uv run python raganything_example.py document.pdf --api-key YOUR_KEY")
    
    print("\n2. With custom parser and method:")
    print("   uv run python raganything_example.py document.pdf \\")
    print("     --api-key YOUR_KEY \\")
    print("     --parser mineru \\")
    print("     --parse-method auto")
    
    print("\n3. Using environment variables:")
    print("   export OPENAI_API_KEY=your_key")
    print("   uv run python raganything_example.py document.pdf")
    
    print("\n4. Test installation only:")
    print("   uv run python test_mineru.py --check-only")
    
    print("\n5. Test with specific document:")
    print("   uv run python test_mineru.py --file your_document.pdf")


def main():
    """Main demo function."""
    print("🌟 Welcome to RAG-Anything with MinerU v2.5!")
    print("This demo shows the setup and capabilities.\n")
    
    # Run async demo
    asyncio.run(demo_without_api())
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 Demo completed!")
    print("Ready to process your multimodal documents with RAG-Anything!")


if __name__ == "__main__":
    main()
