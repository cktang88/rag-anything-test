#!/usr/bin/env python3
"""
RAG-Anything Example Script with MinerU v2.5 (Fixed Version)
This script demonstrates how to use RAG-Anything for multimodal document processing.
Fixed to avoid the event loop warning.
"""

import asyncio
import os
import argparse
import atexit
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# Global variable to store RAG instance for cleanup
rag_instance = None


def cleanup_rag():
    """Cleanup function to be called on exit."""
    global rag_instance
    if rag_instance:
        try:
            # Force cleanup of any remaining async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(asyncio.sleep(0.1))
            loop.close()
        except Exception:
            pass  # Ignore cleanup errors


async def main():
    global rag_instance
    
    parser = argparse.ArgumentParser(description="RAG-Anything Example with MinerU v2.5")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", help="OpenAI base URL (optional)")
    parser.add_argument("--parser", choices=["mineru", "docling"], default="mineru", help="Parser to use")
    parser.add_argument("--parse-method", choices=["auto", "ocr", "txt"], default="auto", help="Parse method")
    parser.add_argument("--output-dir", default="./output", help="Output directory for processed documents")
    parser.add_argument("--working-dir", default="./rag_storage", help="Working directory for RAG storage")
    
    args = parser.parse_args()
    
    # Set up API configuration
    api_key = args.api_key
    base_url = args.base_url

    print("🚀 Initializing RAG-Anything with MinerU v2.5...")
    
    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir=args.working_dir,
        parser=args.parser,
        parse_method=args.parse_method,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # Define vision model function for image processing
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # Initialize RAGAnything
    print("📚 Creating RAGAnything instance...")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    # Store for cleanup
    rag_instance = rag

    # Check if MinerU is properly installed
    print("🔍 Checking MinerU installation...")
    if rag.check_parser_installation():
        print("✅ MinerU v2.5 installed properly")
    else:
        print("❌ MinerU installation issue detected")
        return

    # Process the document
    print(f"📄 Processing document: {args.file_path}")
    print(f"🔧 Using parser: {args.parser}")
    print(f"⚙️ Parse method: {args.parse_method}")
    
    try:
        await rag.process_document_complete(
            file_path=args.file_path,
            output_dir=args.output_dir,
            parse_method=args.parse_method,
            display_stats=True
        )
        print("✅ Document processing completed successfully!")
        
        # Example queries
        print("\n🔍 Running example queries...")
        
        # Pure text query
        print("\n📝 Text Query:")
        text_result = await rag.aquery(
            "What are the main findings or key points in this document?",
            mode="hybrid"
        )
        print(f"Answer: {text_result}")
        
        # VLM enhanced query (if images are present)
        print("\n🖼️ VLM Enhanced Query:")
        vlm_result = await rag.aquery(
            "Analyze any images, charts, or figures in the document and explain what they show.",
            mode="hybrid"
        )
        print(f"Answer: {vlm_result}")
        
        # Multimodal query example
        print("\n🔗 Multimodal Query:")
        multimodal_result = await rag.aquery_with_multimodal(
            "Summarize the key information from this document",
            multimodal_content=[{
                "type": "text",
                "text": "This is a sample multimodal query to test the system's capabilities."
            }],
            mode="hybrid"
        )
        print(f"Answer: {multimodal_result}")
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return

    print("\n🎉 RAG-Anything example completed successfully!")
    print(f"📁 Processed files saved to: {args.output_dir}")
    print(f"🗄️ RAG storage location: {args.working_dir}")


def run_main():
    """Run the main function with proper event loop handling."""
    # Register cleanup function
    atexit.register(cleanup_rag)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_main()
