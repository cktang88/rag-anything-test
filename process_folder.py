#!/usr/bin/env python3
"""
Process entire input_files folder with RAG-Anything
This script will process all files in the input_files directory.
"""

import asyncio
import os
import argparse
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


async def process_folder():
    parser = argparse.ArgumentParser(description="Process entire input_files folder with RAG-Anything")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", help="OpenAI base URL (optional)")
    parser.add_argument("--input-dir", default="./input_files", help="Input directory to process")
    parser.add_argument("--output-dir", default="./output", help="Output directory for processed documents")
    parser.add_argument("--working-dir", default="./rag_storage", help="Working directory for RAG storage")
    parser.add_argument("--parser", choices=["mineru", "docling"], default="mineru", help="Parser to use")
    parser.add_argument("--parse-method", choices=["auto", "ocr", "txt"], default="auto", help="Parse method")
    parser.add_argument("--file-extensions", nargs="+", default=[".pdf", ".html", ".docx", ".pptx", ".txt", ".md"], 
                       help="File extensions to process")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Set up API configuration
    api_key = args.api_key
    base_url = args.base_url

    print("🚀 Processing entire folder with RAG-Anything...")
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🗄️ RAG storage: {args.working_dir}")
    print(f"🔧 Parser: {args.parser}")
    print(f"⚙️ Parse method: {args.parse_method}")
    print(f"📄 File extensions: {args.file_extensions}")
    
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
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
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
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
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

    # Check if MinerU is properly installed
    print("🔍 Checking MinerU installation...")
    if rag.check_parser_installation():
        print("✅ MinerU v2.5 installed properly")
    else:
        print("❌ MinerU installation issue detected")
        return

    # Find all files to process
    files_to_process = []
    for root, dirs, files in os.walk(args.input_dir):
        if not args.recursive and root != args.input_dir:
            continue
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in args.file_extensions):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)

    print(f"\n📄 Found {len(files_to_process)} files to process:")
    for i, file_path in enumerate(files_to_process, 1):
        print(f"  {i}. {os.path.basename(file_path)}")

    if not files_to_process:
        print("❌ No files found to process!")
        return

    # Process each file
    processed_count = 0
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n📄 Processing file {i}/{len(files_to_process)}: {os.path.basename(file_path)}")
        
        try:
            await rag.process_document_complete(
                file_path=file_path,
                output_dir=args.output_dir,
                parse_method=args.parse_method,
                display_stats=True
            )
            processed_count += 1
            print(f"✅ Successfully processed: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
            continue

    print(f"\n🎉 Folder processing completed!")
    print(f"✅ Successfully processed: {processed_count}/{len(files_to_process)} files")
    print(f"📁 Processed files saved to: {args.output_dir}")
    print(f"🗄️ RAG storage location: {args.working_dir}")
    
    # Test queries across all processed files
    if processed_count > 0:
        print(f"\n🔍 Testing queries across all {processed_count} processed files...")
        
        # Test query 1: General knowledge
        print("\n📝 General Knowledge Query:")
        result1 = await rag.aquery(
            "What are the main topics covered in all the processed documents?",
            mode="hybrid"
        )
        print(f"Answer: {result1}")
        
        # Test query 2: Specific content
        print("\n🔍 Specific Content Query:")
        result2 = await rag.aquery(
            "Are there any characters, places, or scientific concepts mentioned across the documents?",
            mode="hybrid"
        )
        print(f"Answer: {result2}")


if __name__ == "__main__":
    try:
        asyncio.run(process_folder())
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
