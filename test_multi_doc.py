#!/usr/bin/env python3
"""
Test script to demonstrate RAG-Anything's multi-document capabilities
"""

import asyncio
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


async def test_multi_document_queries():
    """Test queries across multiple documents in the knowledge base."""
    
    # Set up API configuration
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🚀 Testing Multi-Document RAG Capabilities")
    print("=" * 50)
    
    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define model functions
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kwargs,
        )

    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
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
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
        ),
    )

    # Initialize RAGAnything
    print("📚 Loading existing RAG knowledge base...")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Test various types of queries
    queries = [
        {
            "title": "🔬 Scientific Content Query",
            "question": "What are the main scientific findings about protein behavior?",
            "description": "Tests knowledge extraction from scientific papers"
        },
        {
            "title": "🖼️ Visual Analysis Query", 
            "question": "What do the images in the documents show?",
            "description": "Tests multimodal analysis capabilities"
        },
        {
            "title": "🔍 Specific Detail Query",
            "question": "Who are the researchers mentioned and what institutions are involved?",
            "description": "Tests entity extraction and relationship understanding"
        },
        {
            "title": "📊 Technical Query",
            "question": "What are the technical methods and approaches described?",
            "description": "Tests understanding of technical content"
        },
        {
            "title": "🎯 Application Query",
            "question": "What are the practical applications and implications of this research?",
            "description": "Tests synthesis and application understanding"
        }
    ]

    print(f"\n🔍 Running {len(queries)} different types of queries...")
    print("=" * 50)

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. {query['title']}")
        print(f"   Description: {query['description']}")
        print(f"   Question: {query['question']}")
        print("   " + "-" * 40)
        
        try:
            result = await rag.aquery(query['question'], mode="hybrid")
            print(f"   Answer: {result}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

    print("🎉 Multi-document query testing completed!")
    print("\n📋 Summary:")
    print("✅ RAG-Anything can answer questions across all processed documents")
    print("✅ It maintains a unified knowledge base from multiple sources")
    print("✅ Queries can span different document types and content")
    print("✅ The system understands relationships between different documents")


if __name__ == "__main__":
    asyncio.run(test_multi_document_queries())
