#!/usr/bin/env python3
"""
Simple embedding-based RAG system for LOTR screenplay chunks
Uses OpenAI embeddings for similarity search and LLM for final answers
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
import openai
from sklearn.metrics.pairwise import cosine_similarity
import argparse

class SimpleRAG:
    def __init__(self, api_key: str, base_url: str = None):
        """Initialize the RAG system with OpenAI API key"""
        self.api_key = api_key
        self.base_url = base_url
        self.chunks = []
        self.embeddings = None
        self.chunk_embeddings = None
        
        # Set up OpenAI client
        if base_url:
            openai.api_base = base_url
        openai.api_key = api_key
    
    def load_chunks(self, chunks_dir: str = "./chunks_output"):
        """Load all text chunks from the chunks directory"""
        print(f"Loading chunks from {chunks_dir}...")
        
        chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")])
        print(f"Found {len(chunk_files)} chunk files")
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunks_dir, chunk_file)
            with open(chunk_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract just the content between the chunk markers
                lines = content.split('\n')
                chunk_content = []
                in_chunk = False
                for line in lines:
                    if line.startswith("=== CHUNK"):
                        in_chunk = True
                        continue
                    elif line.startswith("=== END CHUNK"):
                        break
                    elif in_chunk and line.strip():
                        chunk_content.append(line)
                
                if chunk_content:
                    self.chunks.append('\n'.join(chunk_content))
        
        print(f"Loaded {len(self.chunks)} chunks")
        return len(self.chunks)
    
    def embed_chunks(self):
        """Create embeddings for all chunks using OpenAI"""
        print("Creating embeddings for all chunks...")
        
        # Create embeddings for all chunks
        chunk_texts = [chunk for chunk in self.chunks]
        
        # Use OpenAI embeddings
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk_texts
        )
        
        self.chunk_embeddings = np.array([item.embedding for item in response.data])
        print(f"Created embeddings with shape: {self.chunk_embeddings.shape}")
        
        return self.chunk_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Create embedding for a query"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return np.array(response.data[0].embedding)
    
    def find_similar_chunks(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Find the most similar chunks to a query"""
        if self.chunk_embeddings is None:
            raise ValueError("Chunks must be embedded first. Call embed_chunks().")
        
        # Embed the query
        query_embedding = self.embed_query(query)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            chunk_text = self.chunks[idx]
            results.append((int(idx), float(similarity_score), chunk_text))
        
        return results
    
    def query(self, question: str, top_k: int = 10) -> str:
        """Answer a question using the most relevant chunks"""
        print(f"Searching for similar chunks to: '{question}'")
        
        # Find similar chunks
        similar_chunks = self.find_similar_chunks(question, top_k)
        
        print(f"Found {len(similar_chunks)} similar chunks:")
        for i, (idx, score, chunk) in enumerate(similar_chunks):
            print(f"  {i+1}. Chunk {idx+1} (similarity: {score:.3f})")
            print(f"     Preview: {chunk[:100]}...")
            print()
        
        # Prepare context for LLM
        context_chunks = []
        for idx, score, chunk in similar_chunks:
            context_chunks.append(f"Chunk {idx+1} (similarity: {score:.3f}):\n{chunk}\n")
        
        context = "\n".join(context_chunks)
        
        # Create prompt for LLM
        prompt = f"""Based on the following context from The Lord of the Rings screenplay, answer the question: {question}

Context:
{context}

Please provide a comprehensive answer based on the context provided. If the answer is not found in the context, say so clearly."""

        # Get answer from LLM
        try:
            response = openai.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about The Lord of the Rings screenplay based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"Error getting LLM response: {e}"
    
    def save_embeddings(self, filepath: str = "./embeddings.json"):
        """Save embeddings to file for faster loading"""
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings to save. Call embed_chunks() first.")
        
        data = {
            "chunks": self.chunks,
            "embeddings": self.chunk_embeddings.tolist()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str = "./embeddings.json"):
        """Load embeddings from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data["chunks"]
        self.chunk_embeddings = np.array(data["embeddings"])
        
        print(f"Loaded {len(self.chunks)} chunks and embeddings from {filepath}")
        return len(self.chunks)

def main():
    parser = argparse.ArgumentParser(description="Simple RAG system for LOTR screenplay")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", help="OpenAI base URL (optional)")
    parser.add_argument("--chunks-dir", default="./chunks_output", help="Directory containing chunk files")
    parser.add_argument("--embeddings-file", default="./embeddings.json", help="File to save/load embeddings")
    parser.add_argument("--load-embeddings", action="store_true", help="Load existing embeddings instead of creating new ones")
    parser.add_argument("--question", help="Question to ask (interactive mode if not provided)")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = SimpleRAG(api_key=args.api_key, base_url=args.base_url)
    
    if args.load_embeddings and os.path.exists(args.embeddings_file):
        print("Loading existing embeddings...")
        rag.load_embeddings(args.embeddings_file)
    else:
        print("Loading chunks and creating embeddings...")
        rag.load_chunks(args.chunks_dir)
        rag.embed_chunks()
        rag.save_embeddings(args.embeddings_file)
    
    # Interactive mode
    if args.question:
        print(f"\nQuestion: {args.question}")
        answer = rag.query(args.question)
        print(f"\nAnswer: {answer}")
    else:
        print("\n🎭 LOTR Screenplay RAG System Ready!")
        print("Ask questions about The Lord of the Rings screenplay. Type 'quit' to exit.")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if question:
                    answer = rag.query(question)
                    print(f"\n💬 Answer: {answer}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
