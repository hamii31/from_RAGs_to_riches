import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import pickle

try:
    import PyPDF2
    import requests
except ImportError:
    import subprocess
    import sys
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2", "requests", "numpy"])
    import PyPDF2
    import requests


@dataclass
class TextChunk:
    """Represents a chunk of text from the book"""
    text: str
    page_number: int
    chunk_id: int
    embedding: List[float] = None
    metadata: Dict = None


class BookRAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system for large books
    
    Features:
    - Indexes entire book into searchable chunks
    - Uses embeddings for semantic search
    - Retrieves only relevant sections for a query
    - Generates answers based on relevant context
    """
    
    def __init__(
        self, 
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api"
        
        self.chunks: List[TextChunk] = []
        self.index_built = False
        
        self.check_models()
    
    def check_models(self):
        """Check if required models are available"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            models = [m['name'] for m in response.json().get('models', [])]
            
            if self.embedding_model not in models:
                print(f"Pulling embedding model: {self.embedding_model}")
                print("This is needed for semantic search (one-time setup)...")
                os.system(f"ollama pull {self.embedding_model}")
            
            print(f"✓ Models ready: {self.model_name}, {self.embedding_model}")
        except Exception as e:
            print(f"Warning: Could not verify models: {e}")
    
    def extract_text_with_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract text from PDF, keeping track of page numbers
        
        Returns:
            List of (page_number, text) tuples
        """
        print(f"Extracting text from: {pdf_path}")
        pages_text = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        pages_text.append((page_num, text))
                    
                    if page_num % 50 == 0:
                        print(f"  Processed {page_num}/{total_pages} pages...")
                
                print(f"✓ Extracted text from {len(pages_text)} pages")
                return pages_text
        
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return []
    
    def create_chunks(
        self, 
        pages_text: List[Tuple[int, str]], 
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks for better context
        
        Args:
            pages_text: List of (page_num, text) tuples
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
        """
        print(f"Creating chunks (size={chunk_size}, overlap={overlap})...")
        chunks = []
        chunk_id = 0
        
        for page_num, page_text in pages_text:
            # Split by paragraphs first
            paragraphs = page_text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(TextChunk(
                            text=current_chunk.strip(),
                            page_number=page_num,
                            chunk_id=chunk_id,
                            metadata={'length': len(current_chunk)}
                        ))
                        chunk_id += 1
                    
                    # Keep overlap from previous chunk
                    current_chunk = current_chunk[-overlap:] + para + "\n\n"
            
            # Add remaining text
            if current_chunk.strip():
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    page_number=page_num,
                    chunk_id=chunk_id,
                    metadata={'length': len(current_chunk)}
                ))
                chunk_id += 1
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using Ollama
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                return response.json()['embedding']
            else:
                print(f"Error getting embedding: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def build_index(self, pdf_path: str, cache_path: str = None):
        """
        Build searchable index of the entire book
        
        Args:
            pdf_path: Path to PDF file
            cache_path: Optional path to save/load index cache
        """
        # Check if cache exists
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached index from: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.chunks = pickle.load(f)
            self.index_built = True
            print(f"✓ Loaded {len(self.chunks)} chunks from cache")
            return
        
        # Extract and chunk text
        pages_text = self.extract_text_with_pages(pdf_path)
        if not pages_text:
            print("Failed to extract text")
            return
        
        self.chunks = self.create_chunks(pages_text)
        
        # Generate embeddings for each chunk
        print("Generating embeddings for semantic search...")
        print("This may take a while for large books (one-time process)...")
        
        for i, chunk in enumerate(self.chunks):
            if i % 20 == 0:
                print(f"  Processing chunk {i+1}/{len(self.chunks)}...")
            
            # Truncate very long chunks for embedding
            text_for_embedding = chunk.text[:500]
            chunk.embedding = self.get_embedding(text_for_embedding)
        
        self.index_built = True
        print("✓ Index built successfully")
        
        # Save cache
        if cache_path:
            print(f"Saving index cache to: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            print("✓ Cache saved")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_array = np.array(a)
        b_array = np.array(b)
        
        return np.dot(a_array, b_array) / (
            np.linalg.norm(a_array) * np.linalg.norm(b_array)
        )
    
    def search_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Tuple[TextChunk, float]]:
        """
        Search for chunks most relevant to the query
        
        Args:
            query: Search query (e.g., "thalamus")
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.index_built:
            print("Error: Index not built. Call build_index() first.")
            return []
        
        print(f"Searching for: '{query}'")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        if not query_embedding:
            print("Failed to get query embedding")
            return []
        
        # Calculate similarity with all chunks
        similarities = []
        for chunk in self.chunks:
            if chunk.embedding:
                similarity = self.cosine_similarity(query_embedding, chunk.embedding)
                similarities.append((chunk, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        top_results = similarities[:top_k]
        
        print(f"✓ Found {len(top_results)} relevant sections")
        for i, (chunk, score) in enumerate(top_results[:3], 1):
            print(f"  {i}. Page {chunk.page_number} (similarity: {score:.3f})")
        
        return top_results
    
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[Tuple[TextChunk, float]],
        max_context_length: int = 6000
    ) -> str:
        """
        Generate answer using relevant context
        
        Args:
            query: User's question
            context_chunks: Relevant chunks with similarity scores
            max_context_length: Maximum characters to include in context
            
        Returns:
            Generated answer
        """
        # Build context from top chunks
        context_parts = []
        current_length = 0
        
        for chunk, score in context_chunks:
            if current_length + len(chunk.text) > max_context_length:
                break
            context_parts.append(
                f"[Page {chunk.page_number}]\n{chunk.text}\n"
            )
            current_length += len(chunk.text)
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant analyzing a book. Based on the following excerpts from the book, answer the user's question comprehensively.

        Question: {query}
        
        Relevant excerpts from the book:
        
        {context}
        
        Instructions:
        1. Provide a comprehensive answer based on the excerpts above
        2. Cite page numbers when referencing specific information
        3. If the information spans multiple pages, mention the page range
        4. If the excerpts don't fully answer the question, acknowledge what's covered and what's not
        5. Be specific and detailed in your response
        
        Answer:"""
        
        print("💭 Generating answer...")
        
        # Call Ollama
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"Error: {response.status_code}"
        
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def query(self, question: str, top_k: int = 10) -> Dict:
        """
        Main query interface - search and generate answer
        
        Args:
            question: User's question (e.g., "What does the book say about the thalamus?")
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.index_built:
            return {
                "error": "Index not built. Call build_index() first.",
                "answer": None
            }
        
        # Search for relevant chunks
        relevant_chunks = self.search_relevant_chunks(question, top_k=top_k)
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the book.",
                "sources": [],
                "pages": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        # Extract source information
        pages = sorted(set([chunk.page_number for chunk, _ in relevant_chunks]))
        sources = [
            {
                "page": chunk.page_number,
                "similarity": float(score),
                "preview": chunk.text[:200] + "..."
            }
            for chunk, score in relevant_chunks[:5]
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "pages": pages,
            "total_chunks_used": len(relevant_chunks)
        }


def main():
    """Example usage"""
    import sys
    
    print("="*80)
    print("Book RAG System - Query Large PDFs with Ollama")
    print("="*80)
    
    # Initialize system
    rag = BookRAGSystem(
        model_name="llama3.2",
        embedding_model="nomic-embed-text"
    )
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("\nEnter path to book PDF: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Build index (with caching)
    cache_path = f"./cache/{Path(pdf_path).stem}_index.pkl"
    rag.build_index(pdf_path, cache_path=cache_path)
    
    print("\n" + "="*80)
    print("Index built!")
    print("="*80)
    
    # Query loop
    while True:
        print("\n" + "-"*80)
        query = input("\nYour question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Query the book
        result = rag.query(query, top_k=10)
        
        # Display results
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        
        print("\n" + "-"*80)
        print(f"Sources: Pages {', '.join(map(str, result['pages']))}")
        print(f"Used {result['total_chunks_used']} relevant sections")
        
        # Optional fact-check
        show_sources = input("\nShow source excerpts? (y/n): ").lower()
        if show_sources == 'y':
            print("\n" + "="*80)
            print("SOURCE EXCERPTS:")
            print("="*80)
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. Page {source['page']} (relevance: {source['similarity']:.3f})")
                print(f"{source['preview']}")
                print("-" * 60)


if __name__ == "__main__":
    main()
