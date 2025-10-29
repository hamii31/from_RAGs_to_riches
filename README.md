# 📚 Ollama Book RAG

Query large books (900+ pages) locally using RAG (Retrieval-Augmented Generation). 100% private, no cloud APIs.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ What It Does

Ask questions about **any book** and get detailed, cited answers:
```bash
$ python ollama_book_rag.py textbook.pdf

Your question: What does the book say about the thalamus?

🔍 Searching for: 'thalamus'
✓ Found 10 relevant sections
💭 Generating answer...

The thalamus serves as a critical relay station for sensory 
information (Page 234). It processes nearly all sensory inputs 
before they reach the cortex...

[Additional detailed answer with multiple page citations]

📄 Sources: Pages 234, 235, 236, 237, 412, 413
```

**Features:**
- 🔒 100% Local & Private
- 📚 Handles 900+ page books
- ⚡ Fast queries (5-10 sec after indexing)
- 💾 Smart caching (index once, query forever)
- 📄 Automatic page citations

## 🚀 Quick Start

### 1. Install Ollama
```bash
# Windows
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh
```

### 2. Pull Models
```bash
# LLM (choose based on your RAM)
ollama pull llama3.2        # 16GB+ RAM
# OR
ollama pull phi3            # 8GB RAM

# Embedding model (required)
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run It
```bash
python ollama_book_rag.py your_book.pdf
```

## 💡 Usage

### Basic Usage
```python
from ollama_book_rag import BookRAGSystem

# Initialize
rag = BookRAGSystem(model_name="llama3.2")

# Build index (one-time per book, ~10 min for 900 pages)
rag.build_index(
    "neuroscience.pdf",
    cache_path="./cache/neuro_index.pkl"
)

# Query (instant after indexing)
result = rag.query("What is neuroplasticity?")

print(result['answer'])
print(f"Sources: Pages {result['pages']}")
```

### Interactive Mode
```bash
# Run with any PDF
python ollama_book_rag.py book.pdf

# Ask unlimited questions
Your question: What about X?
Your question: Explain Y
Your question: Describe Z
```

### Multiple Questions
```python
questions = [
    "What is the main thesis?",
    "What methodology was used?",
    "What are the key findings?"
]

for q in questions:
    result = rag.query(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print(f"Pages: {result['pages']}")
```

## ⚙️ Configuration
```python
# Use different models
rag = BookRAGSystem(
    model_name="phi3",              # For 8GB RAM
    embedding_model="nomic-embed-text"
)

# Retrieve more context
result = rag.query(question, top_k=15)  # Default is 10

# Adjust chunk size (in the code)
chunks = rag.create_chunks(
    pages_text, 
    chunk_size=1000,    # Larger = more context
    overlap=200         # Overlap between chunks
)
```

## 🎯 Model Recommendations

| Your RAM | Model | Command |
|----------|-------|---------|
| 8 GB | phi3 | `ollama pull phi3` |
| 16 GB+ | llama3.2 | `ollama pull llama3.2` |
| 32 GB+ | llama3.1:8b | `ollama pull llama3.1:8b` |

**Always needed:** `ollama pull nomic-embed-text`

## 🔍 How It Works
```
1. Index Book (one-time)
   PDF → Extract text → Split into chunks → Generate embeddings → Cache

2. Query Book (instant)
   Question → Find relevant chunks → Send to Ollama → Get answer
```

**Why RAG?**
- Traditional: Can't handle 900 pages (context limit)
- RAG: Searches first, reads only relevant parts

## 🐛 Troubleshooting

### "Ollama is not running"
```bash
# Start Ollama
ollama serve

# Verify
ollama list
```

### "Model not found"
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### "Out of memory"
```python
# Use smaller model
rag = BookRAGSystem(model_name="phi3")
```

### Slow first-time indexing?
**This is normal!** 
- 900 pages = ~10-15 minutes
- Only happens once (cached after)
- Queries are instant afterward

## 📊 Performance

| Book Size | Index Time | Query Time | Cache Size |
|-----------|------------|------------|------------|
| 100 pages | 2 min | 5 sec | 30 MB |
| 500 pages | 7 min | 6 sec | 120 MB |
| 900 pages | 12 min | 7 sec | 245 MB |

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Credits

- [Ollama](https://ollama.ai/) - Local LLM framework
- [PyPDF2](https://pypdf2.readthedocs.io/) - PDF extraction
- [Nomic](https://www.nomic.ai/) - Embedding model

## 🆘 Support

**Issues**: [GitHub Issues](https://github.com/hamii31/from_RAGs_to_riches/issues)

---

⭐ **Star this repo if it helped you!**
