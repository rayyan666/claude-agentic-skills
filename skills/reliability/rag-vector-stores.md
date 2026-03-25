# Skill: RAG & Vector Stores

## Role
You are an expert in retrieval-augmented generation systems. Help design, implement, and optimize RAG pipelines — from document ingestion and chunking through embedding, retrieval, reranking, and generation.

## RAG Pipeline Stages

```
Documents → Chunking → Embedding → Vector Store → [Query] → Retrieval → Reranking → LLM → Answer
```

## Chunking Strategies

| Strategy | Best For | Chunk Size |
|---|---|---|
| Fixed-size | Uniform docs (logs, CSVs) | 256–512 tokens |
| Recursive character | General text | 512–1024 tokens |
| Semantic | Long articles, books | Variable |
| Document-level | Short docs (FAQs, tickets) | Full doc |

**Rule of thumb:** overlap = 10–15% of chunk size (e.g., 100 tokens overlap for 512-token chunks)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)
```

## Embedding Models

| Model | Dims | Use Case |
|---|---|---|
| `text-embedding-3-small` | 1536 | General, cost-efficient |
| `text-embedding-3-large` | 3072 | High accuracy needed |
| `BAAI/bge-large-en-v1.5` | 1024 | Open-source, strong performance |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Fast, offline |

Always normalize embeddings before storing: `embedding / np.linalg.norm(embedding)`

## Vector Store Setup

### FAISS (local/prototype)
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

### Pinecone (production)
```python
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-east-1")
index = pinecone.Index("my-index")
vectorstore = Pinecone(index, embeddings.embed_query, "text")
```

## Retrieval Patterns

### Hybrid search (recommended for production)
Combine dense (semantic) + sparse (BM25) retrieval:
```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5
dense = vectorstore.as_retriever(search_kwargs={"k": 5})
ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
```

### Reranking (cross-encoder)
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, doc.page_content) for doc in retrieved_docs])
reranked = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
top_k = reranked[:3]
```

## Generation Prompt Template
```
Answer the question using ONLY the context below. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
```

## Evaluation Metrics
- **Faithfulness** — does the answer stick to the retrieved context? (use Ragas)
- **Answer relevancy** — does it answer the actual question?
- **Context recall** — did retrieval fetch the right chunks?
- **Latency** — target < 2s for interactive, < 10s for batch

## Common Failure Modes
- Chunks too large → irrelevant context dilutes the answer
- No overlap → important context split across chunk boundaries
- Wrong embedding model → poor semantic similarity
- Missing metadata filtering → retrieval from wrong document set
