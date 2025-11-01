"""Hybrid retrieval system (BM25 + dense embeddings)."""
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import (
    PERSONA_DIR,
    VECTOR_STORE_TYPE,
    EMBEDDING_MODEL,
    K_RETRIEVE_INITIAL,
)
from src.data.models import CanonicalFact


class HybridRetriever:
    """Hybrid retriever using BM25 + dense embeddings."""
    
    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.facts_file = PERSONA_DIR / persona_name / "canonical_facts.jsonl"
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Load facts
        self.facts: List[CanonicalFact] = []
        self.fact_texts: List[str] = []
        self._load_facts()
        
        # Initialize BM25 (only if we have facts)
        if self.facts:
            tokenized_corpus = [fact.text.lower().split() for fact in self.facts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
        
        # Initialize vector store (only if we have facts)
        if self.facts and VECTOR_STORE_TYPE == "chroma":
            self._init_chroma()
        elif not self.facts:
            self.collection = None  # No facts, no vector store needed
        else:
            raise NotImplementedError(f"Vector store type {VECTOR_STORE_TYPE} not implemented")
    
    def _load_facts(self):
        """Load canonical facts from JSONL."""
        if not self.facts_file.exists():
            return
        
        with open(self.facts_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    fact = CanonicalFact(**data)
                    self.facts.append(fact)
                    self.fact_texts.append(fact.text)
    
    def _init_chroma(self):
        """Initialize ChromaDB for dense embeddings using new API."""
        # Use PersistentClient for the new ChromaDB API
        persist_path = str(PERSONA_DIR / self.persona_name / "chroma_db")
        client = chromadb.PersistentClient(path=persist_path)
        
        collection_name = f"{self.persona_name}_facts"
        
        # Try to get existing collection or create new
        try:
            self.collection = client.get_collection(collection_name)
            # Verify it has the right number of documents
            count = self.collection.count()
            if count != len(self.facts):
                # Rebuild if mismatch
                client.delete_collection(collection_name)
                self.collection = None
        except:
            self.collection = None
        
        if self.collection is None:
            # Create and populate collection
            self.collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            
            # Generate embeddings and add to collection
            embeddings = self.embedding_model.encode(
                self.fact_texts,
                show_progress_bar=True,
            )
            
            ids = [fact.id for fact in self.facts]
            metadatas = [
                {
                    "source": fact.source,
                    "date": fact.date or "",
                    "confidence": str(fact.confidence),
                }
                for fact in self.facts
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=self.fact_texts,
                metadatas=metadatas,
            )
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 search."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense embedding search."""
        if not self.collection:
            return []
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        
        # Map back to indices
        retrieved_ids = results["ids"][0]
        distances = results["distances"][0]
        
        id_to_idx = {fact.id: idx for idx, fact in enumerate(self.facts)}
        results_list = []
        for doc_id, distance in zip(retrieved_ids, distances):
            if doc_id in id_to_idx:
                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1.0 - distance
                results_list.append((id_to_idx[doc_id], similarity))
        
        return results_list
    
    def search(self, query: str, k: int = K_RETRIEVE_INITIAL) -> List[Dict[str, Any]]:
        """
        Hybrid search: combine BM25 and dense, return top-k.
        
        Returns list of dicts with keys: fact, score, fact_id
        """
        if not self.facts:
            return []
        
        # Get results from both methods
        bm25_results = self._bm25_search(query, k) if self.bm25 else []
        dense_results = self._dense_search(query, k) if self.collection else []
        
        # Combine and normalize scores
        combined_scores: Dict[int, float] = {}
        
        # Normalize BM25 scores (0-1 range)
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            min_bm25 = min(score for _, score in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            
            for idx, score in bm25_results:
                normalized = (score - min_bm25) / bm25_range if bm25_range > 0 else 0.5
                combined_scores[idx] = combined_scores.get(idx, 0) + 0.5 * normalized
        
        # Add dense scores (already 0-1 for cosine similarity)
        for idx, score in dense_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + 0.5 * score
        
        # Sort by combined score and take top-k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        
        # Format results
        results = []
        for idx, score in sorted_results:
            fact = self.facts[idx]
            results.append({
                "fact": fact,
                "score": score,
                "fact_id": fact.id,
                "text": fact.text,
                "confidence": fact.confidence,
                "source": fact.source,
            })
        
        return results
    
    def build_conversation_query(
        self,
        current_message: str,
        conversation_history: List[Dict[str, str]],
        entity_mentions: List[str],
    ) -> str:
        """
        Build enhanced query from conversation context.
        
        Combines current message + recent history + entity mentions.
        """
        parts = [current_message]
        
        # Add last 3 turns
        if conversation_history:
            recent = conversation_history[-3:]
            for turn in recent:
                parts.append(turn.get("user", ""))
                parts.append(turn.get("assistant", ""))
        
        # Add entity mentions
        if entity_mentions:
            parts.extend(entity_mentions)
        
        return " ".join(parts)

