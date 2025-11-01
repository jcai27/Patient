"""Cross-encoder reranker for retrieval results."""
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.config import K_RETRIEVE


class Reranker:
    """Cross-encoder reranker."""
    
    def __init__(self):
        # Use a cross-encoder model for reranking
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = K_RETRIEVE,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: Search query
            results: List of results from retriever (with 'text' key)
            top_k: Number of top results to return
            
        Returns:
            Reranked results (top_k items)
        """
        if not results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result["text"]) for result in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score and return top-k
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        reranked = [result for result, _ in scored_results[:top_k]]
        
        # Update scores in results
        for i, (_, score) in enumerate(scored_results[:top_k]):
            reranked[i]["rerank_score"] = float(score)
        
        return reranked

