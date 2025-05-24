import os
import re
import logging
import numpy as np
import requests
import json
import time
from sklearn.metrics.pairwise import cosine_similarity

class RAGService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        if not self.api_key:
            self.logger.warning("MISTRAL_API_KEY not found in environment variables")
        
        self.base_url = "https://api.mistral.ai/v1"
        self.documents = {}  # {doc_id: {'text': str, 'title': str, 'chunks': [str], 'embeddings': np.array}}
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def chunk_text(self, text, chunk_size=None, overlap=None):
        """Split text into overlapping chunks"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def get_embeddings(self, texts):
        """Get embeddings for texts using Mistral API"""
        if not self.api_key:
            self.logger.error("Mistral API key not found")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-embed",
                "input": texts
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return np.array([embedding["embedding"] for embedding in result["data"]])
            else:
                self.logger.error(f"Embeddings API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get embeddings: {str(e)}")
            return None
    
    def add_document(self, doc_id, text, title):
        """Add a document to the knowledge base"""
        try:
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Get embeddings for chunks
            embeddings = self.get_embeddings(chunks)
            
            if embeddings is not None:
                self.documents[doc_id] = {
                    'text': text,
                    'title': title,
                    'chunks': chunks,
                    'embeddings': embeddings
                }
                self.logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            else:
                self.logger.error(f"Failed to create embeddings for document {doc_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to add document {doc_id}: {str(e)}")
    
    def remove_document(self, doc_id):
        """Remove a document from the knowledge base"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.logger.info(f"Removed document {doc_id}")
    
    def search_similar_chunks(self, query, top_k=5):
        """Find most similar chunks to the query"""
        if not self.documents:
            return []
        
        # Get query embedding
        query_embedding = self.get_embeddings([query])
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding[0]
        
        # Collect all chunks with their metadata
        all_chunks = []
        for doc_id, doc_data in self.documents.items():
            for i, chunk in enumerate(doc_data['chunks']):
                all_chunks.append({
                    'doc_id': doc_id,
                    'chunk_idx': i,
                    'text': chunk,
                    'title': doc_data['title'],
                    'embedding': doc_data['embeddings'][i]
                })
        
        if not all_chunks:
            return []
        
        # Calculate similarities
        chunk_embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get top_k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk_data = all_chunks[idx]
            results.append({
                'text': chunk_data['text'],
                'title': chunk_data['title'],
                'doc_id': chunk_data['doc_id'],
                'similarity': similarities[idx]
            })
        
        return results
    
    def chat(self, message):
        """Generate response using RAG"""
        if not self.api_key:
            return {
                'answer': "Sorry, the AI service is not available. Please check that MISTRAL_API_KEY is set.",
                'sources': []
            }
        
        try:
            # Search for relevant chunks
            relevant_chunks = self.search_similar_chunks(message, top_k=3)
            
            if not relevant_chunks:
                return {
                    'answer': "I don't have any relevant information to answer your question. Please make sure you have uploaded and processed some YouTube videos.",
                    'sources': []
                }
            
            # Create context from relevant chunks
            context = "\n\n".join([f"From '{chunk['title']}':\n{chunk['text']}" for chunk in relevant_chunks])
            
            # Create prompt
            prompt = f"""Based on the following YouTube video transcripts, please answer the user's question. If the answer is not in the transcripts, say so clearly.

Context from video transcripts:
{context}

User question: {message}

Please provide a helpful and accurate answer based only on the information in the transcripts above."""

            # Get response from Mistral
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            # Retry logic for rate limiting
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    break
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.info(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            'answer': "I'm getting too many requests right now. Please wait a moment and try again.",
                            'sources': []
                        }
                else:
                    self.logger.error(f"Chat API error: {response.status_code} - {response.text}")
                    return {
                        'answer': "Sorry, I encountered an error with the AI service. Please try again.",
                        'sources': []
                    }
            
            # Prepare sources
            sources = []
            for chunk in relevant_chunks:
                source = {
                    'title': chunk['title'],
                    'doc_id': chunk['doc_id'],
                    'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                }
                if source not in sources:
                    sources.append(source)
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            self.logger.error(f"Chat error: {str(e)}")
            return {
                'answer': "Sorry, I encountered an error while processing your question. Please try again.",
                'sources': []
            }
