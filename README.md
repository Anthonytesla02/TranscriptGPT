# YouTube Transcript RAG Chatbot

A powerful Flask web application that extracts YouTube video transcripts and creates an intelligent chatbot using Retrieval-Augmented Generation (RAG) with Mistral AI.

## 🎯 Overview

This application transforms YouTube videos into an interactive knowledge base. Users can upload multiple YouTube videos, and the system automatically extracts transcripts, processes them into searchable chunks, creates embeddings, and enables natural language conversations with the content.

## 🔧 How It Works

### 1. YouTube Transcript Extraction

**File: `transcript_service.py`**

The transcript extraction process follows these steps:

```python
def extract_transcript(self, youtube_url):
    """Extract transcript from YouTube video"""
    # 1. Extract video ID from various YouTube URL formats
    video_id = self.extract_video_id(youtube_url)
    
    # 2. Fetch transcript using YouTube Transcript API
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    
    # 3. Combine text segments into full transcript
    transcript_text = ' '.join([item['text'] for item in transcript_list])
    
    # 4. Clean transcript (remove annotations, normalize whitespace)
    transcript_text = re.sub(r'\[.*?\]', '', transcript_text)
    transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()
```

**Supported URL Formats:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

### 2. Text Chunking and Processing

**File: `rag_service.py`**

The RAG service processes transcripts into manageable chunks:

```python
def chunk_text(self, text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for better context preservation"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Smart boundary detection - break at sentence endings
        if end < len(text):
            for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity
```

**Chunking Strategy:**
- **Chunk Size:** 1000 characters (optimal for embeddings)
- **Overlap:** 200 characters (maintains context between chunks)
- **Smart Boundaries:** Breaks at sentence endings when possible

### 3. Embedding Generation

The system uses Mistral's embedding API to convert text chunks into vector representations:

```python
def get_embeddings(self, texts):
    """Generate embeddings using Mistral API"""
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "mistral-embed",
        "input": texts
    }
    
    response = requests.post(f"{self.base_url}/embeddings", headers=headers, json=data)
    return np.array([embedding["embedding"] for embedding in result["data"]])
```

**Embedding Details:**
- **Model:** `mistral-embed`
- **Dimensions:** High-dimensional vectors for semantic similarity
- **Storage:** NumPy arrays for efficient similarity calculations

### 4. Vector Search and Retrieval

When users ask questions, the system performs semantic search:

```python
def search_similar_chunks(self, query, top_k=5):
    """Find most relevant chunks using cosine similarity"""
    # 1. Generate query embedding
    query_embedding = self.get_embeddings([query])[0]
    
    # 2. Calculate similarities with all stored chunks
    chunk_embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # 3. Return top-k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[idx] for idx in top_indices]
```

### 5. RAG Response Generation

The chatbot generates responses using retrieved context:

```python
def chat(self, message):
    """Generate contextual response using RAG"""
    # 1. Find relevant chunks
    relevant_chunks = self.search_similar_chunks(message, top_k=3)
    
    # 2. Build context from retrieved chunks
    context = "\n\n".join([f"From '{chunk['title']}':\n{chunk['text']}" 
                          for chunk in relevant_chunks])
    
    # 3. Create structured prompt
    prompt = f"""Based on the following YouTube video transcripts, please answer the user's question.

Context from video transcripts:
{context}

User question: {message}

Please provide a helpful and accurate answer based only on the information in the transcripts above."""

    # 4. Generate response with Mistral
    response = requests.post(f"{self.base_url}/chat/completions", 
                           headers=headers, json=chat_data)
```

### 6. Database Schema

**File: `models.py`**

```python
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    youtube_url = db.Column(db.String(500), unique=True, nullable=False)
    transcript = db.Column(db.Text, nullable=False)
    processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

### 7. Frontend Architecture

**Chat Interface (`static/js/app.js`):**

```javascript
class ChatApp {
    async handleSubmit(e) {
        // 1. Send user message
        this.addMessage(message, 'user');
        
        // 2. Show typing indicator
        const typingIndicator = this.addTypingIndicator();
        
        // 3. Call chat API
        const response = await fetch('/chat/message', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message })
        });
        
        // 4. Display response with sources
        this.addMessage(data.response, 'assistant', data.sources);
    }
}
```

## 🚀 API Endpoints

### Core Routes

**Upload Videos:**
```
POST /upload
- Accepts multiple URLs (newlines, commas, spaces separated)
- Extracts transcripts
- Generates embeddings
- Stores in database
```

**Chat Interface:**
```
GET /chat
- Displays chat interface
- Shows knowledge base videos
- Real-time messaging
```

**Chat API:**
```
POST /chat/message
- Processes user questions
- Performs semantic search
- Returns AI responses with sources
```

**Video Management:**
```
POST /delete_video/<id>
- Removes video from database
- Cleans up embeddings
- Updates knowledge base
```

## 🛠 Technology Stack

- **Backend:** Flask (Python)
- **Database:** SQLAlchemy with PostgreSQL
- **AI/ML:** Mistral AI API
- **Vector Operations:** NumPy, scikit-learn
- **YouTube API:** youtube-transcript-api
- **Frontend:** Bootstrap 5, Vanilla JavaScript
- **Styling:** Replit Dark Theme

## 🔒 Security Features

- **API Key Management:** Environment variable storage
- **Rate Limiting:** Intelligent retry with exponential backoff
- **Input Validation:** URL format verification
- **Error Handling:** Graceful degradation

## 📊 Performance Optimizations

1. **Chunking Strategy:** Optimal size for embedding quality
2. **Caching:** In-memory storage of embeddings
3. **Batch Processing:** Multiple videos in single request
4. **Smart Retries:** Automatic handling of API limits

## 🎨 User Experience

- **Responsive Design:** Works on all devices
- **Real-time Chat:** Instant responses with typing indicators
- **Source Attribution:** Shows which videos provided answers
- **Progress Feedback:** Clear status messages and loading states

## 🔄 Data Flow

```
1. User uploads YouTube URLs
   ↓
2. Extract video transcripts
   ↓
3. Split into chunks with overlap
   ↓
4. Generate embeddings via Mistral API
   ↓
5. Store in vector database
   ↓
6. User asks question
   ↓
7. Convert question to embedding
   ↓
8. Find similar chunks (cosine similarity)
   ↓
9. Build context from top chunks
   ↓
10. Generate response via Mistral Chat API
    ↓
11. Display answer with sources
```

This architecture ensures fast, accurate, and contextually relevant responses based on your YouTube video content!