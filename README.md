# YouTube Transcript RAG Chatbot

A powerful Flask web application that extracts YouTube video transcripts and creates an intelligent chatbot using Retrieval-Augmented Generation (RAG) with Mistral AI. This is a premium-ready application with advanced AI features that users will pay for.

## 🎯 Overview

This application transforms YouTube videos into an interactive knowledge base with premium AI features. Users can upload multiple YouTube videos, and the system automatically extracts transcripts, processes them into searchable chunks, creates embeddings, and enables natural language conversations with the content.

### Key Features

**Core Functionality:**
- Multi-format YouTube URL support (youtube.com, youtu.be, embed)
- Automatic transcript extraction and processing
- Vector-based semantic search using embeddings
- Real-time chat interface with contextual responses
- Source attribution and citation

**Premium Features (🚀 Revenue Generators):**
1. **AI-Powered Video Summaries** - Intelligent 2-3 sentence summaries of any video
2. **Smart Question Generator** - AI creates thoughtful questions to explore content deeper
3. **Knowledge Base Insights** - Cross-video analysis revealing themes and connections

**Technical Excellence:**
- Rate limiting with exponential backoff
- Error handling and retry logic
- Responsive dark-themed UI
- Real-time feedback and loading states
- Database schema with relationship management

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

## 🚀 Premium Features Deep Dive

### Feature 1: AI-Powered Video Summaries

**Business Value:** Users save time by getting instant, intelligent summaries instead of watching entire videos.

**Implementation:**
```python
@app.route('/summarize/<int:video_id>')
def summarize_video(video_id):
    video = models.Video.query.get_or_404(video_id)
    
    # Check cache first
    existing_summary = models.VideoSummary.query.filter_by(
        video_id=video_id, summary_type='brief'
    ).first()
    
    if existing_summary:
        return jsonify({'summary': existing_summary.content})
    
    # Generate new summary
    summary = generate_video_summary(video)
    # Cache result
    summary_obj = models.VideoSummary(
        video_id=video_id,
        summary_type='brief',
        content=summary
    )
    db.session.add(summary_obj)
    db.session.commit()
    
    return jsonify({'summary': summary})
```

**Frontend Integration:**
```javascript
async function getSummary(videoId, videoTitle) {
    const response = await fetch(`/summarize/${videoId}`);
    const data = await response.json();
    // Display in modal with professional styling
}
```

### Feature 2: Smart Question Generator

**Business Value:** Helps users discover deeper insights and engage more meaningfully with content.

**AI Prompt Engineering:**
```python
prompt = f"""Based on this YouTube video transcript, generate 5 smart, engaging questions:

{video.transcript[:3000]}

Questions should be:
- Thought-provoking and specific to the content
- Help users understand key concepts better
- Encourage deeper exploration of the topic

Format as a simple list."""
```

**Revenue Model:** Premium users get unlimited smart questions, free users get 3 per day.

### Feature 3: Knowledge Base Insights

**Business Value:** Reveals patterns and connections across multiple videos that users wouldn't notice manually.

**Cross-Video Analysis:**
```python
def generate_knowledge_insights(videos):
    # Analyze up to 5 videos for token efficiency
    content_sample = "\n\n".join([
        f"Video: {video.title}\nContent: {video.transcript[:500]}..."
        for video in videos[:5]
    ])
    
    prompt = f"""Analyze this collection and provide:
    1. Common themes across videos
    2. Key topics that emerge
    3. Interesting connections between different videos
    4. Overall knowledge base characteristics"""
```

## 🛠 Complete Setup Guide

### Prerequisites

1. **Python 3.11+** installed
2. **Mistral AI API Key** (get from https://console.mistral.ai)
3. **Git** for version control

### Step 1: Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd youtube-transcript-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Environment Variables

Create a `.env` file:
```bash
MISTRAL_API_KEY=your_mistral_api_key_here
SESSION_SECRET=your_random_secret_key
DATABASE_URL=sqlite:///transcripts.db  # For development
```

### Step 3: Database Setup

```python
# The app automatically creates tables on first run
# For manual setup:
from app import app, db
with app.app_context():
    db.create_all()
```

### Step 4: Run the Application

```bash
# Development mode
python main.py

# Production mode with Gunicorn
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

## 📁 Project Structure

```
youtube-transcript-rag/
├── app.py                 # Main Flask application
├── main.py               # Entry point
├── models.py             # Database models
├── rag_service.py        # RAG implementation
├── transcript_service.py # YouTube transcript extraction
├── premium_features.py   # Premium feature implementations
├── static/
│   ├── css/
│   │   └── style.css     # Custom styling
│   └── js/
│       └── app.js        # Chat functionality
├── templates/
│   ├── index.html        # Main page with premium features
│   └── chat.html         # Chat interface
├── vercel.json           # Vercel deployment config
└── README.md             # This comprehensive guide
```

## 🔗 API Endpoints Reference

### Core Endpoints

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| GET | `/` | Main page | None |
| POST | `/upload` | Process YouTube URLs | `urls` (textarea) |
| GET | `/chat` | Chat interface | None |
| POST | `/chat/message` | Send chat message | `message` (JSON) |
| POST | `/delete_video/<id>` | Delete video | `video_id` (URL) |

### Premium Endpoints

| Method | Endpoint | Description | Revenue Feature |
|--------|----------|-------------|-----------------|
| GET | `/summarize/<id>` | AI video summary | ⭐ Premium |
| GET | `/smart_questions/<id>` | Smart questions | ⭐ Premium |
| GET | `/insights` | Knowledge insights | ⭐ Premium |

## 🎨 Frontend Architecture

### Bootstrap Integration
```html
<!-- Replit Dark Theme -->
<link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">

<!-- Font Awesome Icons -->
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
```

### Premium Feature UI Pattern
```javascript
// Consistent pattern for all premium features
async function premiumFeature(params) {
    // 1. Show loading modal
    const modal = new bootstrap.Modal(document.getElementById('premiumModal'));
    showLoadingState();
    modal.show();
    
    try {
        // 2. API call
        const response = await fetch(endpoint);
        const data = await response.json();
        
        // 3. Display results
        displayResults(data);
    } catch (error) {
        // 4. Error handling
        showError();
    }
}
```

## 🚀 Deployment Guide

### Vercel Deployment

1. **Install Vercel CLI:**
```bash
npm i -g vercel
```

2. **Configure vercel.json:**
```json
{
  "version": 2,
  "builds": [{"src": "main.py", "use": "@vercel/python"}],
  "routes": [{"src": "/(.*)", "dest": "main.py"}],
  "env": {"FLASK_ENV": "production"}
}
```

3. **Deploy:**
```bash
vercel --prod
```

4. **Set Environment Variables in Vercel Dashboard:**
- `MISTRAL_API_KEY`
- `SESSION_SECRET`
- `DATABASE_URL` (use PostgreSQL for production)

### Alternative Deployment Options

**Railway:**
```bash
railway login
railway new
railway add
railway deploy
```

**Heroku:**
```bash
heroku create your-app-name
heroku config:set MISTRAL_API_KEY=your_key
git push heroku main
```

## 💰 Monetization Strategy

### Freemium Model

**Free Tier:**
- Upload 3 videos
- Basic chat functionality
- 1 summary per day

**Premium Tier ($9.99/month):**
- Unlimited videos
- All premium features
- Advanced insights
- Priority support

### Implementation

```python
# Add to models.py
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscription_tier = db.Column(db.String(20), default='free')
    feature_usage = db.Column(db.JSON, default={})

# Usage tracking
def track_feature_usage(user_id, feature_name):
    user = User.query.get(user_id)
    if feature_name in user.feature_usage:
        user.feature_usage[feature_name] += 1
    else:
        user.feature_usage[feature_name] = 1
    db.session.commit()
```

## 🔧 Advanced Configuration

### Rate Limiting Configuration

```python
# In rag_service.py
class RAGService:
    def __init__(self):
        self.rate_limit_config = {
            'max_retries': 3,
            'base_delay': 1,
            'max_delay': 16,
            'backoff_factor': 2
        }
```

### Embedding Optimization

```python
# Chunk configuration for optimal performance
OPTIMAL_CHUNK_CONFIG = {
    'chunk_size': 1000,      # Characters per chunk
    'overlap': 200,          # Overlap for context
    'max_chunks_per_query': 3,  # Retrieval limit
    'similarity_threshold': 0.7  # Minimum similarity
}
```

### Database Optimization

```sql
-- Indexes for better performance
CREATE INDEX idx_video_processed ON video(processed);
CREATE INDEX idx_video_created_at ON video(created_at);
CREATE INDEX idx_summary_video_type ON video_summary(video_id, summary_type);
```

## 🧪 Testing Guide

### Manual Testing Checklist

**Core Features:**
- [ ] Upload single YouTube URL
- [ ] Upload multiple URLs (comma, space, newline separated)
- [ ] Chat with uploaded content
- [ ] Delete videos
- [ ] Rate limiting handles gracefully

**Premium Features:**
- [ ] AI Summary generates and caches
- [ ] Smart Questions are relevant and engaging
- [ ] Knowledge Insights reveal cross-video patterns

### API Testing

```bash
# Test summary endpoint
curl -X GET http://localhost:5000/summarize/1

# Test chat endpoint
curl -X POST http://localhost:5000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main points?"}'
```

## 🔒 Security Considerations

### API Key Protection
```python
# Never expose API keys in frontend
# Use environment variables only
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found")
```

### Input Validation
```python
# Validate YouTube URLs
def is_valid_youtube_url(url):
    patterns = [
        r'youtube\.com\/watch\?v=',
        r'youtu\.be\/',
        r'youtube\.com\/embed\/'
    ]
    return any(re.search(pattern, url) for pattern in patterns)
```

## 📊 Analytics & Monitoring

### Key Metrics to Track

**Business Metrics:**
- User registrations
- Premium conversions
- Feature usage frequency
- Retention rates

**Technical Metrics:**
- API response times
- Error rates
- Rate limit hits
- Database query performance

### Implementation

```python
import logging
import time

# Performance monitoring
def monitor_api_call(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} failed: {str(e)}")
            raise
    return wrapper
```

## 🔄 Maintenance & Updates

### Regular Tasks

**Daily:**
- Monitor error logs
- Check API rate limits
- Review user feedback

**Weekly:**
- Database maintenance
- Performance optimization
- Feature usage analysis

**Monthly:**
- Security updates
- Dependency updates
- Feature roadmap review

This comprehensive guide ensures anyone can understand, replicate, and enhance your YouTube Transcript RAG Chatbot with premium features!