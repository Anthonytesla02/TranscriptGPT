import os
import logging
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from transcript_service import TranscriptService
from rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# configure the database with Render.com PostgreSQL support
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url or "sqlite:///transcripts.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "pool_size": 20,
    "max_overflow": 40,
}

# initialize the app with the extension
db.init_app(app)

# Initialize services
transcript_service = TranscriptService()
rag_service = RAGService()

with app.app_context():
    # Import models
    import models
    
    # Create tables
    db.create_all()
    
    # Add new columns to existing video table if they don't exist
    try:
        with db.engine.connect() as conn:
            # Check if new columns exist, if not add them
            result = conn.execute(db.text("PRAGMA table_info(video)"))
            columns = [row[1] for row in result]
            
            if 'duration_seconds' not in columns:
                conn.execute(db.text("ALTER TABLE video ADD COLUMN duration_seconds INTEGER DEFAULT 0"))
            if 'view_count' not in columns:
                conn.execute(db.text("ALTER TABLE video ADD COLUMN view_count INTEGER DEFAULT 0"))
            if 'author' not in columns:
                conn.execute(db.text("ALTER TABLE video ADD COLUMN author VARCHAR(200) DEFAULT ''"))
            
            conn.commit()
    except Exception as e:
        logging.warning(f"Schema update warning: {e}")

@app.route('/')
def landing():
    """Landing page"""
    return render_template('landing.html')

@app.route('/dashboard')
def index():
    """Main dashboard for uploading YouTube videos"""
    videos = models.Video.query.filter_by(processed=True).all()
    return render_template('index.html', videos=videos)

@app.route('/upload', methods=['POST'])
def upload_videos():
    """Process YouTube video URLs"""
    urls = request.form.get('urls', '').strip()
    if not urls:
        flash('Please provide at least one YouTube URL', 'error')
        return redirect(url_for('index'))
    
    # Handle both single URLs and multiple URLs (separated by newlines, commas, or spaces)
    import re
    # Split by newlines, commas, or multiple spaces
    url_list = re.split(r'[\n,\s]+', urls)
    url_list = [url.strip() for url in url_list if url.strip()]
    
    if not url_list:
        flash('Please provide valid YouTube URLs', 'error')
        return redirect(url_for('index'))
    
    # Process each video
    processed_count = 0
    failed_urls = []
    
    for url in url_list:
        try:
            # Extract transcript
            video_info = transcript_service.extract_transcript(url)
            
            if video_info:
                # Save to database
                existing_video = models.Video.query.filter_by(youtube_url=url).first()
                if not existing_video:
                    video = models.Video(
                        title=video_info['title'],
                        youtube_url=url,
                        transcript=video_info['transcript'],
                        processed=False
                    )
                    db.session.add(video)
                    db.session.commit()
                    
                    # Process for RAG
                    rag_service.add_document(video.id, video_info['transcript'], video_info['title'])
                    
                    # Mark as processed
                    video.processed = True
                    db.session.commit()
                    
                    processed_count += 1
                else:
                    # Update existing video
                    existing_video.transcript = video_info['transcript']
                    existing_video.title = video_info['title']
                    if not existing_video.processed:
                        rag_service.add_document(existing_video.id, video_info['transcript'], video_info['title'])
                        existing_video.processed = True
                    db.session.commit()
                    processed_count += 1
            else:
                failed_urls.append(url)
                
        except Exception as e:
            logging.error(f"Failed to process {url}: {str(e)}")
            failed_urls.append(url)
    
    if processed_count > 0:
        flash(f'Successfully processed {processed_count} video(s)', 'success')
    
    if failed_urls:
        flash(f'Failed to process {len(failed_urls)} URL(s): {", ".join(failed_urls)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/chat')
def chat():
    """Chat interface"""
    videos = models.Video.query.filter_by(processed=True).all()
    if not videos:
        flash('Please upload and process some videos first', 'info')
        return redirect(url_for('index'))
    
    return render_template('chat.html', videos=videos)

@app.route('/chat/message', methods=['POST'])
def chat_message():
    """Handle chat messages"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    try:
        # Get response from RAG service
        response = rag_service.chat(message)
        return jsonify({
            'response': response['answer'],
            'sources': response['sources']
        })
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Sorry, I encountered an error processing your message'}), 500

@app.route('/delete_video/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    """Delete a video and its embeddings"""
    video = models.Video.query.get_or_404(video_id)
    
    try:
        # Remove from RAG service
        rag_service.remove_document(video_id)
        
        # Delete from database
        db.session.delete(video)
        db.session.commit()
        
        flash(f'Video "{video.title}" deleted successfully', 'success')
    except Exception as e:
        logging.error(f"Failed to delete video {video_id}: {str(e)}")
        flash('Failed to delete video', 'error')
    
    return redirect(url_for('index'))

# PREMIUM FEATURE 1: AI-Powered Video Summaries
@app.route('/summarize/<int:video_id>')
def summarize_video(video_id):
    """Generate AI summary of video content"""
    video = models.Video.query.get_or_404(video_id)
    
    # Check if summary already exists
    existing_summary = models.VideoSummary.query.filter_by(
        video_id=video_id, summary_type='brief'
    ).first()
    
    if existing_summary:
        summary = existing_summary.content
    else:
        # Generate new summary
        summary = generate_video_summary(video)
        if summary:
            # Save to database
            summary_obj = models.VideoSummary(
                video_id=video_id,
                summary_type='brief',
                content=summary
            )
            db.session.add(summary_obj)
            db.session.commit()
    
    return jsonify({'summary': summary})

# PREMIUM FEATURE 2: Smart Question Suggestions
@app.route('/smart_questions/<int:video_id>')
def smart_questions(video_id):
    """Generate intelligent questions about video content"""
    video = models.Video.query.get_or_404(video_id)
    questions = generate_smart_questions(video)
    return jsonify({'questions': questions})

# PREMIUM FEATURE 3: Knowledge Base Insights
@app.route('/insights')
def knowledge_insights():
    """Generate insights across all videos in knowledge base"""
    videos = models.Video.query.filter_by(processed=True).all()
    if not videos:
        return jsonify({'insights': 'No videos in knowledge base yet.'})
    
    insights = generate_knowledge_insights(videos)
    return jsonify({'insights': insights})

def generate_video_summary(video):
    """Generate AI-powered video summary"""
    try:
        prompt = f"""Create a concise 2-3 sentence summary of this YouTube video transcript:

{video.transcript[:3000]}

Focus on the main topic and key takeaway."""

        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return "Could not generate summary at this time."
            
    except Exception as e:
        logging.error(f"Summary generation error: {str(e)}")
        return "Error generating summary."

def generate_smart_questions(video):
    """Generate intelligent questions about video content"""
    try:
        prompt = f"""Based on this YouTube video transcript, generate 5 smart, engaging questions that would help someone explore the content deeper:

{video.transcript[:3000]}

Questions should be:
- Thought-provoking and specific to the content
- Help users understand key concepts better
- Encourage deeper exploration of the topic

Format as a simple list."""

        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.4
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            questions_text = result["choices"][0]["message"]["content"]
            # Split into individual questions
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.strip().isdigit()]
            return questions[:5]  # Return max 5 questions
        else:
            return ["What are the main points discussed in this video?"]
            
    except Exception as e:
        logging.error(f"Questions generation error: {str(e)}")
        return ["What are the key insights from this video?"]

def generate_knowledge_insights(videos):
    """Generate insights across multiple videos"""
    try:
        # Combine video titles and brief content samples
        content_sample = "\n\n".join([
            f"Video: {video.title}\nContent: {video.transcript[:500]}..."
            for video in videos[:5]  # Limit to 5 videos for token management
        ])
        
        prompt = f"""Analyze this collection of YouTube video content and provide insights:

{content_sample}

Please provide:
1. Common themes across the videos
2. Key topics that emerge
3. Interesting connections between different videos
4. Overall knowledge base characteristics

Keep the analysis concise but insightful."""

        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return "Unable to generate insights at this time."
            
    except Exception as e:
        logging.error(f"Insights generation error: {str(e)}")
        return "Error analyzing knowledge base."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
