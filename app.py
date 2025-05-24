import os
import logging
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

# configure the database, relative to the app instance folder
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///transcripts.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# initialize the app with the extension
db.init_app(app)

# Initialize services
transcript_service = TranscriptService()
rag_service = RAGService()

with app.app_context():
    # Import models
    import models
    db.create_all()

@app.route('/')
def index():
    """Main page for uploading YouTube videos"""
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
