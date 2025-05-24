from app import db
from datetime import datetime

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    youtube_url = db.Column(db.String(500), unique=True, nullable=False)
    transcript = db.Column(db.Text, nullable=False)
    processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Premium features
    duration_seconds = db.Column(db.Integer, default=0)
    view_count = db.Column(db.Integer, default=0)
    author = db.Column(db.String(200), default="")
    
    def __repr__(self):
        return f'<Video {self.title}>'

class VideoSummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    summary_type = db.Column(db.String(50), nullable=False)  # 'brief', 'detailed', 'key_points'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    video = db.relationship('Video', backref=db.backref('summaries', lazy=True))

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    message_type = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text)  # JSON string of sources
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    session = db.relationship('ChatSession', backref=db.backref('messages', lazy=True))
