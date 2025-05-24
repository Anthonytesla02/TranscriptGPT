import os
import json
import logging
import requests
from datetime import datetime

class PremiumFeatures:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        self.base_url = "https://api.mistral.ai/v1"
    
    def generate_video_summary(self, video, summary_type="brief", db=None, VideoSummary=None):
        """Generate AI-powered video summaries"""
        try:
            # Check if summary already exists
            if VideoSummary and db:
                existing = VideoSummary.query.filter_by(
                    video_id=video.id, 
                    summary_type=summary_type
                ).first()
                
                if existing:
                    return existing.content
            
            prompts = {
                "brief": f"""Create a concise 2-3 sentence summary of this YouTube video transcript:

{video.transcript[:3000]}

Focus on the main topic and key takeaway.""",
                
                "detailed": f"""Create a comprehensive summary of this YouTube video transcript in 3-4 paragraphs:

{video.transcript[:5000]}

Include:
- Main topic and purpose
- Key points discussed
- Important insights or conclusions
- Target audience""",
                
                "key_points": f"""Extract the 5-7 most important key points from this YouTube video transcript:

{video.transcript[:4000]}

Format as a bulleted list with brief explanations for each point."""
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompts[summary_type]}],
                "max_tokens": 800,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                summary_content = result["choices"][0]["message"]["content"]
                
                # Save summary to database if models provided
                if VideoSummary and db:
                    summary = VideoSummary(
                        video_id=video.id,
                        summary_type=summary_type,
                        content=summary_content
                    )
                    db.session.add(summary)
                    db.session.commit()
                
                return summary_content
            else:
                self.logger.error(f"Summary API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            return None
    
    def create_chat_session(self, session_name=None):
        """Create a new chat session for conversation history"""
        if not session_name:
            session_name = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(session_name=session_name)
        db.session.add(session)
        db.session.commit()
        return session
    
    def save_chat_message(self, session_id, message_type, content, sources=None):
        """Save chat message to persistent history"""
        sources_json = json.dumps(sources) if sources else None
        
        message = ChatMessage(
            session_id=session_id,
            message_type=message_type,
            content=content,
            sources=sources_json
        )
        db.session.add(message)
        db.session.commit()
        return message
    
    def get_chat_sessions(self):
        """Get all chat sessions ordered by most recent"""
        return ChatSession.query.order_by(ChatSession.updated_at.desc()).all()
    
    def get_session_messages(self, session_id):
        """Get all messages for a specific chat session"""
        messages = ChatMessage.query.filter_by(session_id=session_id)\
                                   .order_by(ChatMessage.created_at.asc()).all()
        
        formatted_messages = []
        for msg in messages:
            sources = json.loads(msg.sources) if msg.sources else None
            formatted_messages.append({
                'type': msg.message_type,
                'content': msg.content,
                'sources': sources,
                'timestamp': msg.created_at
            })
        return formatted_messages
    
    def generate_smart_questions(self, video):
        """Generate intelligent follow-up questions based on video content"""
        try:
            prompt = f"""Based on this YouTube video transcript, generate 5 smart, engaging questions that would help someone explore the content deeper:

{video.transcript[:3000]}

Questions should be:
- Thought-provoking and specific to the content
- Help users understand key concepts better
- Encourage deeper exploration of the topic
- Be suitable for someone who watched/read this content

Format as a numbered list."""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.4
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                questions = result["choices"][0]["message"]["content"]
                return questions.split('\n')
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to generate questions: {str(e)}")
            return []
    
    def generate_content_insights(self, videos):
        """Generate insights across multiple videos in the knowledge base"""
        try:
            if not videos:
                return None
            
            # Combine summaries or first parts of transcripts
            content_sample = "\n\n".join([
                f"Video: {video.title}\nContent: {video.transcript[:1000]}..."
                for video in videos[:5]  # Limit to 5 videos for token management
            ])
            
            prompt = f"""Analyze this collection of YouTube video content and provide insights:

{content_sample}

Please provide:
1. Common themes across the videos
2. Key topics that emerge
3. Interesting connections between different videos
4. Overall knowledge base characteristics
5. Suggested areas for further exploration

Keep the analysis concise but insightful."""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate insights: {str(e)}")
            return None