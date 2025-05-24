import re
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

class TranscriptService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        # Handle different YouTube URL formats
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def extract_transcript(self, youtube_url):
        """Extract transcript from YouTube video"""
        try:
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                self.logger.error(f"Could not extract video ID from URL: {youtube_url}")
                return None
            
            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine transcript text
            transcript_text = ' '.join([item['text'] for item in transcript_list])
            
            # Clean up transcript text
            transcript_text = re.sub(r'\[.*?\]', '', transcript_text)  # Remove timestamps/annotations
            transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()  # Clean whitespace
            
            # Get video title (simple approach using video ID)
            title = f"YouTube Video {video_id}"
            
            return {
                'title': title,
                'transcript': transcript_text,
                'video_id': video_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract transcript from {youtube_url}: {str(e)}")
            return None
