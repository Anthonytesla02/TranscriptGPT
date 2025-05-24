class ChatApp {
    constructor() {
        this.chatForm = document.getElementById('chat-form');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.chatMessages = document.getElementById('chat-messages');
        
        this.init();
    }
    
    init() {
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSubmit(e);
            }
        });
        
        // Auto-focus input
        this.messageInput.focus();
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input and show loading
        this.messageInput.value = '';
        this.setLoading(true);
        
        // Show typing indicator
        const typingIndicator = this.addTypingIndicator();
        
        try {
            const response = await fetch('/chat/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            typingIndicator.remove();
            
            if (response.ok) {
                this.addMessage(data.response, 'assistant', data.sources);
            } else {
                this.addMessage(data.error || 'Sorry, something went wrong.', 'assistant');
            }
        } catch (error) {
            console.error('Chat error:', error);
            typingIndicator.remove();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(content, sender, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (sender === 'user') {
            messageContent.innerHTML = `<i class="fas fa-user me-2"></i>${this.escapeHtml(content)}`;
        } else {
            messageContent.innerHTML = `<i class="fas fa-robot me-2"></i>${this.escapeHtml(content)}`;
            
            // Add sources if provided
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                sourcesDiv.innerHTML = '<small class="text-muted"><i class="fas fa-book me-1"></i>Sources:</small>';
                
                sources.forEach(source => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    sourceItem.innerHTML = `
                        <div class="source-title">${this.escapeHtml(source.title)}</div>
                        <p class="source-text">${this.escapeHtml(source.text)}</p>
                    `;
                    sourcesDiv.appendChild(sourceItem);
                });
                
                messageContent.appendChild(sourcesDiv);
            }
        }
        
        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <i class="fas fa-robot me-2"></i>
            <span class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </span>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
        
        return typingDiv;
    }
    
    setLoading(loading) {
        if (loading) {
            this.sendBtn.classList.add('btn-loading');
            this.messageInput.disabled = true;
        } else {
            this.sendBtn.classList.remove('btn-loading');
            this.messageInput.disabled = false;
            this.messageInput.focus();
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chat app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
