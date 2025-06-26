/**
 * ExpertORT Agent Chat UI - JavaScript Controller
 * Handles all chat functionality, API communication, and local storage
 */

class ExpertORTChat {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.conversation = [];
        this.isLoading = false;
        this.abortController = null;
        
        this.initializeElements();
        this.loadConversation();
        this.attachEventListeners();
        this.setupAutoResize();
        
        console.log('ExpertORT Chat initialized');
    }
    
    /**
     * Initialize DOM elements
     */
    initializeElements() {
        this.messagesArea = document.getElementById('messagesArea');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.clearChatBtn = document.getElementById('clearChatBtn');
        this.welcomeMessage = document.getElementById('welcomeMessage');
        this.loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        
        // Suggested questions
        this.suggestedQuestions = document.querySelectorAll('.suggested-question');
        
        // Initial send button state
        this.updateSendButton();
    }
    
    /**
     * Load conversation from localStorage
     */
    loadConversation() {
        try {
            const saved = localStorage.getItem('expertort_conversation');
            if (saved) {
                this.conversation = JSON.parse(saved);
                this.renderConversation();
            }
        } catch (error) {
            console.error('Error loading conversation:', error);
            this.conversation = [];
        }
    }
    
    /**
     * Save conversation to localStorage
     */
    saveConversation() {
        try {
            localStorage.setItem('expertort_conversation', JSON.stringify(this.conversation));
        } catch (error) {
            console.error('Error saving conversation:', error);
        }
    }
    
    /**
     * Clear conversation and restart
     */
    clearConversation() {
        this.conversation = [];
        this.saveConversation();
        this.messagesArea.innerHTML = '';
        this.showWelcomeMessage();
        this.messageInput.focus();
    }
    
    /**
     * Show welcome message
     */
    showWelcomeMessage() {
        if (this.welcomeMessage) {
            this.welcomeMessage.style.display = 'block';
        }
    }
    
    /**
     * Hide welcome message
     */
    hideWelcomeMessage() {
        if (this.welcomeMessage) {
            this.welcomeMessage.style.display = 'none';
        }
    }
    
    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Send button click
        this.sendBtn.addEventListener('click', (e) => {
            console.log('Send button clicked');
            e.preventDefault();
            this.sendMessage();
        });
        
        // Enter key to send message
        this.messageInput.addEventListener('keydown', (e) => {
            console.log('Key pressed:', e.key, 'Shift:', e.shiftKey);
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Attempting to send message...');
                this.sendMessage();
            }
        });
        
        // Input change to enable/disable send button
        this.messageInput.addEventListener('input', (e) => {
            console.log('Input changed:', e.target.value);
            this.updateSendButton();
        });
        
        // New chat button
        this.newChatBtn.addEventListener('click', () => {
            this.clearConversation();
        });
        
        // Clear chat button
        this.clearChatBtn.addEventListener('click', () => {
            if (confirm('¿Estás seguro de que quieres limpiar toda la conversación?')) {
                this.clearConversation();
            }
        });
        
        // Suggested questions
        this.suggestedQuestions.forEach(btn => {
            btn.addEventListener('click', () => {
                this.messageInput.value = btn.textContent.trim();
                this.updateSendButton();
                this.sendMessage();
            });
        });
        
        // Page visibility change - clear conversation when page is refreshed/closed
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'hidden') {
                // Optionally save state here
            }
        });
        
        // Before unload - clear conversation
        window.addEventListener('beforeunload', () => {
            // Clear conversation on page refresh/close
            localStorage.removeItem('expertort_conversation');
        });
    }
    
    /**
     * Setup auto-resize for textarea
     */
    setupAutoResize() {
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
    }
    
    /**
     * Update send button state
     */
    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const shouldEnable = hasText && !this.isLoading;
        
        if (this.sendBtn) {
            this.sendBtn.disabled = !shouldEnable;
            console.log('Send button state:', { hasText, isLoading: this.isLoading, disabled: this.sendBtn.disabled });
        }
    }
    
    /**
     * Send message to the agent
     */
    async sendMessage() {
        const message = this.messageInput.value.trim();
        console.log('sendMessage called with:', { message, isLoading: this.isLoading });
        
        if (!message || this.isLoading) {
            console.log('Message not sent - empty or loading');
            return;
        }
        
        // Hide welcome message if this is the first message
        if (this.conversation.length === 0) {
            this.hideWelcomeMessage();
        }
        
        // Add user message to conversation
        const userMessage = {
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        };
        
        this.conversation.push(userMessage);
        this.renderMessage(userMessage);
        this.saveConversation();
        
        // Clear input and disable send button
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.updateSendButton();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            this.isLoading = true;
            this.abortController = new AbortController();
            
            // Prepare API request
            const requestBody = {
                model: "expertort-agent",
                messages: this.conversation.map(msg => ({
                    role: msg.role,
                    content: msg.content
                })),
                stream: false
            };
            
            // Make API call
            const response = await fetch(`${this.apiBaseUrl}/v1/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
                signal: this.abortController.signal
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Extract agent response
            const agentContent = data.choices?.[0]?.message?.content || 'Lo siento, no pude procesar tu consulta.';
            
            const agentMessage = {
                role: 'assistant',
                content: agentContent,
                timestamp: new Date().toISOString()
            };
            
            this.conversation.push(agentMessage);
            this.hideTypingIndicator();
            this.renderMessage(agentMessage);
            this.saveConversation();
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            
            if (error.name !== 'AbortError') {
                this.renderErrorMessage('Error al comunicarse con el agente. Por favor, inténtalo de nuevo.');
            }
        } finally {
            this.isLoading = false;
            this.updateSendButton();
            this.abortController = null;
        }
    }
    
    /**
     * Render the entire conversation
     */
    renderConversation() {
        this.messagesArea.innerHTML = '';
        
        if (this.conversation.length === 0) {
            this.showWelcomeMessage();
        } else {
            this.hideWelcomeMessage();
            this.conversation.forEach(message => this.renderMessage(message));
        }
    }
    
    /**
     * Render a single message
     */
    renderMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role === 'user' ? 'user' : 'agent'}`;
        
        const time = new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        const avatar = message.role === 'user' 
            ? '<i class="fas fa-user"></i>'
            : '<i class="fas fa-robot"></i>';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                ${this.formatMessageContent(message.content)}
                <div class="message-time">${time}</div>
            </div>
        `;
        
        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    /**
     * Format message content (handle markdown, links, etc.)
     */
    formatMessageContent(content) {
        // Simple formatting for now
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
    }
    
    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        this.messagesArea.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    /**
     * Render error message
     */
    renderErrorMessage(errorText) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message agent error';
        
        const time = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        errorDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="message-content">
                ${errorText}
                <div class="message-time">${time}</div>
            </div>
        `;
        
        this.messagesArea.appendChild(errorDiv);
        this.scrollToBottom();
    }
    
    /**
     * Scroll to bottom of messages area
     */
    scrollToBottom() {
        setTimeout(() => {
            this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
        }, 100);
    }
    
    /**
     * Check if agent is online
     */
    async checkAgentStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            return response.ok;
        } catch (error) {
            console.error('Error checking agent status:', error);
            return false;
        }
    }
}

// Initialize the chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Clear conversation on page load (fresh start)
    localStorage.removeItem('expertort_conversation');
    
    // Initialize chat
    window.expertORTChat = new ExpertORTChat();
    
    // Focus on input
    document.getElementById('messageInput').focus();
    
    // Check agent status
    window.expertORTChat.checkAgentStatus().then(isOnline => {
        console.log('Agent status:', isOnline ? 'Online' : 'Offline');
    });
});

// Handle page unload - ensure conversation is cleared
window.addEventListener('unload', () => {
    localStorage.removeItem('expertort_conversation');
});

// Export for global access
window.ExpertORTChat = ExpertORTChat;
