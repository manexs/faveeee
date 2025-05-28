document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const userDropdown = document.getElementById('user-dropdown');
    const switchUserBtn = document.getElementById('switch-user-btn');
    const userIndicator = document.getElementById('user-indicator');
    
    // Keep track of conversation history
    let conversationHistory = [];
    let isWaitingForResponse = false;
    let currentFilename = 'latest_clustered_data.csv'; // Default filename
    let currentUserId = null;
    let activeEventMessageId = null; // Track which event invitation we're responding to
    let sessionId = generateSessionId(); // Generate a unique session ID

    // Load users for dropdown
    loadUsers();
    
    // Load current user context
    loadUserContext();
    
    // Add reset database button to the UI
    addResetDatabaseButton();
    
    // Listen for the database-reset event from other pages
    document.addEventListener('database-reset', function() {
        console.log('Database reset event detected');
        handleDatabaseReset();
    });

    // User switching - UPDATED TO AVOID PAGE RELOAD
    switchUserBtn.addEventListener('click', function() {
        const selectedUserId = userDropdown.value;
        if (!selectedUserId) {
            alert('Please select a user first');
            return;
        }
        
        // Set user context
        fetch('/set_user_context', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: selectedUserId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // CHANGE: Update UI without page reload
                currentUserId = parseInt(selectedUserId);
                
                // Update user indicator
                userIndicator.textContent = `Chatting as User #${currentUserId}`;
                
                // Clear chat history
                conversationHistory = [];
                
                // Clear previous messages
                chatMessages.innerHTML = '';
                
                // Add welcome message
                addMessageToChat('assistant', `Hello! I'm the Social Connection Platform assistant. How can I help you, User #${currentUserId}?`);
                
                // Reset active event message ID
                activeEventMessageId = null;
                
                // Generate new session ID
                sessionId = generateSessionId();
                
                // Load messages for the new user
                loadMessages();
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error setting user context:', error);
            alert('Failed to switch user. Please try again.');
        });
    });

    // Auto-resize the textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Enable/disable send button based on input
        if (this.value.trim() && !isWaitingForResponse) {
            sendButton.disabled = false;
        } else {
            sendButton.disabled = true;
        }
    });

    // Send message when Enter key is pressed (without Shift)
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && !isWaitingForResponse) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Send message when Send button is clicked
    sendButton.addEventListener('click', function() {
        if (!isWaitingForResponse) {
            sendMessage();
        }
    });

    function loadUsers() {
        // Find an available dataset file
        fetch('/get_available_file')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.filename) {
                    // Load users from dataset
                    fetch(`/get_users?filename=${data.filename}`)
                        .then(response => response.json())
                        .then(userData => {
                            if (userData.success && userData.users) {
                                // Populate dropdown
                                userDropdown.innerHTML = '<option value="">Select a user...</option>';
                                
                                userData.users.forEach(user => {
                                    const option = document.createElement('option');
                                    option.value = user.user_id;
                                    option.textContent = `User #${user.user_id} (${user.age}, ${user.location})`;
                                    userDropdown.appendChild(option);
                                });
                                
                                // Select current user if set
                                if (currentUserId) {
                                    userDropdown.value = currentUserId;
                                }
                            }
                        })
                        .catch(error => {
                            console.error('Error loading users:', error);
                            addSystemMessage('Failed to load users. Please try refreshing the page.');
                        });
                }
            })
            .catch(error => console.error('Error finding dataset file:', error));
    }

    function loadUserContext() {
        fetch('/get_user_context')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.user_id) {
                    currentUserId = data.user_id;
                    
                    // Update user indicator
                    userIndicator.textContent = `Chatting as User #${data.user_id}`;
                    
                    // Select the user in dropdown
                    userDropdown.value = data.user_id;
                    
                    // Load any messages for this user
                    loadMessages();
                } else {
                    // ADDED: No user selected message
                    console.log('No user context found');
                    userIndicator.textContent = 'No user selected';
                    addSystemMessage('Please select a user to start chatting.');
                }
            })
            .catch(error => {
                console.error('Error loading user context:', error);
                addSystemMessage('Failed to load user context. Please try refreshing the page.');
            });
    }
    
    function loadMessages() {
        if (!currentUserId) {
            console.log('Cannot load messages: No user selected');
            return;
        }
        
        console.log(`Loading messages for user ${currentUserId}`);
        fetch('/api/get_messages')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Clear existing welcome message if this is the first load
                    if (chatMessages.childElementCount <= 1) {
                        chatMessages.innerHTML = '';
                    }
                    
                    // Add a system message about user login
                    if (!chatMessages.querySelector('.user-login-message')) {
                        const loginMessage = `You are now logged in as User #${currentUserId}.`;
                        
                        if (data.details) {
                            loginMessage += ` Your interests include: ${data.details.interests.join(', ')}.`;
                        }
                        
                        addSystemMessage(loginMessage, 'user-login-message');
                    }
                    
                    // Display received messages in chat
                    if (data.received && data.received.length > 0) {
                        // Add a separator if not already present
                        if (!chatMessages.querySelector('.messages-separator')) {
                            addSystemMessage('--- Your Messages ---', 'messages-separator');
                        }
                        
                        // Add each message to the chat
                        data.received.forEach(msg => {
                            // Only add if we haven't already displayed this message
                            const msgId = msg.id || msg._id; // Support both MongoDB and SQLite IDs
                            if (!document.querySelector(`.message-id-${msgId}`)) {
                                const messageContent = `
                                    <div class="message-header">From User #${msg.from_user_id} - ${msg.timestamp}</div>
                                    <div class="message-content">${msg.content}</div>
                                `;
                                
                                addMessageToChat('assistant', messageContent, `message-id-${msgId}`);
                                
                                // If this is an event invitation and hasn't been responded to,
                                // mark it as the active event message for response
                                if (!msg.response && msg.message_type === 'event_invitation') {
                                    activeEventMessageId = msgId;
				    console.log('Setting active event message ID:', msgId);
                                } else if (msg.response) {
                                    // Show the response that was already given
                                    addSystemMessage(`You responded: ${msg.response}`);
                                }
                            }
                        });
                    }
                    
                    // Display sent messages in chat (events created)
                    if (data.sent && data.sent.length > 0) {
                        // Check for event invitations
                        const eventInvitations = data.sent.filter(msg => msg.message_type === 'event_invitation');
                        
                        if (eventInvitations.length > 0) {
                            // Add a separator if not already present
                            if (!chatMessages.querySelector('.events-separator')) {
                                addSystemMessage('--- Events You Created ---', 'events-separator');
                            }
                            
                            // Group by event_id
                            const eventGroups = {};
                            eventInvitations.forEach(msg => {
                                if (!eventGroups[msg.event_id]) {
                                    eventGroups[msg.event_id] = [];
                                }
                                eventGroups[msg.event_id].push(msg);
                            });
                            
                            // Display each event group
                            Object.keys(eventGroups).forEach(eventId => {
                                const eventMessages = eventGroups[eventId];
                                // Handle both string and object event details
                                let eventDetails;
                                if (typeof eventMessages[0].event_details === 'string') {
                                    try {
                                        eventDetails = JSON.parse(eventMessages[0].event_details);
                                    } catch(e) {
                                        console.error('Error parsing event details:', e);
                                        eventDetails = {title: 'Event', date: 'Unknown'};
                                    }
                                } else {
                                    eventDetails = eventMessages[0].event_details;
                                }
                                
                                // Only add if we haven't already displayed this event
                                if (!document.querySelector(`.event-id-${eventId}`)) {
                                    const messageContent = `
                                        <div class="message-header">Event: ${eventDetails.title} on ${eventDetails.date}</div>
                                        <div class="message-content">
                                            <p>You created this event and invited ${eventMessages.length} users.</p>
                                            <p>Responses:</p>
                                            <ul class="response-list-${eventId}">
                                                ${eventMessages.map(msg => `
                                                    <li>User #${msg.to_user_id}: ${msg.response || 'No response yet'}</li>
                                                `).join('')}
                                            </ul>
                                        </div>
                                    `;
                                    
                                    addMessageToChat('assistant', messageContent, `event-id-${eventId}`);
                                }
                            });
                        }
                    }
                } else if (data.message === 'No user logged in') {
                    // Handle gracefully when no user is logged in
                    console.log('No user logged in when trying to load messages');
                    // Don't display an error, as this is expected when no user is selected
                } else {
                    console.error('Error loading messages:', data.message);
                    addSystemMessage('Failed to load messages: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error loading messages:', error);
                addSystemMessage('Failed to load messages. Please try again later.');
            });
    }
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) {
            console.log('Prevented sending empty message');
            return;  // Exit early if message is empty
        }
        
        // Check if user is selected
        if (!currentUserId) {
            alert('Please select a user before sending messages.');
            return;
        }
        
        // Disable UI during processing
        isWaitingForResponse = true;
        sendButton.disabled = true;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input and reset height
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Add to conversation history
        conversationHistory.push({
            role: 'user',
            content: message
        });
        
        // UPDATED: Use both 'user_message' and 'query' for backward compatibility
        // Prepare request payload - including event context if applicable
        const payload = {
            user_message: message,  // ADDED: Use the key backend expects
            query: message,         // KEEP: For backward compatibility
            filename: currentFilename,
            user_id: currentUserId,
            session_id: sessionId,
            conversation_history: conversationHistory
        };
        
        // Include event context if this message is responding to an event
        if (activeEventMessageId) {
            payload.event_message_id = activeEventMessageId;
            payload.event_response = true;
            
            // Add to the log that we're sending event context
            console.log('Message is responding to event ID:', activeEventMessageId);
            console.log('Full payload being sent:', payload);
        }
        
        console.log('Sending message with payload:', payload);
        
        // Send request to the backend (uses a single endpoint for all messages)
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Get the response text from the API response
            const responseText = data.answer || data.message || data.response || 'Sorry, I could not process your request.';
            
            // Add assistant message to chat
            addMessageToChat('assistant', responseText);
            
            // Add to conversation history
            conversationHistory.push({
                role: 'assistant',
                content: responseText
            });
            
            // If this was an event response, clear the active event ID
            if (activeEventMessageId && payload.event_response) {
                // Clear the active event message ID now that it's been processed
                console.log('Event response processed for event ID:', activeEventMessageId);
                activeEventMessageId = null;
                
                // Refresh messages to update any changes
                setTimeout(loadMessages, 1000);
            }
            
            // Re-enable UI
            isWaitingForResponse = false;
            if (userInput.value.trim()) {
                sendButton.disabled = false;
            }
            
            // Scroll to bottom
            scrollToBottom();
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Handle error with context instead of showing generic message
            handleErrorWithContext(message, error.toString());
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Re-enable UI
            isWaitingForResponse = false;
            if (userInput.value.trim()) {
                sendButton.disabled = false;
            }
        });
    }
    
    function handleErrorWithContext(userMessage, errorMessage) {
        console.log('Handling error with context:', errorMessage);
        
        // IMPROVED: Better fallback handling if error endpoint fails
        // First try to use the error API
        const errorContext = {
            user_message: userMessage,
            error_details: errorMessage,
            request_type: activeEventMessageId ? 'event_response' : 'general_chat',
            event_id: activeEventMessageId
        };
        
        // Send to backend endpoint that will use LLM to generate friendly error message
        fetch('/api/generate_error_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(errorContext)
        })
        .then(response => {
            // Check response status before trying to parse as JSON
            if (!response.ok) {
                throw new Error(`Error API returned status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.message) {
                // Use the LLM-generated error message
                addMessageToChat('assistant', data.message);
                
                // Add to conversation history
                conversationHistory.push({
                    role: 'assistant',
                    content: data.message
                });
            } else {
                throw new Error('No valid message in error API response');
            }
        })
        .catch(err => {
            // In case the error handler endpoint fails or returns invalid data
            console.error('Error in error handler:', err);
            let fallbackMessage;
            
            // Check if the error mentions "API" or "message required"
            if (errorMessage.includes('message') && errorMessage.includes('required')) {
                fallbackMessage = "I'm having trouble understanding your request. Could you please try again with a more detailed message?";
            } else if (errorMessage.includes('Network') || errorMessage.includes('Failed to fetch')) {
                fallbackMessage = "I'm having trouble connecting to the server. Please check your internet connection and try again.";
            } else {
                fallbackMessage = "I'm sorry, I encountered an issue processing your request. Could you please try again or rephrase your message?";
            }
            
            addMessageToChat('assistant', fallbackMessage);
            
            // Still add to conversation history
            conversationHistory.push({
                role: 'assistant',
                content: fallbackMessage
            });
        });
    }

    function addMessageToChat(role, content, extraClass = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role} ${extraClass}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.textContent = role === 'user' ? 'U' : 'T';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Check if content has HTML
        if (content.includes('<div') || content.includes('<p')) {
            contentDiv.innerHTML = content;
        } else {
            // Convert markdown to HTML
            const formattedContent = formatMarkdown(content);
            contentDiv.innerHTML = formattedContent;
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom
        scrollToBottom();
    }
    
    function addSystemMessage(message, extraClass = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `system-message ${extraClass}`;
        messageDiv.textContent = message;
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom
        scrollToBottom();
    }
    
    function formatMarkdown(text) {
        // This is a simple implementation. For production, use a proper markdown library
        let formattedText = text;
        
        // Convert code blocks (```code```)
        formattedText = formattedText.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Convert inline code (`code`)
        formattedText = formattedText.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Convert headers (# Header)
        formattedText = formattedText.replace(/^# (.*$)/gm, '<h1>$1</h1>');
        formattedText = formattedText.replace(/^## (.*$)/gm, '<h2>$1</h2>');
        formattedText = formattedText.replace(/^### (.*$)/gm, '<h3>$1</h3>');
        
        // Convert bold (**text**)
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert italic (*text*)
        formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Convert links ([text](url))
        formattedText = formattedText.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');
        
        // Convert paragraphs (line breaks)
        formattedText = formattedText.replace(/\n\n/g, '</p><p>');
        
        // Wrap in paragraphs
        formattedText = '<p>' + formattedText + '</p>';
        
        return formattedText;
    }
    
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing';
        typingDiv.id = 'typing-indicator';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.textContent = 'T';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            indicatorDiv.appendChild(dot);
        }
        
        contentDiv.appendChild(indicatorDiv);
        typingDiv.appendChild(avatarDiv);
        typingDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(typingDiv);
        
        scrollToBottom();
    }
    
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Generate a unique session ID
    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    // Function to add reset button to chat interface
    function addResetDatabaseButton() {
        // Create button container
        const resetButtonContainer = document.createElement('div');
        resetButtonContainer.className = 'reset-button-container';
        
        // Create button
        const resetButton = document.createElement('button');
        resetButton.id = 'chatResetDbBtn';
        resetButton.className = 'btn danger small';
        resetButton.innerHTML = '<i class="fas fa-trash-alt"></i> Reset Database';
        
        // Add to container
        resetButtonContainer.appendChild(resetButton);
        
        // Find a good place to add the button (e.g., near the user dropdown)
        const userControls = document.querySelector('.user-controls');
        if (userControls) {
            userControls.appendChild(resetButtonContainer);
        } else {
            // Create user controls if it doesn't exist
            const userControlsDiv = document.createElement('div');
            userControlsDiv.className = 'user-controls';
            userControlsDiv.appendChild(resetButtonContainer);
            
            // Insert user controls before the chat messages
            if (chatMessages && chatMessages.parentNode) {
                chatMessages.parentNode.insertBefore(userControlsDiv, chatMessages);
            }
        }
        
        // Create modal dialog
        const modalHTML = `
            <div class="modal-overlay" id="chatResetConfirmModal">
                <div class="modal-container">
                    <div class="modal-header">
                        <h3>Confirm Database Reset</h3>
                        <button class="modal-close" id="chatCloseModalBtn">&times;</button>
                    </div>
                    <div class="modal-body">
                        <p>This will delete all saved events, messages, and invitations. This action cannot be undone.</p>
                        <p>Are you sure you want to reset the database?</p>
                    </div>
                    <div class="modal-footer">
                        <button class="btn secondary" id="chatCancelResetBtn">Cancel</button>
                        <button class="btn danger" id="chatConfirmResetBtn">Reset Database</button>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to document
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Add event listeners
        const resetConfirmModal = document.getElementById('chatResetConfirmModal');
        const confirmResetBtn = document.getElementById('chatConfirmResetBtn');
        const cancelResetBtn = document.getElementById('chatCancelResetBtn');
        const closeModalBtn = document.getElementById('chatCloseModalBtn');
        
        resetButton.addEventListener('click', function() {
            resetConfirmModal.style.display = 'flex';
        });
        
        confirmResetBtn.addEventListener('click', function() {
            resetDatabase();
        });
        
        // Close modal handlers
        cancelResetBtn.addEventListener('click', function() {
            resetConfirmModal.style.display = 'none';
        });
        
        closeModalBtn.addEventListener('click', function() {
            resetConfirmModal.style.display = 'none';
        });
        
        // Close modal when clicking outside
        window.addEventListener('click', function(event) {
            if (event.target === resetConfirmModal) {
                resetConfirmModal.style.display = 'none';
            }
        });
        
        // Add necessary styles
        const styles = `
            .reset-button-container {
                margin-left: 10px;
            }
            
            .btn.danger {
                background-color: #dc3545;
                color: white;
            }

            .btn.danger:hover {
                background-color: #c82333;
            }
            
            .btn.danger.small {
                padding: 0.25rem 0.5rem;
                font-size: 0.875rem;
            }
            
            .modal-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 1000;
                justify-content: center;
                align-items: center;
            }
            
            .modal-container {
                background-color: white;
                border-radius: 8px;
                width: 90%;
                max-width: 500px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .modal-header {
                padding: 1rem;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .modal-body {
                padding: 1rem;
            }
            
            .modal-footer {
                padding: 1rem;
                border-top: 1px solid #dee2e6;
                display: flex;
                justify-content: flex-end;
                gap: 0.5rem;
            }
            
            .modal-close {
                background: none;
                border: none;
                font-size: 1.5rem;
                cursor: pointer;
                color: #6c757d;
            }
            
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                background-color: #28a745;
                color: white;
                border-radius: 4px;
                z-index: 1001;
                opacity: 1;
                transition: opacity 0.5s;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            }
            
            .notification.success {
                background-color: #28a745;
            }
            
            .notification.error {
                background-color: #dc3545;
            }
            
            .btn.secondary {
                background-color: #6c757d;
                color: white;
            }
            
            .btn.secondary:hover {
                background-color: #5a6268;
            }
            
            .system-message {
                padding: 10px;
                margin: 10px 0;
                background-color: #f8f9fa;
                border-left: 3px solid #10a37f;
                border-radius: 3px;
                font-style: italic;
                color: #666;
            }
            
            .alert {
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 15px;
            }
            
            .alert.success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .alert.error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                margin: 10px auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        
        // Add styles to head
        const styleElement = document.createElement('style');
        styleElement.textContent = styles;
        document.head.appendChild(styleElement);
    }

    // Function to reset the database
    function resetDatabase() {
        const resetConfirmModal = document.getElementById('chatResetConfirmModal');
        const modalBody = resetConfirmModal.querySelector('.modal-body');
        const originalContent = modalBody.innerHTML;
        
        // Show loading spinner
        modalBody.innerHTML = '<div class="spinner"></div><p>Resetting database...</p>';
        
        // Disable buttons
        const confirmBtn = document.getElementById('chatConfirmResetBtn');
        const cancelBtn = document.getElementById('chatCancelResetBtn');
        confirmBtn.disabled = true;
        cancelBtn.disabled = true;
        
        // Call the reset API with the full path to ensure it works
        fetch('/api/reset_all', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // Empty body as JSON to ensure proper content type
            body: JSON.stringify({})
        })
        .then(response => {
            // Check if response is OK before trying to parse JSON
            if (!response.ok) {
                return response.text().then(text => {
                    console.error(`Server error (${response.status}):`, text);
                    throw new Error(`Server responded with status ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            // Restore original content
            modalBody.innerHTML = originalContent;
            
            if (data.success) {
                // Change modal content to success message
                modalBody.innerHTML = '<div class="alert success"><p>Database has been reset successfully!</p></div>';
                
                // Change footer buttons
                resetConfirmModal.querySelector('.modal-footer').innerHTML = 
                    '<button class="btn primary" id="chatCloseSuccessBtn">Close</button>';
                
                // Add event listener to new close button
                document.getElementById('chatCloseSuccessBtn').addEventListener('click', function() {
                    resetConfirmModal.style.display = 'none';
                    
                    // Handle the reset in the UI
                    handleDatabaseReset();
                });
                
                // Dispatch reset event
                const resetEvent = new CustomEvent('database-reset');
                document.dispatchEvent(resetEvent);
            } else {
                // Show error message
                modalBody.innerHTML = `<div class="alert error"><p>Error: ${data.message || 'Unknown error'}</p></div>`;
                
                // Re-enable buttons
                confirmBtn.disabled = false;
                cancelBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error resetting database:', error);
            modalBody.innerHTML = `<div class="alert error"><p>An error occurred: ${error.message}</p></div>`;
            
            // Re-enable buttons
            confirmBtn.disabled = false;
            cancelBtn.disabled = false;
        });
    }

    // Function to handle UI updates after database reset
    function handleDatabaseReset() {
        const chatMessages = document.getElementById('chat-messages');
        
        // Clear all messages except system messages
        if (chatMessages) {
            // Keep only the initial welcome message
            const welcomeMessage = chatMessages.querySelector('.message.assistant:first-child');
            chatMessages.innerHTML = '';
            
            if (welcomeMessage) {
                chatMessages.appendChild(welcomeMessage);
            }
            
            // Add a system message about the reset
            const systemMessage = document.createElement('div');
            systemMessage.className = 'system-message';
            systemMessage.textContent = 'Database has been reset. All events and invitations have been cleared. Please select a new user to continue.';
            chatMessages.appendChild(systemMessage);
            
            // If we had active event message ID, clear it
            activeEventMessageId = null;
        }
        
        // Clear the current user indicator and reset user ID
        currentUserId = null;
        if (userIndicator) {
            userIndicator.textContent = 'No user selected';
        }
        
        // Disable send button until a new user is selected
        if (sendButton) {
            sendButton.disabled = true;
        }
        
        // Reload the users list to ensure dropdown is populated
        loadUsers();
        
        // Reset session ID to start fresh
        sessionId = generateSessionId();
        conversationHistory = [];
        
        // Optionally, refresh messages from server to confirm they're gone
        if (typeof loadMessages === 'function') {
            setTimeout(loadMessages, 1000);
        }
        
        // Show a notification
        const notification = document.createElement('div');
        notification.className = 'notification success';
        notification.textContent = 'Database reset successful!';
        document.body.appendChild(notification);
        
        // Auto-remove notification after 3 seconds
        setTimeout(function() {
            notification.style.opacity = '0';
            setTimeout(function() {
                notification.remove();
            }, 500);
        }, 3000);
    }
    
    // Initial welcome message
    addMessageToChat('assistant', 'Hello! I\'m the Social Connection Platform assistant. How can I help you find connections and events today?');
});