// 全局变量
let currentUser = null;
let sessionId = null;

// DOM元素
const loginModal = document.getElementById('loginModal');
const loginForm = document.getElementById('loginForm');
const userNameInput = document.getElementById('userName');
const errorMessage = document.getElementById('errorMessage');
const userInfo = document.getElementById('userInfo');
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');

// 初始化应用
function initApp() {
    // 检查是否已有用户信息（通过sessionStorage模拟，实际应由后端处理）
    const savedUser = sessionStorage.getItem('currentUser');
    const savedSessionId = sessionStorage.getItem('sessionId');
    
    if (savedUser && savedSessionId) {
        currentUser = JSON.parse(savedUser);
        sessionId = savedSessionId;
        hideLoginModal();
        loadChatHistory();
    } else {
        showLoginModal();
    }
}

// 显示登录模态框
function showLoginModal() {
    loginModal.style.display = 'flex';
    messageInput.disabled = true;
    sendButton.disabled = true;
}

// 隐藏登录模态框
function hideLoginModal() {
    loginModal.style.display = 'none';
    messageInput.disabled = false;
    sendButton.disabled = false;
    updateUserInfo();
    messageInput.focus();
}

// 更新用户信息显示
function updateUserInfo() {
    userInfo.textContent = `用户: ${currentUser.name}`;
}

// 处理登录表单提交
async function handleLogin(event) {
    event.preventDefault();
    
    const userName = userNameInput.value.trim();
    if (!userName) {
        showError('请输入您的真实姓名');
        return;
    }

    try {
        // 发送用户信息到后端
        const response = await fetch('/api/user/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: userName })
        });

        if (!response.ok) {
            throw new Error('登录失败');
        }

        const data = await response.json();
        currentUser = { name: userName };
        sessionId = data.sessionId;

        // 保存到sessionStorage（实际应用中这些信息应由后端管理）
        sessionStorage.setItem('currentUser', JSON.stringify(currentUser));
        sessionStorage.setItem('sessionId', sessionId);

        hideLoginModal();
        loadChatHistory();
    } catch (error) {
        console.error('Login error:', error);
        showError('登录失败，请稍后重试');
    }
}

// 显示错误信息
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 3000);
}

// 加载聊天历史记录
async function loadChatHistory() {
    if (!sessionId) return;

    try {
        const response = await fetch(`/api/chat/history?sessionId=${sessionId}`);
        
        if (!response.ok) {
            throw new Error('获取聊天记录失败');
        }

        const messages = await response.json();
        
        // 清空聊天容器
        chatContainer.innerHTML = '';
        
        // 如果没有历史记录，显示欢迎消息
        if (messages.length === 0) {
            addMessage('你好！我是你的AI助手。有什么问题我可以帮你解答吗？', 'ai');
        } else {
            // 显示历史消息
            messages.forEach(msg => {
                addMessage(msg.content, msg.sender);
            });
        }
    } catch (error) {
        console.error('Load history error:', error);
        // 即使加载失败，也要显示欢迎消息
        addMessage('你好！我是你的AI助手。有什么问题我可以帮你解答吗？', 'ai');
    }
}

// 自动调整文本框高度
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// 发送消息
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || !currentUser || !sessionId) return;

    // 添加用户消息到聊天界面
    addMessage(message, 'user');
    
    // 清空输入框并重置高度
    messageInput.value = '';
    messageInput.style.height = 'auto';
    messageInput.focus();

    // 显示AI正在输入的指示器
    showTypingIndicator();

    try {
        // 调用后端API（流式响应），包含用户标识
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message,
                sessionId: sessionId,
                userName: currentUser.name
            })
        });

        if (!response.ok) {
            throw new Error('网络请求失败');
        }

        // 处理流式响应
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let aiMessageElement = null;
        let accumulatedText = '';

        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                // 移除输入指示器
                removeTypingIndicator();
                break;
            }

            const chunk = decoder.decode(value, { stream: true });
            accumulatedText += chunk;

            // 如果还没有创建AI消息元素，先创建一个
            if (!aiMessageElement) {
                aiMessageElement = addMessage('', 'ai');
            }

            // 更新AI消息内容
            aiMessageElement.querySelector('.message-content').textContent = accumulatedText;
            
            // 滚动到底部
            scrollToBottom();
        }
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessage('抱歉，处理您的请求时出现了错误。请稍后再试。', 'ai');
    }
}

// 添加消息到聊天界面
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = `avatar ${sender}`;
    avatar.textContent = sender === 'ai' ? 'AI' : currentUser?.name?.charAt(0) || '我';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);
    
    scrollToBottom();
    return messageDiv;
}

// 显示AI正在输入的指示器
function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message ai typing-indicator';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="avatar ai">AI</div>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatContainer.appendChild(indicator);
    scrollToBottom();
}

// 移除AI正在输入的指示器
function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// 滚动到底部
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 事件监听器
loginForm.addEventListener('submit', handleLogin);
sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 初始化应用
initApp();