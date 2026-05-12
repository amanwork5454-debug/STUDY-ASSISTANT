css = """
<style>
.chat-message {
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.chat-message.user {
    background-color: #e8f4fd;
    flex-direction: row-reverse;
}
.chat-message.bot {
    background-color: #f0fdf4;
}
.chat-message .avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.chat-message.user .avatar { background: #2563eb; }
.chat-message.bot  .avatar { background: #16a34a; }
.chat-message .message {
    flex: 1;
    font-size: 15px;
    line-height: 1.6;
    color: #1e293b;
}
</style>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">🧑</div>
    <div class="message">{{MSG}}</div>
</div>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
"""
