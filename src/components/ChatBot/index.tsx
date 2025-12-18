import React, { useState, useEffect, useRef, useCallback } from 'react';
import styles from './styles.module.css';

// Types
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: { source: string; score: number }[];
  contextUsed?: string;
}

// Backend URL
const API_URL = 'http://localhost:8000';

// SVG Icons
const SparkleIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 0L14.59 9.41L24 12L14.59 14.59L12 24L9.41 14.59L0 12L9.41 9.41L12 0Z"/>
  </svg>
);

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2Z"/>
  </svg>
);

const CloseIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M18 6L6 18M6 6L18 18"/>
  </svg>
);

const TrashIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 6H21M8 6V4C8 3.44772 8.44772 3 9 3H15C15.5523 3 16 3.44772 16 4V6M19 6V20C19 20.5523 18.5523 21 18 21H6C5.44772 21 5 20.5523 5 20V6H19Z"/>
  </svg>
);

const BotIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="11" width="18" height="10" rx="2"/>
    <circle cx="12" cy="5" r="2"/>
    <path d="M12 7V11"/>
    <circle cx="8" cy="16" r="1" fill="currentColor"/>
    <circle cx="16" cy="16" r="1" fill="currentColor"/>
  </svg>
);

// Simple markdown renderer
const renderMarkdown = (text: string): React.ReactNode[] => {
  return text.split(/(```[\s\S]*?```|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g).map((part, i) => {
    if (part.startsWith('```') && part.endsWith('```')) {
      const code = part.slice(3, -3).replace(/^\w+\n/, '');
      return <pre key={i} className={styles.codeBlock}><code>{code}</code></pre>;
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={i} className={styles.inlineCode}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    return part.split('\n').map((line, j, arr) => (
      <React.Fragment key={`${i}-${j}`}>{line}{j < arr.length - 1 && <br/>}</React.Fragment>
    ));
  });
};

export default function ChatBot(): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatWindowRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen]);

  // Listen for text selection (mouseup event)
  useEffect(() => {
    const handleMouseUp = () => {
      // Skip if selection is inside chat window
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed) return;

      const anchorNode = selection.anchorNode;
      if (chatWindowRef.current?.contains(anchorNode as Node)) return;

      const text = selection.toString().trim();
      if (text.length >= 20 && text.length <= 4000) {
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    return () => document.removeEventListener('mouseup', handleMouseUp);
  }, []);

  const generateId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

  const sendMessage = useCallback(async () => {
    const query = input.trim();
    if (!query || isLoading) return;

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const body: { query: string; context_text?: string } = { query };

      // Attach selected text as context
      if (selectedText) {
        body.context_text = selectedText;
      }

      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        sources: data.sources,
        contextUsed: data.context_used,
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Clear context after use
      if (selectedText) {
        setSelectedText(null);
      }

    } catch (error) {
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Sorry, I couldn't connect to the server. Please ensure the backend is running at ${API_URL}.\n\nError: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, selectedText]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSelectedText(null);
  };

  const clearContext = () => {
    setSelectedText(null);
  };

  return (
    <>
      {/* Floating Action Button */}
      <button
        className={`${styles.fab} ${isOpen ? styles.fabHidden : ''}`}
        onClick={() => setIsOpen(true)}
        aria-label="Open AI Assistant"
      >
        <SparkleIcon />
        <span className={styles.fabLabel}>Ask AI</span>
        {selectedText && <span className={styles.fabBadge} />}
      </button>

      {/* Chat Window */}
      <div
        ref={chatWindowRef}
        className={`${styles.chatWindow} ${isOpen ? styles.chatOpen : ''}`}
      >
        {/* Header */}
        <header className={styles.header}>
          <div className={styles.headerInfo}>
            <BotIcon />
            <div>
              <h3 className={styles.headerTitle}>Physical AI Assistant</h3>
              <span className={styles.headerStatus}>
                <span className={styles.statusDot} />
                Powered by Gemini
              </span>
            </div>
          </div>
          <div className={styles.headerActions}>
            <button onClick={clearChat} title="Clear chat" className={styles.headerBtn}>
              <TrashIcon />
            </button>
            <button onClick={() => setIsOpen(false)} title="Close" className={styles.headerBtn}>
              <CloseIcon />
            </button>
          </div>
        </header>

        {/* Context Badge */}
        {selectedText && (
          <div className={styles.contextBadge}>
            <div className={styles.contextInfo}>
              <SparkleIcon />
              <span>Context Attached ({selectedText.length} chars)</span>
            </div>
            <div className={styles.contextPreview}>
              "{selectedText.slice(0, 80)}{selectedText.length > 80 ? '...' : ''}"
            </div>
            <button className={styles.contextClear} onClick={clearContext}>
              Remove
            </button>
          </div>
        )}

        {/* Messages */}
        <div className={styles.messages}>
          {messages.length === 0 ? (
            <div className={styles.welcome}>
              <div className={styles.welcomeIcon}><BotIcon /></div>
              <h4>Welcome!</h4>
              <p>Ask me anything about Physical AI and Humanoid Robotics.</p>
              <p className={styles.welcomeTip}>
                <SparkleIcon /> <strong>Tip:</strong> Select text on any page to use it as context!
              </p>
              <div className={styles.suggestions}>
                {[
                  'What is Physical AI?',
                  'Explain robot locomotion',
                  'How do robots perceive the world?'
                ].map(q => (
                  <button key={q} onClick={() => setInput(q)} className={styles.suggestion}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map(msg => (
              <div key={msg.id} className={`${styles.message} ${styles[msg.role]}`}>
                <div className={styles.messageContent}>
                  {renderMarkdown(msg.content)}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className={styles.sources}>
                    <span>Sources:</span>
                    {msg.sources.map((s, i) => (
                      <span key={i} className={styles.sourceChip}>
                        {s.source}
                      </span>
                    ))}
                  </div>
                )}

                {msg.contextUsed === 'selected' && (
                  <div className={styles.contextUsedBadge}>
                    <SparkleIcon /> Used selected context
                  </div>
                )}

                <time className={styles.timestamp}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </time>
              </div>
            ))
          )}

          {isLoading && (
            <div className={`${styles.message} ${styles.assistant}`}>
              <div className={styles.typing}>
                <span /><span /><span />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className={styles.inputArea}>
          <div className={styles.inputWrapper}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={selectedText ? "Ask about the selected text..." : "Ask a question..."}
              rows={1}
              disabled={isLoading}
              className={styles.input}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className={styles.sendBtn}
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          </div>
          <p className={styles.inputHint}>Press Enter to send, Shift+Enter for new line</p>
        </div>
      </div>
    </>
  );
}
