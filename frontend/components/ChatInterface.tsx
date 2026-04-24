// Author: Bradley R. Kinnard
'use client';

import {
    BeliefEvent,
    ChatSession,
    createChatSession,
    deleteAllChatSessions,
    deleteChatSession,
    getChatSession,
    listChatSessions,
    sendChatMessage
} from '@/lib/api';
import clsx from 'clsx';
import { Activity, Menu, Plus, Send, Zap } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { BeliefActivityPanel } from './BeliefActivityPanel';
import { Sidebar } from './Sidebar';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  beliefs?: {
    created: string[];
    reinforced: string[];
    mutated: string[];
    used: string[];
  };
}

interface ChatInterfaceProps {
  token?: string | null;
}

export function ChatInterface({ token }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [beliefPanelOpen, setBeliefPanelOpen] = useState(true);
  const [beliefEvents, setBeliefEvents] = useState<BeliefEvent[]>([]);
  const [wsConnected, setWsConnected] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    loadSessions();
  }, []);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//localhost:8000/chat/ws`;

    const connect = () => {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsConnected(true);
      };

      ws.onclose = () => {
        setWsConnected(false);
        setTimeout(connect, 3000);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'belief_event') {
          // Add event with unique key to prevent duplicates
          setBeliefEvents(prev => {
            const newEvent = data.data;
            // Check if this event already exists (by belief_id + event_type + similar timestamp)
            const isDuplicate = prev.some(e =>
              e.belief_id === newEvent.belief_id &&
              e.event_type === newEvent.event_type &&
              Math.abs(new Date(e.timestamp).getTime() - new Date(newEvent.timestamp).getTime()) < 5000
            );
            if (isDuplicate) return prev;
            return [newEvent, ...prev].slice(0, 50);
          });
        }
      };

      ws.onerror = () => {};
      wsRef.current = ws;
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const loadSessions = async () => {
    try {
      const result = await listChatSessions();
      setSessions(result.sessions);
    } catch {}
  };

  const loadSession = async (id: string) => {
    try {
      const session = await getChatSession(id);
      setSessionId(id);

      const msgs: Message[] = [];
      for (const turn of session.turns) {
        if (turn.user_message) {
          msgs.push({
            id: `${turn.id}-user`,
            role: 'user',
            content: turn.user_message,
            timestamp: new Date(turn.timestamp),
          });
        }
        if (turn.assistant_message) {
          msgs.push({
            id: `${turn.id}-assistant`,
            role: 'assistant',
            content: turn.assistant_message,
            timestamp: new Date(turn.timestamp),
            beliefs: {
              created: turn.beliefs_created,
              reinforced: turn.beliefs_reinforced,
              mutated: turn.beliefs_mutated,
              used: turn.beliefs_used,
            },
          });
        }
      }
      setMessages(msgs);
    } catch {}
  };

  const startNewSession = async () => {
    try {
      const session = await createChatSession();
      setSessionId(session.id);
      setMessages([]);
      await loadSessions();
    } catch {}
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteChatSession(id);
      if (sessionId === id) {
        setSessionId(null);
        setMessages([]);
      }
      await loadSessions();
    } catch {}
  };

  const handleDeleteAllSessions = async () => {
    try {
      await deleteAllChatSessions();
      setSessionId(null);
      setMessages([]);
      setSessions([]);
      setBeliefEvents([]);
    } catch {}
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const turn = await sendChatMessage(userMessage.content, sessionId || undefined, token);

      if (!sessionId) {
        await loadSessions();
      }

      const assistantMessage: Message = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: turn.assistant_message,
        timestamp: new Date(turn.timestamp),
        beliefs: {
          created: turn.beliefs_created,
          reinforced: turn.beliefs_reinforced,
          mutated: turn.beliefs_mutated,
          used: turn.beliefs_used,
        },
      };

      console.log('Chat turn received:', {
        events: turn.events,
        created: turn.beliefs_created,
        reinforced: turn.beliefs_reinforced
      });

      setMessages(prev => [...prev, assistantMessage]);
      setBeliefEvents(prev => {
        // Deduplicate events by belief_id + event_type
        const newEvents = turn.events.filter(newEvent =>
          !prev.some(e =>
            e.belief_id === newEvent.belief_id &&
            e.event_type === newEvent.event_type &&
            Math.abs(new Date(e.timestamp).getTime() - new Date(newEvent.timestamp).getTime()) < 5000
          )
        );
        const updated = [...newEvents, ...prev].slice(0, 50);
        console.log('Updated beliefEvents:', updated.length);
        return updated;
      });
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: `msg-${Date.now()}-error`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-neutral-100">
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        sessions={sessions}
        currentSessionId={sessionId}
        onNewSession={startNewSession}
        onSelectSession={loadSession}
        onDeleteSession={handleDeleteSession}
        onDeleteAllSessions={handleDeleteAllSessions}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-[#1f1f1f]">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 hover:bg-[#1a1a1a] rounded-lg transition-colors text-neutral-500"
              >
                <Menu className="w-5 h-5" />
              </button>
            )}
            <span className="font-medium text-neutral-200">ABES</span>
            <div className={clsx(
              "flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-medium",
              wsConnected
                ? "bg-emerald-500/10 text-emerald-500"
                : "bg-neutral-800 text-neutral-500"
            )}>
              <span className={clsx(
                "w-1.5 h-1.5 rounded-full",
                wsConnected ? "bg-emerald-500" : "bg-neutral-500"
              )} />
              {wsConnected ? 'Live' : 'Offline'}
            </div>
          </div>
          <button
            onClick={() => setBeliefPanelOpen(!beliefPanelOpen)}
            className={clsx(
              "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors",
              beliefPanelOpen
                ? "bg-[#1a1a1a] text-neutral-300 border border-[#2a2a2a]"
                : "text-neutral-500 hover:bg-[#141414]"
            )}
          >
            <Activity className="w-4 h-4" />
            <span className="hidden sm:inline">Activity</span>
          </button>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-2xl mx-auto py-6 px-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30 flex items-center justify-center mb-4">
                  <span className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">A</span>
                </div>
                <h2 className="text-xl font-medium text-neutral-200 mb-2">
                  ABES Ecology Debugger
                </h2>
                <p className="text-neutral-500 text-sm max-w-sm leading-relaxed">
                  Inject context to create beliefs. Watch the ecology respond: reinforcement, contradiction, tension, mutation, and decay in real time.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((msg) => (
                  <MessageBubble key={msg.id} message={msg} />
                ))}
                {isLoading && <LoadingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-[#1f1f1f] px-4 py-4">
          <div className="max-w-2xl mx-auto">
            <div className="flex items-end gap-2 bg-[#141414] rounded-xl border border-[#2a2a2a] focus-within:border-[#3a3a3a] transition-colors">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Inject context into the ecology..."
                rows={1}
                className="flex-1 bg-transparent px-4 py-3 text-neutral-100 placeholder-neutral-600 resize-none focus:outline-none min-h-[48px] max-h-[200px] text-sm"
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                className={clsx(
                  "p-2.5 m-1.5 rounded-lg transition-colors",
                  inputValue.trim() && !isLoading
                    ? "bg-neutral-200 text-neutral-900 hover:bg-white"
                    : "bg-[#1a1a1a] text-neutral-600 cursor-not-allowed"
                )}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <BeliefActivityPanel
        isOpen={beliefPanelOpen}
        onClose={() => setBeliefPanelOpen(false)}
        events={beliefEvents}
      />
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <div className={clsx("flex items-start gap-3", isUser && "flex-row-reverse")}>
      <div className={clsx(
        "w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-medium",
        isUser
          ? "bg-neutral-200 text-neutral-900"
          : "bg-[#1a1a1a] border border-[#2a2a2a] text-neutral-400"
      )}>
        {isUser ? 'U' : 'A'}
      </div>
      <div className={clsx("flex-1 min-w-0", isUser && "text-right")}>
        <div className={clsx(
          "inline-block px-4 py-2.5 rounded-2xl max-w-[85%] text-sm leading-relaxed",
          isUser
            ? "bg-neutral-200 text-neutral-900 rounded-tr-md"
            : "bg-[#1a1a1a] text-neutral-200 rounded-tl-md border border-[#2a2a2a]"
        )}>
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
        {!isUser && message.beliefs && (
          <BeliefIndicators beliefs={message.beliefs} />
        )}
      </div>
    </div>
  );
}

function BeliefIndicators({ beliefs }: { beliefs: Message['beliefs'] }) {
  if (!beliefs) return null;

  const total = beliefs.created.length + beliefs.reinforced.length + beliefs.mutated.length;
  if (total === 0 && beliefs.used.length === 0) return null;

  return (
    <div className="flex items-center gap-2 mt-1.5 text-[11px]">
      {beliefs.created.length > 0 && (
        <span className="flex items-center gap-1 text-emerald-600">
          <Plus className="w-3 h-3" />
          {beliefs.created.length}
        </span>
      )}
      {beliefs.reinforced.length > 0 && (
        <span className="flex items-center gap-1 text-sky-600">
          <Zap className="w-3 h-3" />
          {beliefs.reinforced.length}
        </span>
      )}
      {beliefs.used.length > 0 && (
        <span className="text-neutral-600">
          {beliefs.used.length} beliefs used
        </span>
      )}
    </div>
  );
}

function LoadingIndicator() {
  return (
    <div className="flex items-start gap-3">
      <div className="w-7 h-7 rounded-full bg-[#1a1a1a] border border-[#2a2a2a] flex items-center justify-center text-xs text-neutral-400">
        A
      </div>
      <div className="flex items-center gap-1.5 px-4 py-3">
        <span className="w-1.5 h-1.5 bg-neutral-600 rounded-full animate-pulse" />
        <span className="w-1.5 h-1.5 bg-neutral-600 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
        <span className="w-1.5 h-1.5 bg-neutral-600 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  );
}
