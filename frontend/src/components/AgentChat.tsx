import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Bot, User, Loader2, MapPin, X, Maximize2, Minimize2, Database } from 'lucide-react';
import type { BBox } from '../types';
import { API_BASE } from '../api/config';
const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 150;

/** Rows returned by the read-only data agent (query_database tool). */
interface QueryPayload {
  sql: string | null;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  jobId?: string;
  jobStatus?: 'pending' | 'processing' | 'completed' | 'failed';
  jobProgress?: number;
  jobMessage?: string;
  query?: QueryPayload;
}

interface AgentChatProps {
  onJobCompleted?: (jobId: string, result: any) => void;
  onFlyTo?: (bbox: BBox) => void;
}

function cleanAgentText(text: string): string {
  return text.replace(/\[JOB_CREATED:[^\]]*\]/g, '').trim();
}

export function AgentChat({ onJobCompleted, onFlyTo }: AgentChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const pollTimers = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => { pollTimers.current.forEach(clearTimeout); };
  }, []);

  const updateMessageJob = useCallback(
    (msgId: string, jobStatus: string, progress: number, jobMessage?: string) => {
      setMessages(prev =>
        prev.map(m =>
          m.id === msgId
            ? { ...m, jobStatus: jobStatus as Message['jobStatus'], jobProgress: progress, jobMessage }
            : m
        )
      );
    }, []
  );

  const pollJob = useCallback(
    (jobId: string, msgId: string, attempt = 0) => {
      if (attempt >= MAX_POLL_ATTEMPTS) {
        updateMessageJob(msgId, 'failed', 0);
        pollTimers.current.delete(msgId);
        return;
      }

      const timer = window.setTimeout(async () => {
        try {
          const res = await fetch(`${API_BASE}/jobs/${jobId}`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);

          const data = await res.json();
          updateMessageJob(msgId, data.status, data.progress ?? 0, data.message);

          if (data.status === 'completed') {
            pollTimers.current.delete(msgId);
            onJobCompleted?.(jobId, data.result);
          } else if (data.status === 'failed') {
            pollTimers.current.delete(msgId);
          } else {
            pollJob(jobId, msgId, attempt + 1);
          }
        } catch {
          pollJob(jobId, msgId, attempt + 1);
        }
      }, POLL_INTERVAL_MS);

      pollTimers.current.set(msgId, timer);
    },
    [updateMessageJob, onJobCompleted]
  );

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', text: trimmed };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, session_id: sessionId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();

      if (data.session_id && !sessionId) setSessionId(data.session_id);

      // Fly to analysis area immediately (don't wait for job completion)
      if (data.bbox && onFlyTo) {
        onFlyTo(data.bbox as BBox);
      }

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: cleanAgentText(data.text),
        jobId: data.job_id || undefined,
        jobStatus: data.job_id ? 'pending' : undefined,
        query: data.query ?? undefined,
      };
      setMessages(prev => [...prev, assistantMsg]);

      if (data.job_id) {
        pollJob(data.job_id, assistantMsg.id);
      }
    } catch {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: 'Unable to reach the server. Please verify the backend is running and try again.',
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 bg-slate-900 text-white rounded-full shadow-lg flex items-center justify-center hover:bg-slate-800 transition-colors"
        title="Open EO Brain Agent"
      >
        <Bot className="w-6 h-6" />
      </button>
    );
  }

  return (
    <div
      className={`fixed z-50 bg-white rounded-xl shadow-2xl border border-gray-200 flex flex-col transition-all duration-200 ${
        isExpanded
          ? 'bottom-4 right-4 left-4 top-4 sm:left-auto sm:w-[560px] sm:top-4'
          : 'bottom-6 right-6 w-[380px] h-[520px]'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 bg-slate-50 rounded-t-xl">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-slate-700" />
          <span className="text-sm font-semibold text-slate-800">EO Brain Agent</span>
          {sessionId && <span className="text-xs text-slate-400 font-mono">session active</span>}
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setIsExpanded(!isExpanded)} className="p-1.5 hover:bg-gray-200 rounded-lg">
            {isExpanded ? <Minimize2 className="w-4 h-4 text-gray-500" /> : <Maximize2 className="w-4 h-4 text-gray-500" />}
          </button>
          <button onClick={() => setIsOpen(false)} className="p-1.5 hover:bg-gray-200 rounded-lg">
            <X className="w-4 h-4 text-gray-500" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-center py-8 text-gray-400">
            <MapPin className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Ask me to analyze any location.</p>
            <p className="text-xs mt-1 text-gray-300">
              e.g. "Analyze vegetation in Milan" or "Flood risk near Po River"
            </p>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && <Bot className="w-5 h-5 text-slate-500 mt-1 flex-shrink-0" />}
            <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm leading-relaxed ${
              msg.role === 'user' ? 'bg-slate-800 text-white' : 'bg-gray-100 text-gray-800'
            }`}>
              <p className="whitespace-pre-wrap">{msg.text}</p>
              {msg.query && <QueryResult query={msg.query} />}
              {msg.jobId && msg.jobStatus && (
                <JobStatusBadge status={msg.jobStatus} progress={msg.jobProgress ?? 0} message={msg.jobMessage} />
              )}
            </div>
            {msg.role === 'user' && <User className="w-5 h-5 text-slate-400 mt-1 flex-shrink-0" />}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-2 items-center">
            <Bot className="w-5 h-5 text-slate-500" />
            <div className="bg-gray-100 rounded-lg px-3 py-2">
              <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-3 border-t border-gray-100">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Analyze vegetation near Rome..."
            disabled={isLoading}
            className="flex-1 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-300 disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="px-3 py-2 bg-slate-800 text-white rounded-lg hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '-';
  return String(value);
}

/** Executed SQL (collapsed) and the rows it returned, exactly as the API sent them. */
function QueryResult({ query }: { query: QueryPayload }) {
  return (
    <div className="mt-2 rounded-lg border border-gray-200 bg-white text-xs text-gray-700">
      <div className="flex items-center gap-1.5 px-2 py-1.5 border-b border-gray-100 text-gray-500">
        <Database className="w-3 h-3" />
        <span>
          {query.row_count} row{query.row_count === 1 ? '' : 's'}
          {query.truncated ? ' (truncated)' : ''}
        </span>
      </div>
      {query.sql && (
        <details className="px-2 py-1.5 border-b border-gray-100">
          <summary className="cursor-pointer select-none text-gray-500">SQL</summary>
          <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-[11px] text-gray-600">{query.sql}</pre>
        </details>
      )}
      {query.columns.length > 0 && query.rows.length > 0 && (
        <div className="max-h-48 overflow-auto">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-gray-50">
              <tr>
                {query.columns.map(col => (
                  <th key={col} className="px-2 py-1 text-left font-medium text-gray-600 whitespace-nowrap">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {query.rows.map((row, i) => (
                <tr key={i} className="border-t border-gray-100">
                  {row.map((cell, j) => (
                    <td key={j} className="px-2 py-1 font-mono whitespace-nowrap">{formatCell(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function JobStatusBadge({ status, progress, message }: { status: string; progress: number; message?: string }) {
  const cfg: Record<string, { bg: string; bar: string; text: string; label: string }> = {
    pending:    { bg: 'bg-yellow-50',  bar: 'bg-yellow-400', text: 'text-yellow-700', label: 'Queued...' },
    processing: { bg: 'bg-blue-50',    bar: 'bg-blue-500',   text: 'text-blue-700',   label: `${progress}%` },
    completed:  { bg: 'bg-green-50',   bar: 'bg-green-500',  text: 'text-green-700',  label: 'Map updated' },
    failed:     { bg: 'bg-red-50',     bar: 'bg-red-500',    text: 'text-red-700',    label: 'Failed' },
  };
  const c = cfg[status] ?? cfg.pending;
  const isActive = status === 'pending' || status === 'processing';

  return (
    <div className={`mt-2 rounded-lg p-2 ${c.bg}`}>
      <div className="flex items-center justify-between mb-1">
        <div className={`flex items-center gap-1.5 text-xs font-medium ${c.text}`}>
          {isActive && <Loader2 className="w-3 h-3 animate-spin" />}
          {c.label}
        </div>
        {message && isActive && (
          <span className={`text-xs ${c.text} opacity-75`}>{message}</span>
        )}
      </div>
      {isActive && (
        <div className="w-full h-1.5 bg-black/5 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-[2000ms] ease-linear ${c.bar}`}
            style={{ width: `${Math.max(progress, 3)}%` }}
          />
        </div>
      )}
      {status === 'completed' && (
        <div className={`text-xs ${c.text} flex items-center gap-1`}>
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          Analysis complete - results on map
        </div>
      )}
    </div>
  );
}

export default AgentChat;