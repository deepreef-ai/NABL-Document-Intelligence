import { useEffect, useRef, useState } from "react";
import type { ChatEntry } from "../api/client";

interface Props {
  history: ChatEntry[];
  onSend: (message: string) => Promise<void>;
  placeholder?: string;
  disabled?: boolean;
  title?: string;
  subtitle?: string;
}

function Avatar({ role }: { role: ChatEntry["role"] }) {
  if (role === "user") {
    return (
      <div className="chat-avatar chat-avatar-user">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="8" r="3.5" />
          <path d="M4.5 20c1.5-4 4.5-6 7.5-6s6 2 7.5 6" strokeLinecap="round" />
        </svg>
      </div>
    );
  }
  return (
    <div className="chat-avatar chat-avatar-assistant">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="4" y="8" width="16" height="11" rx="3" />
        <path d="M12 8V4" strokeLinecap="round" />
        <circle cx="12" cy="3" r="1.2" fill="currentColor" stroke="none" />
        <circle cx="9" cy="13.5" r="1.3" fill="currentColor" stroke="none" />
        <circle cx="15" cy="13.5" r="1.3" fill="currentColor" stroke="none" />
      </svg>
    </div>
  );
}

export default function ChatPanel({ history, onSend, placeholder, disabled, title, subtitle }: Props) {
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [history, sending]);

  async function handleSend() {
    const message = draft.trim();
    if (!message || sending) return;
    setSending(true);
    setDraft("");
    try {
      await onSend(message);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="chat-panel">
      {title && (
        <div className="chat-panel-header">
          <div className="chat-avatar chat-avatar-assistant chat-avatar-header">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="4" y="8" width="16" height="11" rx="3" />
              <path d="M12 8V4" strokeLinecap="round" />
              <circle cx="12" cy="3" r="1.2" fill="currentColor" stroke="none" />
              <circle cx="9" cy="13.5" r="1.3" fill="currentColor" stroke="none" />
              <circle cx="15" cy="13.5" r="1.3" fill="currentColor" stroke="none" />
            </svg>
          </div>
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
        </div>
      )}
      <div className="chat-history" ref={scrollRef}>
        {history.map((entry, i) => (
          <div key={i} className={`chat-message chat-message-${entry.role}`}>
            <Avatar role={entry.role} />
            <div className="chat-bubble">{entry.content}</div>
          </div>
        ))}
        {sending && (
          <div className="chat-message chat-message-assistant">
            <Avatar role="assistant" />
            <div className="chat-bubble chat-typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </div>
      <div className="chat-input-row">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder={placeholder ?? "Type your reply…"}
          disabled={disabled || sending}
        />
        <button onClick={handleSend} disabled={disabled || sending || !draft.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
