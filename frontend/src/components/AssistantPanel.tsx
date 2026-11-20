import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ChatEntry } from "../api/client";
import { ChatIcon, MinimiseIcon, SendIcon } from "./Icon";

interface Props {
  history: ChatEntry[];
  onSend: (message: string) => Promise<void>;
  /** Names the document the answers are about — used on the source chips. */
  documentLabel: string;
  suggestions?: string[];
}

/**
 * The document assistant, docked rather than floating.
 *
 * It sits in the bottom-right corner over the workspace and collapses to a
 * single pill. A reviewer asks it about a value they are looking at, so it
 * has to stay reachable while they scroll — but it must also get out of the
 * way, because the thing it is answering about is underneath it.
 */

const DEFAULT_SUGGESTIONS = [
  "Which values need checking?",
  "Where did the sample date come from?",
  "Summarise the test methods",
];

/**
 * The page an answer leans on, if it named one.
 *
 * Read out of the reply's own text rather than asserted: the chip is there so
 * a reviewer can go and look, and a citation the answer did not actually make
 * would send them to the wrong page with more confidence than the answer
 * deserves. No page named, no chip.
 */
function citedPage(content: string): number | null {
  const m = /\bp(?:age|g)?\.?\s?(\d{1,3})\b/i.exec(content);
  if (!m) return null;
  const page = Number(m[1]);
  return Number.isFinite(page) && page > 0 ? page : null;
}

export default function AssistantPanel({ history, onSend, documentLabel, suggestions }: Props) {
  const [open, setOpen] = useState(true);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const observers = useRef(new Map<HTMLElement, ResizeObserver>());

  /**
   * Publish this panel's height as --dock-h.
   *
   * The dock is `position: fixed` in the bottom-right corner, so it is
   * outside the page's flow and nothing below it knows it is there — the
   * submit row ends up permanently underneath it, unreachable no matter how
   * far you scroll. The page reserves space with padding, and the padding
   * has to match whatever the panel currently is: ~450px expanded, ~44px as
   * a pill, and different again when a long answer arrives. A guessed
   * constant is either a hole in the layout or not enough room, so it is
   * measured rather than guessed.
   */
  const measure = useCallback((node: HTMLElement | null) => {
    if (!node) {
      document.documentElement.style.removeProperty("--dock-h");
      return;
    }
    const publish = () =>
      document.documentElement.style.setProperty("--dock-h", `${Math.ceil(node.offsetHeight)}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(node);
    observers.current.set(node, observer);
  }, []);

  useLayoutEffect(() => {
    const current = observers.current;
    return () => {
      for (const observer of current.values()) observer.disconnect();
      current.clear();
      document.documentElement.style.removeProperty("--dock-h");
    };
  }, []);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: "smooth" });
  }, [history, sending, open]);

  async function send(message: string) {
    const text = message.trim();
    if (!text || sending) return;
    setSending(true);
    setDraft("");
    try {
      await onSend(text);
    } finally {
      setSending(false);
    }
  }

  if (!open) {
    return (
      <button className="assistant-launcher" ref={measure} onClick={() => setOpen(true)}>
        <ChatIcon size={17} />
        Ask about this document
      </button>
    );
  }

  return (
    <aside className="assistant" ref={measure} aria-label="Document assistant">
      <div className="assistant-head">
        <span className="assistant-avatar" aria-hidden="true">
          <ChatIcon size={17} />
        </span>
        <div>
          <h2>Ask about this document</h2>
          <p>Answers cite the page they came from</p>
        </div>
        <button className="assistant-min" onClick={() => setOpen(false)} aria-label="Minimise the assistant">
          <MinimiseIcon size={16} />
        </button>
      </div>

      <div className="assistant-body" ref={bodyRef}>
        {history.length === 0 && (
          <div className="msg msg-assistant">
            <div className="bubble">
              Ask about any value on this page — where it was read from, why its confidence is low,
              or what the document says about a method.
            </div>
          </div>
        )}
        {history
          .filter((entry) => entry.role !== "system")
          .map((entry, i) => {
            const page = entry.role === "assistant" ? citedPage(entry.content) : null;
            return (
              <div className={`msg msg-${entry.role}`} key={i}>
                <div className="bubble">{entry.content}</div>
                {page !== null && (
                  <span className="msg-source">
                    p{page} · {documentLabel}
                  </span>
                )}
              </div>
            );
          })}
        {sending && (
          <div className="msg msg-assistant">
            <div className="bubble">Reading the document…</div>
          </div>
        )}
      </div>

      <div className="assistant-suggestions">
        {(suggestions ?? DEFAULT_SUGGESTIONS).map((s) => (
          <button key={s} className="suggestion" onClick={() => void send(s)} disabled={sending}>
            {s}
          </button>
        ))}
      </div>

      <div className="assistant-input">
        <label className="visually-hidden" htmlFor="assistant-input">
          Ask a question about this document
        </label>
        <input
          id="assistant-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void send(draft);
          }}
          placeholder="Ask about an extracted value…"
          disabled={sending}
        />
        <button
          className="assistant-send"
          onClick={() => void send(draft)}
          disabled={sending || !draft.trim()}
          aria-label="Send message"
        >
          <SendIcon size={17} />
        </button>
      </div>
    </aside>
  );
}
