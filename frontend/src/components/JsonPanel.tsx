import { useMemo, useState } from "react";
import { CodeIcon, CopyIcon, DownloadIcon } from "./Icon";

interface Props {
  /** The extraction in the labelled dataset's shape, or null while loading. */
  data: unknown;
  /** Schema the payload conforms to, shown as a chip — e.g. "NABL 151". */
  schema: string;
  /** Suggested filename for the download, without extension. */
  filename: string;
  fieldCount: number;
  testCount: number;
  error?: string | null;
}

/**
 * The extraction, as JSON, on the page rather than behind a link.
 *
 * It used to be a "View JSON" link that opened a new tab. That is fine for
 * someone who already knows the shape and worse for everyone else: the JSON
 * is what the next system consumes, so a reviewer about to approve should be
 * able to see the thing they are approving without leaving the screen they
 * are approving it on.
 *
 * Highlighting is done by tokenising the serialised text rather than walking
 * the object, so nesting, key order and indentation are exactly what
 * JSON.stringify produced and exactly what Copy and Download hand over.
 */

const TOKEN =
  /("(?:\\.|[^"\\])*")(\s*:)|("(?:\\.|[^"\\])*")|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b|([{}[\],:])/g;

function highlight(text: string) {
  const out: React.ReactNode[] = [];
  let last = 0;
  let key = 0;

  for (const m of text.matchAll(TOKEN)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));

    const [, propName, colon, str, num, literal, punct] = m;
    if (propName !== undefined) {
      out.push(<span className="json-key" key={key++}>{propName}</span>);
      out.push(<span className="json-punct" key={key++}>{colon}</span>);
    } else if (str !== undefined) {
      out.push(<span className="json-string" key={key++}>{str}</span>);
    } else if (num !== undefined) {
      out.push(<span className="json-number" key={key++}>{num}</span>);
    } else if (literal !== undefined) {
      out.push(<span className="json-number" key={key++}>{literal}</span>);
    } else if (punct !== undefined) {
      out.push(<span className="json-punct" key={key++}>{punct}</span>);
    }

    last = at + m[0].length;
  }

  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function JsonPanel({ data, schema, filename, fieldCount, testCount, error }: Props) {
  const [copied, setCopied] = useState(false);

  const text = useMemo(() => (data === null || data === undefined ? "" : JSON.stringify(data, null, 2)), [data]);
  const body = useMemo(() => highlight(text), [text]);
  const lineCount = text ? text.split("\n").length : 0;

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access can be refused outright (an insecure origin, a
      // permissions policy). Saying nothing would look like the button did
      // not respond at all.
      setCopied(false);
    }
  }

  function download() {
    // A blob of the text already on screen, not a link to the API: what gets
    // saved is then byte-for-byte what the reviewer just read, and it works
    // the same whether or not the API is on this origin.
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="json-panel">
      <div className="json-head">
        <span className="json-head-icon">
          <CodeIcon size={16} />
        </span>
        <h2>Extracted JSON</h2>
        <span className="chip json-schema-chip">{schema}</span>
        <div className="json-head-actions">
          <button className="btn btn-onDark" onClick={copy} disabled={!text}>
            <CopyIcon size={15} />
            {copied ? "Copied" : "Copy"}
          </button>
          <button className="btn btn-onDark" onClick={download} disabled={!text}>
            <DownloadIcon size={15} />
            Download
          </button>
        </div>
      </div>

      <pre className="json-body">
        {error ? error : text ? body : "Loading the extraction…"}
      </pre>

      <p className="json-foot">
        {fieldCount} field{fieldCount === 1 ? "" : "s"} · {testCount} result row
        {testCount === 1 ? "" : "s"} · {lineCount} line{lineCount === 1 ? "" : "s"} of JSON
      </p>
    </section>
  );
}
