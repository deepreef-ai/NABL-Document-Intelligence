import { useState } from "react";
import { renderDocumentUrl, type DocumentOut } from "../api/client";
import { ChevronLeftIcon, ChevronRightIcon } from "./Icon";

interface Props {
  document: DocumentOut;
  page: number;
  onPageChange: (page: number) => void;
  highlightedFieldId?: string | null;
}

/**
 * The scanned page, beside the values read off it.
 *
 * This panel does not scroll. It is sticky instead, so it stays level with
 * whichever part of the form the reviewer has scrolled to — a scroller inside
 * a sticky column gives two scrollbars racing each other and no way to tell
 * which one the wheel is about to move.
 */
export default function DocumentViewer({ document, page, onPageChange, highlightedFieldId }: Props) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [renderFailed, setRenderFailed] = useState(false);

  const isRenderable = document.content_type !== "" && !document.filename.toLowerCase().endsWith(".docx");
  const boxesOnThisPage = document.fields.filter((f) => f.source_bbox && (f.source_page ?? 0) === page);
  const pageCount = document.page_count ?? null;
  // Fields grounded to THIS page — the quickest way to see whether extraction
  // actually reached the later pages of a long scan or only the first few.
  const fieldsOnThisPage = document.fields.filter((f) => f.source_page === page).length;
  const isLastPage = pageCount !== null && page >= pageCount - 1;

  if (!isRenderable) {
    return (
      <div className="viewer">
        <p className="viewer-unrenderable">
          This document has no visual rendering (DOCX source). The extracted values beside it are
          grounded to the whole document rather than to a region of a page.
        </p>
      </div>
    );
  }

  return (
    <div className="viewer">
      <div className="viewer-toolbar">
        <button
          className="btn-icon"
          onClick={() => onPageChange(Math.max(0, page - 1))}
          disabled={page === 0}
          aria-label="Previous page"
        >
          <ChevronLeftIcon size={16} />
        </button>
        <button
          className="btn-icon"
          onClick={() => onPageChange(page + 1)}
          disabled={renderFailed || isLastPage}
          aria-label="Next page"
        >
          <ChevronRightIcon size={16} />
        </button>
        <span className="viewer-page-label">
          {pageCount !== null ? `Page ${page + 1} of ${pageCount}` : `Page ${page + 1}`}
        </span>
        <span className="chip viewer-toolbar-spacer">
          {fieldsOnThisPage} {fieldsOnThisPage === 1 ? "field" : "fields"}
        </span>
        {document.extraction_source && (
          <span className="chip chip-accent">{document.extraction_source.replace(/_/g, " ")}</span>
        )}
      </div>

      {pageCount !== null && pageCount > 1 && (
        <div className="viewer-pages">
          {Array.from({ length: pageCount }, (_, i) => {
            const count = document.fields.filter((f) => f.source_page === i).length;
            return (
              <button
                key={i}
                className={`page-chip${i === page ? " is-active" : ""}${count === 0 ? " is-empty" : ""}`}
                onClick={() => onPageChange(i)}
                aria-label={`Go to page ${i + 1}, ${count} field${count === 1 ? "" : "s"} extracted`}
                aria-current={i === page ? "true" : undefined}
                title={`Page ${i + 1} — ${count} field(s) extracted`}
              >
                {i + 1}
              </button>
            );
          })}
        </div>
      )}

      <div className="viewer-sheet">
        {!renderFailed && (
          <img
            key={`${document.id}-${page}`}
            src={renderDocumentUrl(document.id, page)}
            // Not the filename: alt text is read aloud and shown when the
            // image fails, and an uploaded filename can identify the patient.
            // The page number is what orients a reader.
            alt={`Page ${page + 1} of the uploaded document`}
            onLoad={(e) => {
              const img = e.currentTarget;
              setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
              setRenderFailed(false);
            }}
            onError={() => setRenderFailed(true)}
          />
        )}
        {renderFailed && <p className="viewer-sheet-empty">Could not render page {page + 1}.</p>}
        {naturalSize &&
          boxesOnThisPage.map((f) => {
            const box = f.source_bbox!;
            return (
              <div
                key={f.id}
                className={`bbox${f.id === highlightedFieldId ? " is-active" : ""}`}
                style={{
                  left: `${(box.x / naturalSize.w) * 100}%`,
                  top: `${(box.y / naturalSize.h) * 100}%`,
                  width: `${(box.w / naturalSize.w) * 100}%`,
                  height: `${(box.h / naturalSize.h) * 100}%`,
                }}
              />
            );
          })}
      </div>
    </div>
  );
}
