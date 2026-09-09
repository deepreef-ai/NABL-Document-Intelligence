import { useState } from "react";
import { renderDocumentUrl, type DocumentOut } from "../api/client";

interface Props {
  document: DocumentOut;
  page: number;
  onPageChange: (page: number) => void;
  highlightedFieldId?: string | null;
}

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
      <div className="document-viewer document-viewer-unrenderable">
        <p>
          <strong>{document.filename}</strong> has no visual rendering (DOCX source). Extracted values below are
          grounded to the whole document, not a specific region.
        </p>
      </div>
    );
  }

  return (
    <div className="document-viewer">
      <div className="document-viewer-toolbar">
        <button onClick={() => onPageChange(Math.max(0, page - 1))} disabled={page === 0}>
          ◀ Prev
        </button>
        <span>{pageCount !== null ? `Page ${page + 1} of ${pageCount}` : `Page ${page + 1}`}</span>
        <button onClick={() => onPageChange(page + 1)} disabled={renderFailed || isLastPage}>
          Next ▶
        </button>
        <span className="document-viewer-page-fields">
          {fieldsOnThisPage} {fieldsOnThisPage === 1 ? "field" : "fields"} from this page
        </span>
        {document.extraction_source && (
          <span className={`extraction-source-badge source-${document.extraction_source}`}>
            {document.extraction_source.replace(/_/g, " ")}
          </span>
        )}
      </div>
      {pageCount !== null && pageCount > 1 && (
        <div className="document-viewer-page-strip">
          {Array.from({ length: pageCount }, (_, i) => {
            const count = document.fields.filter((f) => f.source_page === i).length;
            return (
              <button
                key={i}
                className={`page-chip ${i === page ? "active" : ""} ${count === 0 ? "page-chip-empty" : ""}`}
                onClick={() => onPageChange(i)}
                title={`Page ${i + 1} — ${count} field(s) extracted`}
              >
                {i + 1}
              </button>
            );
          })}
        </div>
      )}
      <div className="document-viewer-canvas">
        {!renderFailed && (
          <img
            key={`${document.id}-${page}`}
            src={renderDocumentUrl(document.id, page)}
            alt={document.filename}
            onLoad={(e) => {
              const img = e.currentTarget;
              setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
              setRenderFailed(false);
            }}
            onError={() => setRenderFailed(true)}
          />
        )}
        {renderFailed && <p className="document-viewer-error">Could not render page {page + 1}.</p>}
        {naturalSize &&
          boxesOnThisPage.map((f) => {
            const box = f.source_bbox!;
            const isHighlighted = f.id === highlightedFieldId;
            return (
              <div
                key={f.id}
                className={`bbox-overlay ${isHighlighted ? "bbox-overlay-active" : ""}`}
                style={{
                  left: `${(box.x / naturalSize.w) * 100}%`,
                  top: `${(box.y / naturalSize.h) * 100}%`,
                  width: `${(box.w / naturalSize.w) * 100}%`,
                  height: `${(box.h / naturalSize.h) * 100}%`,
                }}
                title={f.field_path}
              />
            );
          })}
      </div>
    </div>
  );
}
