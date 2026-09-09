import { useMemo, useState, type ReactNode } from "react";
import type { ExtractedFieldOut } from "../api/client";
import { groupReviewFields, type ReviewGroup, type ReviewSubgroup } from "../api/reviewGrouping";
import FormField from "./FormField";

/**
 * The review panel's fields, laid out as a form: section, sub-heading, fields.
 *
 *   Patient Details            <- section: what kind of thing it is
 *     Header Information       <- sub-heading: the document's own
 *       Patient Name  Sex  DOB
 *
 * Two levels rather than one because the section alone puts thirty test
 * results in a single undifferentiated block, and the document's own heading
 * alone ("Header Information") tells a reviewer nothing about what kind of
 * value they are confirming. Together they read the way the page did.
 *
 * Sections are always expanded. Collapsing was a mistake here: a reviewer
 * working through a document reads every section anyway, so a collapsed one is
 * a value they have to remember to go and look at — and a section that is
 * empty because nothing was extracted looks identical to one that is merely
 * shut. The heading is a signpost, not a control.
 *
 * The grouping itself matters because the previous flat grid reflowed across
 * two columns, which broke repeating tables apart: a results table's five
 * samples interleaved so sample 1's id sat beside sample 2's number.
 */

interface Props {
  fields: ExtractedFieldOut[];
  threshold: number;
  highlightedFieldId: string | null;
  onFocus: (id: string) => void;
  onJumpToPage: (field: ExtractedFieldOut) => void;
  onSave: (fieldId: string, value: string) => Promise<void>;
  onAccept: (field: ExtractedFieldOut) => Promise<void>;
}

type RenderField = (f: ExtractedFieldOut) => ReactNode;

function Subgroup({
  subgroup,
  showHeading,
  renderField,
}: {
  subgroup: ReviewSubgroup;
  showHeading: boolean;
  renderField: RenderField;
}) {
  return (
    <div className="rg-sub">
      {showHeading && subgroup.heading && (
        <h4 className="rg-subheading">{subgroup.heading}</h4>
      )}

      {subgroup.rows.length > 0 && (
        <div className="form-fields-grid">{subgroup.rows.map((r) => renderField(r.field))}</div>
      )}

      {subgroup.records.map((rec) => (
        <div className="rg-record" key={rec.label}>
          <div className="rg-record-label">{rec.label}</div>
          {/* A record's own fields never reflow into a second column — that
              is exactly what scrambled the table in the first place. */}
          <div className="form-fields-grid rg-record-grid">
            {rec.rows.map((r) => renderField(r.field))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Group({
  group,
  threshold,
  highlightedFieldId,
  onFocus,
  onJumpToPage,
  onSave,
  onAccept,
}: { group: ReviewGroup } & Omit<Props, "fields">) {
  const renderField: RenderField = (f) => (
    <FormField
      key={f.id}
      field={f}
      threshold={threshold}
      isFocused={f.id === highlightedFieldId}
      onFocus={() => onFocus(f.id)}
      onJumpToPage={() => onJumpToPage(f)}
      onSave={(value) => onSave(f.id, value)}
      onAccept={() => onAccept(f)}
    />
  );

  // One sub-heading under a section is the section — printing both stacks two
  // headings on top of each other saying the same thing.
  const showSubheadings =
    group.subgroups.length > 1 ||
    (group.subgroups[0]?.heading ?? "").toLowerCase() !== group.heading.toLowerCase();

  return (
    <section className="rg-group">
      <h3 className="rg-heading">{group.heading}</h3>

      <div className="rg-body">
        {group.subgroups.map((sub) => (
          <Subgroup
            key={sub.heading || "__unfiled__"}
            subgroup={sub}
            showHeading={showSubheadings}
            renderField={renderField}
          />
        ))}
      </div>
    </section>
  );
}

export default function ReviewFieldGroups(props: Props) {
  const { fields, threshold } = props;
  const [onlyReview, setOnlyReview] = useState(false);

  const groups = useMemo(() => {
    const visible = onlyReview ? fields.filter((f) => f.confidence < threshold || !f.value) : fields;
    return groupReviewFields(visible, threshold);
  }, [fields, threshold, onlyReview]);

  const toCheck = fields.filter((f) => f.confidence < threshold || !f.value).length;

  if (fields.length === 0) {
    return <p className="form-empty-state">No structured fields extracted from this document.</p>;
  }

  return (
    <div className="rg">
      {toCheck > 0 && (
        <div className="rg-toolbar">
          <label className="rg-toggle">
            <input type="checkbox" checked={onlyReview} onChange={(e) => setOnlyReview(e.target.checked)} />
            Needs checking ({toCheck})
          </label>
        </div>
      )}

      {groups.length === 0 ? (
        <p className="form-empty-state">Nothing matches that filter.</p>
      ) : (
        groups.map((g) => <Group key={g.heading} group={g} {...props} />)
      )}
    </div>
  );
}
