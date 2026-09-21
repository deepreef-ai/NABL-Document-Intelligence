import { Fragment, useMemo, type ReactNode } from "react";
import type { ExtractedFieldOut, TestRowOut } from "../api/client";
import { goldRank } from "../api/goldSections";
import { groupReviewFields, type ReviewGroup, type ReviewSubgroup } from "../api/reviewGrouping";
import FormField from "./FormField";
import TestResultsTable from "./TestResultsTable";

/**
 * The auto-filled fields, laid out as a form: section, sub-heading, cards.
 *
 *   Sample                     <- section: the gold taxonomy's own
 *     Header Information       <- sub-heading: the document's own
 *       Sample Id  ·  Drawn On  ·  Specimen
 *
 * Two levels rather than one because the section alone puts thirty values in
 * a single undifferentiated block, and the document's own heading alone
 * ("Header Information") says nothing about what kind of value is being
 * confirmed. Together they read the way the page did.
 *
 * Sections are always expanded. Collapsing was a mistake here: a reviewer
 * works through every section anyway, so a collapsed one is a value they have
 * to remember to go and look at — and a section empty because nothing was
 * extracted looks identical to one that is merely shut. The heading is a
 * signpost, not a control.
 *
 * Sections run in DOCUMENT order (see reviewGrouping), not taxonomy order,
 * because the page is on the left and the form is on the right and the two
 * have to be readable in the same direction. The results table is one of the
 * sections rather than a banner pinned above them, for the same reason.
 */

interface Props {
  fields: ExtractedFieldOut[];
  /** Results-table rows, shown as a table in the section they belong to. */
  tests?: TestRowOut[];
  /**
   * How many extracted fields were folded away as duplicates of a table cell
   * (see splitRedundantFields). Reported under the table rather than left
   * silent: a reviewer comparing the form against the Extracted JSON would
   * otherwise find values in the JSON that the form never asked about.
   */
  foldedCount?: number;
  threshold: number;
  highlightedFieldId: string | null;
  onFocus: (id: string) => void;
  onJumpToPage: (field: ExtractedFieldOut) => void;
  onSave: (fieldId: string, value: string) => Promise<void>;
  onAccept: (field: ExtractedFieldOut) => Promise<void>;
}

type RenderField = (f: ExtractedFieldOut) => ReactNode;

const TESTS_HEADING = "Test Results";
const METHOD = /(method|protocol|technique)/i;

/**
 * A sub-heading for fields the document did not file under one of its own.
 *
 * A report that prints its test methods as a bare list — one row per method,
 * no heading above them — otherwise arrives as an unnamed block of seven
 * values in the middle of a form. Naming it from what the fields ARE is
 * deterministic and reproducible; nothing here guesses.
 */
function subheadingFor(subgroup: ReviewSubgroup): string {
  if (subgroup.heading) return subgroup.heading;
  const rows = [...subgroup.rows, ...subgroup.records.flatMap((r) => r.rows)];
  if (rows.length > 1 && rows.every((r) => METHOD.test(r.field.field_path))) return "Test methods";
  return "";
}

const EMPTY_SUBGROUP: ReviewSubgroup = { heading: "", rows: [], records: [], place: { rect: null, arrival: 0 } };

/**
 * Whether the document's own sub-headings are worth printing.
 *
 * Plenty of reports print one heading over the whole page — "Header and
 * Sample Details" — which then appears under every section here, six times,
 * saying the same nothing each time. A sub-heading earns its line by
 * distinguishing one block of the page from another, so if the document only
 * ever used one, none are shown.
 */
function subheadingsAreInformative(groups: ReviewGroup[]): boolean {
  const distinct = new Set(
    groups.flatMap((g) => g.subgroups.map((s) => subheadingFor(s))).filter(Boolean),
  );
  return distinct.size > 1;
}

function Subgroup({
  subgroup,
  showHeading,
  renderField,
}: {
  subgroup: ReviewSubgroup;
  showHeading: boolean;
  renderField: RenderField;
}) {
  const heading = subheadingFor(subgroup);
  return (
    <div className="rg-sub">
      {showHeading && heading && <h4 className="rg-subheading">{heading}</h4>}

      {subgroup.rows.length > 0 && (
        <div className="field-grid">{subgroup.rows.map((r) => renderField(r.field))}</div>
      )}

      {subgroup.records.map((rec) => (
        <div className="rg-record" key={rec.label}>
          <div className="rg-record-label">{rec.label}</div>
          {/* A record's own fields never reflow into a second column — that
              is exactly what scrambled the table in the first place. */}
          <div className="field-grid">{rec.rows.map((r) => renderField(r.field))}</div>
        </div>
      ))}
    </div>
  );
}

function SectionHead({ title, count, confirmed }: { title: string; count: number; confirmed?: number }) {
  return (
    <div className="rg-head">
      <h3>{title}</h3>
      <span className="chip">{count}</span>
      {confirmed !== undefined && (
        <span className="rg-head-count">
          {confirmed} of {count} confirmed
        </span>
      )}
    </div>
  );
}

function Group({
  group,
  lead,
  subheadings,
  threshold,
  highlightedFieldId,
  onFocus,
  onJumpToPage,
  onSave,
  onAccept,
}: { group: ReviewGroup; lead?: ReactNode; subheadings: boolean } & Omit<Props, "fields" | "tests" | "foldedCount">) {
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

  const confirmed = group.subgroups
    .flatMap((s) => [...s.rows, ...s.records.flatMap((r) => r.rows)])
    .filter((r) => r.field.accepted).length;

  // One sub-heading under a section IS the section — printing both stacks two
  // headings saying the same thing.
  const showSubheadings =
    subheadings &&
    (group.subgroups.length > 1 ||
      (subheadingFor(group.subgroups[0] ?? EMPTY_SUBGROUP).toLowerCase() !== group.heading.toLowerCase()));

  return (
    <section className="rg-group">
      <SectionHead title={group.heading} count={group.total} confirmed={confirmed} />
      {lead}
      {group.subgroups.map((sub) => (
        <Subgroup
          key={sub.heading || "__unfiled__"}
          subgroup={sub}
          showHeading={showSubheadings}
          renderField={renderField}
        />
      ))}
    </section>
  );
}

/**
 * Where the results table goes when no extracted field was filed under
 * "Test Results" — a table can carry its own page number, but not reliably
 * and not on the same 0-based scale as a field's, so it is placed by meaning
 * instead: after the sections a report prints above its results, before the
 * signatures and notes below them.
 */
function testsSlot(groups: ReviewGroup[]): number {
  const rank = goldRank(TESTS_HEADING);
  const below = groups.findIndex((g) => goldRank(g.heading) > rank);
  return below === -1 ? groups.length : below;
}

function TestsBlock({ tests, foldedCount = 0 }: { tests: TestRowOut[]; foldedCount?: number }) {
  return (
    <>
      <TestResultsTable tests={tests} />
      {foldedCount > 0 && (
        <p className="tests-note">
          {foldedCount} method value{foldedCount === 1 ? "" : "s"} extracted separately{" "}
          {foldedCount === 1 ? "is" : "are"} already shown in the Method column above, so{" "}
          {foldedCount === 1 ? "it is" : "they are"} not repeated as fields to confirm.
        </p>
      )}
    </>
  );
}

function TestsSection({ tests, foldedCount }: { tests: TestRowOut[]; foldedCount?: number }) {
  return (
    <section className="rg-group">
      <SectionHead title={TESTS_HEADING} count={tests.length} />
      <TestsBlock tests={tests} foldedCount={foldedCount} />
    </section>
  );
}

export default function ReviewFieldGroups(props: Props) {
  const { fields, tests = [], threshold, foldedCount = 0 } = props;

  const groups = useMemo(() => groupReviewFields(fields, threshold), [fields, threshold]);
  const subheadings = useMemo(() => subheadingsAreInformative(groups), [groups]);

  // The table joins the section the extracted result fields were filed under
  // when there is one, so the screen does not print "Test Results" twice with
  // half the results under each.
  const inside = tests.length > 0 ? groups.findIndex((g) => g.heading === TESTS_HEADING) : -1;
  const standalone = tests.length > 0 && inside === -1 ? testsSlot(groups) : -1;

  if (fields.length === 0) {
    return tests.length > 0 ? (
      <TestsSection tests={tests} foldedCount={foldedCount} />
    ) : (
      <p className="rg-empty" style={{ padding: 16 }}>
        No structured fields were extracted from this document.
      </p>
    );
  }

  return (
    <>
      {groups.map((g, i) => (
        <Fragment key={g.heading}>
          {i === standalone && <TestsSection tests={tests} foldedCount={foldedCount} />}
          <Group
            group={g}
            lead={i === inside ? <TestsBlock tests={tests} foldedCount={foldedCount} /> : null}
            subheadings={subheadings}
            {...props}
          />
        </Fragment>
      ))}
      {standalone === groups.length && <TestsSection tests={tests} foldedCount={foldedCount} />}
    </>
  );
}
