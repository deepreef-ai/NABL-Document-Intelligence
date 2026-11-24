/**
 * Every icon on these screens, as inline stroke SVG.
 *
 * Inline rather than a font or a sprite: an icon here is almost always inside
 * a button that changes colour on hover, focus or confirmation, and
 * `currentColor` is what makes it follow. Stroke rather than fill keeps them
 * the same weight as the 11px labels they sit beside.
 *
 * They are decorative in every use — the button around them carries the
 * accessible name — so each is `aria-hidden`.
 */

interface Props {
  size?: number;
  className?: string;
}

function svg(size: number, className: string | undefined, children: React.ReactNode) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export const UploadIcon = ({ size = 20, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" />
    </>
  ));

export const CheckIcon = ({ size = 15, className }: Props) =>
  svg(size, className, <path d="m4.5 12.5 4.5 4.5L19.5 6.5" />);

export const CloseIcon = ({ size = 16, className }: Props) =>
  svg(size, className, (
    <>
      <path d="m6 6 12 12" />
      <path d="M18 6 6 18" />
    </>
  ));

export const ChevronLeftIcon = ({ size = 16, className }: Props) =>
  svg(size, className, <path d="m14.5 5-7 7 7 7" />);

export const ChevronRightIcon = ({ size = 16, className }: Props) =>
  svg(size, className, <path d="m9.5 5 7 7-7 7" />);

export const ArrowRightIcon = ({ size = 16, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M4 12h15" />
      <path d="m13 6 6 6-6 6" />
    </>
  ));

export const RefreshIcon = ({ size = 15, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M20 11a8 8 0 0 0-13.7-5.3L4 8" />
      <path d="M4 4v4h4" />
      <path d="M4 13a8 8 0 0 0 13.7 5.3L20 16" />
      <path d="M20 20v-4h-4" />
    </>
  ));

export const CodeIcon = ({ size = 16, className }: Props) =>
  svg(size, className, (
    <>
      <path d="m8.5 8-4 4 4 4" />
      <path d="m15.5 8 4 4-4 4" />
      <path d="m13.5 5-3 14" />
    </>
  ));

export const CopyIcon = ({ size = 15, className }: Props) =>
  svg(size, className, (
    <>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M15 5.5A1.5 1.5 0 0 0 13.5 4h-8A1.5 1.5 0 0 0 4 5.5v8A1.5 1.5 0 0 0 5.5 15" />
    </>
  ));

export const DownloadIcon = ({ size = 15, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M12 4v11" />
      <path d="m7 10 5 5 5-5" />
      <path d="M4 17v1.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V17" />
    </>
  ));

export const FileIcon = ({ size = 16, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M13.5 4H7a1.5 1.5 0 0 0-1.5 1.5v13A1.5 1.5 0 0 0 7 20h10a1.5 1.5 0 0 0 1.5-1.5V9z" />
      <path d="M13.5 4v5h5" />
    </>
  ));

export const PlusIcon = ({ size = 15, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </>
  ));

export const ShieldIcon = ({ size = 16, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M12 3.5 5 6.2v5.1c0 4.1 2.8 7.8 7 9.2 4.2-1.4 7-5.1 7-9.2V6.2z" />
      <path d="m9 12 2.2 2.2L15.5 10" />
    </>
  ));

export const SendIcon = ({ size = 17, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M20 4 3.5 10.6a.5.5 0 0 0 .05.94L10.5 13.5l2 6.9a.5.5 0 0 0 .93.08z" />
      <path d="M20 4 10.5 13.5" />
    </>
  ));

export const MinimiseIcon = ({ size = 16, className }: Props) =>
  svg(size, className, <path d="M6 15h12" />);

export const ChatIcon = ({ size = 17, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M20 5H4a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h2v4l5-4h9a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1Z" />
    </>
  ));

export const GaugeIcon = ({ size = 16, className }: Props) =>
  svg(size, className, (
    <>
      <path d="M4.5 17a8.5 8.5 0 1 1 15 0" />
      <path d="m12 12.5 4-3.5" />
      <circle cx="12" cy="13" r="1.3" />
    </>
  ));
