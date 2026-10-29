export default function Logo({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 2L2 7l10 5 10-5-10-5Z" stroke="var(--accent)" strokeWidth="1.75" strokeLinejoin="round" />
      <path d="M2 17l10 5 10-5" stroke="var(--accent)" strokeWidth="1.75" strokeLinejoin="round" />
      <path d="M2 12l10 5 10-5" stroke="var(--accent)" strokeWidth="1.75" strokeLinejoin="round" />
    </svg>
  )
}
