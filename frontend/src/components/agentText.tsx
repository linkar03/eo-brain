import type { ReactNode } from 'react';

// Chat replies arrive as plain text in which the model uses a little Markdown:
// **bold**, `code` and "* " or "- " bullets. AgentText renders that subset and
// nothing else. A reply is never interpreted as HTML: every piece becomes a
// React text node, so markup in a reply shows up as the characters it is.

const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`)/g;

function inline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.length > 4 && part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="font-semibold">{part.slice(2, -2)}</strong>;
    }
    if (part.length > 2 && part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} className="font-mono text-[0.85em] bg-black/5 rounded px-1 break-all">
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}

export function AgentText({ text }: { text: string }) {
  // A bullet marker is "*" or "-" at the start of a line, followed by a space.
  const withBullets = text.replace(/^([ \t]*)[*-][ \t]+/gm, '$1• ');
  return <p className="whitespace-pre-wrap">{inline(withBullets)}</p>;
}
