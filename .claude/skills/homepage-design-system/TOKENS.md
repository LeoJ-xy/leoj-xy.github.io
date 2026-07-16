# Design Tokens — extracted from `_sass/_research.scss` and `_sass/_variables.scss`

Always reference these as CSS custom properties (defined in `:root` in
`_research.scss`). Do not hardcode the hex values in markup.

## Color

| Token | Value | Use |
|---|---|---|
| `--research-bg` | `#fffdf8` | Page background (warm cream) |
| `--research-text` | `#24292f` | Headings & primary body text |
| `--research-muted` | `#667085` | Secondary text, descriptions, nav |
| `--research-soft` | `#8b8175` | Meta text (dates, venues) |
| `--research-line` | `#e7dfd3` | Hairline dividers & borders |
| `--research-accent` | `#286f6c` | Links, active state (the ONLY accent) |
| `--research-accent-soft` | `#eef6f4` | Hover background on pills |
| link hover | `#174f4c` | Darker teal on `a:hover` |
| separator glyph | `#b8aea0` | The `/` between inline links |
| item top border | `rgba(36,41,47,0.08)` | Divider between `.research-item`s |

## Layout

| Token | Value |
|---|---|
| `--research-max` | `880px` (content column) |
| Detail article width | `720px` (`.research-detail__article`) |
| Prose measure cap | `39rem` (`.research-lead`, section `<p>`) |
| Site padding | `0 1.4rem` (desktop), `0 1rem` (≤760px) |
| Section meta column | `9.5rem` head / `8.5rem` item, gap `2rem` / `1.25rem` |
| Mobile breakpoint | `760px` → single column |

## Typography (system sans: `-apple-system` stack)

| Element | Size | Weight | Line-height |
|---|---|---|---|
| Hero `h1` | `2.15rem` (mobile `1.72rem`) | 760 | 1.15 |
| Page heading `h1` | `2rem` | 760 | 1.15 |
| Detail `h1` | `2rem` (mobile `1.68rem`) | 760 | 1.2 |
| Lead paragraph | `1.08rem` (mobile `1rem`) | — | 1.68 |
| Section head `h2` | `1rem` | 760 | 1.35 |
| Item `h3` | `1.04rem` | 720 | 1.38 |
| Body `p` | `0.96rem` | — | 1.62 |
| Detail body | `1rem` | — | 1.82 |
| Meta / eyebrow | `0.84rem` | 500–650 | 1.45 |
| Detail `h2` / `h3` | `1.35rem` / `1.12rem` | 730 | 1.3 |

Weights available in this stack render at 500, 600, 650, 720, 730, 760 — hierarchy
comes from weight + size, never color.

## Shape & elevation

- Media radius: `8px` (portrait, content images).
- Pills (tags, language switch): `999px`.
- Only shadow on the site: portrait `0 18px 42px rgba(36,41,47,0.12)` and the nav
  dropdown `0 12px 28px rgba(36,41,47,0.1)`. Do not add card shadows.

## Spacing rhythm

- Section vertical padding: `2.15rem 0` (desktop), `1.75rem 0` (mobile).
- Item padding: `1rem 0`.
- Hero padding: `2.2rem 0 2.75rem`.
- Gaps between inline links: `0.35rem 0.7rem`.
