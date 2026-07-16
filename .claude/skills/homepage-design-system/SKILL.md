---
name: homepage-design-system
description: The visual design system for Yi Jing's academic homepage — exact color/type/spacing tokens, markup conventions, and taste rules. Load this BEFORE generating, reviewing, or mocking up any page, section, or component for this site so everything stays visually consistent. Referenced by homepage-page, design-review, and html-mockup.
---

# Homepage Design System

This site is a customized Jekyll `academicpages` theme with a bespoke **research
homepage** layer. The look is a warm, editorial, minimal academic style: cream
paper background, a single teal accent, a strict two-column meta/content grid,
generous line-height, and almost no decoration. Restraint is the whole point —
when in doubt, remove, don't add.

The canonical source of truth is `_sass/_research.scss` (compiled via
`assets/css/main.scss`). Never hardcode hex values in new markup — always use the
CSS custom properties defined there. Exact tokens are listed in
[TOKENS.md](TOKENS.md) — read it before producing any styles or markup.

## Non-negotiable principles

1. **One accent, used sparingly.** `--research-accent` (#286f6c teal) is for links
   and the active state only. Never introduce a second accent hue, gradients, or
   colored backgrounds on cards. Headings are near-black text, not accent-colored.
2. **The two-column grid is the signature.** Content lives inside a `880px` column
   (`--research-max`), and within sections a left meta column (dates, labels) sits
   beside the main content. Preserve this rhythm; don't center-align body text.
3. **Hairline dividers, not boxes.** Sections separate with a single
   `1px solid var(--research-line)` bottom border. Avoid drop-shadowed cards,
   heavy borders, or filled panels (the portrait's soft shadow is the only shadow).
4. **Type does the work.** Weight (600–760) and size create hierarchy, not color or
   boxes. Body copy is `line-height: 1.6–1.82`. Keep measure ≤ `39rem` for prose.
5. **System sans everywhere** (`$sans-serif` / `-apple-system` stack). No web-font
   loading, no serif except the theme's caption default.
6. **Rounded, quiet corners.** `8px` radius on media; `999px` pills for toggles/tags.
7. **Mobile collapses to one column** at `760px` — every new multi-column block must
   have a matching single-column fallback (see the media query in `_research.scss`).

## Markup conventions

Research pages use `layout: research` (see `_layouts/research.html`), which wraps
content in `<main class="research-site">`. Build sections from these existing
classes rather than inventing new ones:

- `.research-hero` — name + lead + inline links + portrait (homepage top).
- `.research-section` + `.research-section__head` (left `<h2>` + optional `<p>`) —
  a labeled content block; content goes in `.research-list`.
- `.research-item` — a two-column row: `.research-item__meta` (left: date/venue) +
  `<h3>`/`<p>`/`.research-link-group`.
- `.research-year-block` — year-grouped lists (publications/talks).
- `.research-inline-links` / `.research-link-group` — slash-separated link rows.
- `.research-tags`, `.research-clean-list`, `.research-language-switch` — see scss.

If a genuinely new pattern is needed, add its styles to `_sass/_research.scss`
(reusing the CSS variables), keep the class prefix `research-`, and add the
`@media (max-width: 760px)` fallback. Prefer extending an existing class first.

## Anti-patterns (reject these)

- Colored/gradient hero banners, hero background images.
- Card grids with shadows and rounded borders for list content.
- Icon-heavy UI, emoji as bullets, decorative rules or dingbats.
- Multiple font families, bold accent-colored headings, all-caps body.
- Center-aligned paragraphs, justified text, measures wider than ~39rem.
- Inline `style="..."` with raw hex — use the CSS variables.

## Workflow hooks

- To create a page/section → use the **homepage-page** skill.
- To critique/polish an existing page → use the **design-review** skill.
- To preview a design in isolation before touching Jekyll → use **html-mockup**.
