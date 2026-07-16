---
name: homepage-page
description: Generate a new page or section for Yi Jing's academic homepage (Jekyll academicpages + custom research layout) that matches the existing style automatically. Use when the user asks to add/build a new page, section, list, hero, publications/talks block, or content block to the personal homepage. Reuses the research-* markup and CSS variables — no restyling from scratch.
---

# Build a homepage page / section

Produce Jekyll markup that drops straight into this site and looks like it was
always there. **First read the `homepage-design-system` skill** (SKILL.md + TOKENS.md)
so tokens, grid rhythm, and taste rules are loaded — then follow the steps below.

## Steps

1. **Confirm scope in one line.** New standalone page (needs a permalink + nav
   entry) vs. a section added to an existing page? What content/data drives it?
   If a data collection already exists (`_publications`, `_talks`, `_portfolio`,
   `_data/`), prefer looping over it in Liquid rather than hardcoding.

2. **Match the layout.** Research pages use front matter `layout: research`
   (wraps content in `<main class="research-site">`). A new page starts:
   ```yaml
   ---
   layout: research
   permalink: /your-path/
   title: "Page Title"
   author_profile: false
   ---
   ```

3. **Compose from existing classes** (do not invent new ones unless truly needed):
   - Page top → `.research-page-heading` (h1 + optional intro `<p>`).
   - A labeled block → `.research-section` with `.research-section__head`
     (left `<h2>` + optional `<p>`) and content in `.research-list`.
   - A row/entry → `.research-item` with `.research-item__meta` (left column:
     date/venue) then `<h3>` + `<p>` + `.research-link-group`.
   - Year-grouped lists → `.research-year-block`.
   - Link rows → `.research-inline-links` / `.research-link-group`
     (slash-separated automatically via CSS `::after`).
   - Tags → `.research-tags`; sub-lists → `.research-clean-list`.

   See `_pages/about.md` for the reference structure and
   `_layouts/research.html` / `research-detail.html` for wrappers.

4. **Wire it up.** New top-level page → add to the nav in `_data/navigation.yml`
   if it should appear in the menu. Detail pages for a collection → set the
   collection's `defaults` layout in `_config.yml` (check what's already there).

5. **Only extend styles as a last resort.** If a genuinely new pattern is needed,
   add rules to `_sass/_research.scss` using the CSS variables, keep the
   `research-` class prefix, and include a `@media (max-width: 760px)` single-column
   fallback. Never add inline `style` with raw hex.

6. **Verify.** Build locally to catch Liquid/SCSS errors before claiming done:
   `bundle exec jekyll build` (or `serve`). Mention the local URL if serving.

## Quality bar

- Reuse before create; the page should be indistinguishable in style from `about.md`.
- Keep prose measure ≤ 39rem, one accent color, hairline dividers, system sans.
- Every multi-column block collapses cleanly on mobile.
- Semantic HTML: `<section>`, `<article>`, real headings in order, `alt` on images.

If the user wants to preview a bolder redesign before committing to Jekyll, suggest
the **html-mockup** skill first, then port the approved result back here.
