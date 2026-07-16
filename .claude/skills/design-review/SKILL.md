---
name: design-review
description: Audit an existing page of Yi Jing's academic homepage for visual quality and apply the fixes — hierarchy, spacing/whitespace, alignment, color use, typography, and responsive behavior. Use when the user asks to polish, refine, critique, improve the look of, or "make prettier" a page or the whole site.
---

# Design review & polish

Critique a page against this site's design system, then apply the high-confidence
fixes directly. **First read the `homepage-design-system` skill** (SKILL.md +
TOKENS.md) so the review is measured against the actual tokens, not generic taste.

## Process

1. **Look at the real thing.** Read the page's markup (`_pages/*.md`, layouts) and
   the relevant SCSS. If it helps, build/serve locally (`bundle exec jekyll serve`)
   or produce a quick standalone render (see **html-mockup**) to see it rendered,
   not just imagined. Note the viewport(s) you're judging (desktop + ≤760px mobile).

2. **Score each dimension**, citing `file:line`. Be concrete — "the meta column
   and body baselines don't align" beats "spacing feels off":

   - **Hierarchy** — does weight+size alone make the reading order obvious? Any
     heading relying on color? Competing focal points?
   - **Whitespace & rhythm** — consistent section padding (`2.15rem`), item padding
     (`1rem`)? Cramped or ballooning gaps? Prose measure ≤ 39rem?
   - **Alignment** — does content sit on the 880px column and the meta/content
     grid? Ragged left edges, off-grid one-offs?
   - **Color** — single teal accent only? Any stray hue, gradient, or filled card?
     Sufficient contrast (muted #667085 on cream is the floor for small text)?
   - **Typography** — system sans throughout? Line-heights in the 1.6–1.82 band for
     prose? Orphaned headings, all-caps, justified text?
   - **Responsive** — every multi-column block collapses to one column at 760px?
     Tap targets ≥ ~40px? Images fluid?
   - **Consistency** — reuses `research-*` classes, or did a one-off diverge?

3. **Report findings** ordered by impact, each with: what's wrong, why it matters,
   and the specific fix (token/class/value). Separate "will fix now" (high
   confidence, clearly better) from "consider" (subjective/riskier — ask first).

4. **Apply the high-confidence fixes.** Edit markup and/or `_sass/_research.scss`
   using CSS variables (never raw hex). Keep changes minimal and reversible;
   preserve the mobile fallbacks. Then rebuild to confirm no SCSS/Liquid breakage.

## Bias

Toward **removal and restraint** — this design earns its elegance by leaving things
out. Prefer deleting a divider, tightening a measure, or unifying a weight over
adding anything new. Don't propose redesigns unless asked; polish what's there.
Flag anti-patterns from the design system (shadowed cards, second accent, icon
clutter, wide measures, center-aligned prose) as defects.
