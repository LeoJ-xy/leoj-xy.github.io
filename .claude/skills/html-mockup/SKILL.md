---
name: html-mockup
description: Build a standalone single-file HTML mockup to preview a homepage design or component in the browser before integrating it into the Jekyll site. Use when the user wants to explore/prototype a new look, compare layout options, or see a design rendered quickly without touching the real site. Ships a starter template pre-loaded with the site's design tokens.
---

# Standalone HTML mockup

Prototype fast in one self-contained `.html` file the user can open directly, so
design decisions happen on something they can see — then port the approved result
into Jekyll. **First read the `homepage-design-system` skill** (SKILL.md +
TOKENS.md) so the mockup uses the real tokens, not approximations.

## How to build

1. **Start from the template.** Copy [template.html](template.html) — it already
   has the `:root` design tokens, the system-sans stack, the 880px column, the
   `research-*` base styles, and the 760px mobile query inlined. Build the new
   idea inside `<main class="research-site">` using those classes.

2. **One file, no build, no network.** Everything inline: `<style>` in `<head>`,
   any JS in a `<script>`. No CDN links, no web fonts, no external images (use a
   local path the user has, a `background: var(--research-line)` placeholder box,
   or an inline SVG). It must render correctly opened as `file://`.

3. **Stay on-system by default.** Same cream bg, single teal accent, hairline
   dividers, weight-based hierarchy. If the user explicitly wants to explore a
   bolder direction, do it — but keep it in the mockup; don't smuggle new patterns
   into the real SCSS until approved.

4. **Offer variants when the direction is open.** For "which layout is better"
   questions, put 2–3 options in the same file (stacked, labeled `<h6>` dividers)
   so they compare side by side in one scroll.

5. **Save and tell the user how to view.** Write to a scratch path like
   `mockups/<name>.html` (create the folder; it's outside the Jekyll build). Give
   the absolute path and suggest `open mockups/<name>.html` (macOS) to view.

6. **Port on approval.** Once the user likes it, hand off to the **homepage-page**
   skill to convert the mockup into proper Jekyll markup + `research-*` classes,
   moving any new rules into `_sass/_research.scss` with the mobile fallback. The
   mockup file is disposable — it's a preview, not the deliverable.

## Guardrails

- The mockup is throwaway; never commit it or reference it from the live site.
- Keep the HTML semantic and the token names identical to the real system so the
  port is a copy-paste, not a rewrite.
- Don't reach for shadowed cards, gradients, or a second accent unless the user
  is explicitly exploring away from the current aesthetic.
