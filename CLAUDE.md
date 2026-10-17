# Geometric Doodles — Blue Ink Archive

A standalone static gallery website displaying hand-drawn blue ballpoint pen sketches as a curated art portfolio.

## Tech Stack

- Plain HTML + CSS + vanilla JS — no frameworks, no build step
- Open `index.html` directly in a browser

## File Structure

- `index.html` — gallery page with inline SVG doodles, sketch sheet photos, logo concepts, about section
- `style.css` — dark navy design (#0f1940), responsive grid, paper-textured cards, lightbox
- `gallery.js` — category filtering, lightbox with keyboard support (Esc/click-outside)
- `images/sketches/` — iPhone photos of A4 sketch sheets (IMG_0089–0104.jpeg) + original scans (.jpg)
- `images/logos/` — SVG logo concepts hand-coded from recurring sketch motifs
- `_figma-export/` — original Figma Make export (React + Vite) used as design reference

## Design Tokens

- Background: `#0f1940`
- Card background: `#faf9f4`
- Accent: `#6378ff`
- Hover glow: `#8fa0ff`
- Text: `#e0e6ff`
- Ink color: `#294baa`
- Font: Inter (Google Fonts)

## Deployment

Intended for GitHub Pages — static files, no build required. Push and enable Pages on the `main` branch.
