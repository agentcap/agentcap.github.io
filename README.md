# Pradeep Yarlagadda — Portfolio

A dependency-free static portfolio for GitHub Pages. Open `index.html` through a local static server, for example `python -m http.server 4173` from this directory.

## Content
- `index.html`: role, selected projects, publications, writing index, and biography.
- `projects/splitwise-ai.html`: project case study and the first engineering note.
- `projects/daybook.html`: Android journaling project, placed immediately after Splitwise AI in selected work.
- `stylesheet.css`: responsive layouts, local fonts, and reduced-motion support.
- `site.js`: progressive scroll reveals and an animation pause button.

## Add a blog post
1. Copy `writing/template.html` to `writing/your-slug.html`.
2. Replace the title, description, heading, date, and article content. Remove its `noindex` meta tag when ready to publish.
3. Add a link with the `note-row` structure in `index.html` under `#writing`. Label drafts as forthcoming rather than linking to an unfinished page.
4. Test desktop/mobile layouts, links, and keyboard focus before pushing to the GitHub Pages publishing branch.

The writing index currently links to a source-grounded project note. Planned topics are explicitly labeled; no fictional blog posts or publication dates are used. No résumé link is included.

## Assets & verification
Splitwise AI screenshots use synthetic demo data, copied from that project's documented screenshots. Daybook images come from its running verification app: journal content, the meal illustration and displayed reviews are synthetic; its recording screenshot shows an actual paused emulator recording with no detected speech. Daybook remains a personal beta, with physical Galaxy and live OpenAI checks pending. Research images and portrait are retained from the original portfolio. DM Sans is self-hosted under its included SIL Open Font License. The original portfolio credited Jon Barron's academic website template; the current layout is a new implementation.

No build step or hosted API is needed. Existing repository GitHub Pages settings remain applicable. This update does not publish itself until its changes reach the configured publishing branch.
