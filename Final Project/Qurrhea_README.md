# Qurrhea

AI career agent for job search, tailored applications, outreach, and interview prep.

**Current version:** `v0.11.0`
**App stage:** Phase 1 complete

## What is built right now

### Core product surface
- Authenticated web app (Next.js 16 + Clerk + Supabase).
- Primary nav: `Dashboard`, `Outreach`, `Applications`, `Interview Prep`.
- Human-in-the-loop workflow from match discovery through application review and submission preview.

### Dashboard
- Document upload and management (resume/cover/supporting) with primary document designation.
- Profile parsing pipeline with parser insights and inferred fields.
- Recent activity feed: unified timeline of document generation, status changes, audit actions, and resume uploads grouped by job. Auto-refreshes every 30s and reacts to in-app events.
- Match list with:
  - ranked scoring + explanations,
  - saved/hidden/search filters + pagination,
  - quick review action,
  - source-aware links,
  - min-score threshold flow,
  - sort by date (newest/oldest toggle),
  - fade animation on hide.
- Board target management with:
  - manual board add/remove (x button),
  - enable/disable with Active/Inactive visual states,
  - profile/company-based discovery,
  - per-target and bulk validation with Validated/Failed tags,
  - half-page scrollable list.
- Job ingest trigger with board-level success/failure summaries.
- Manual URL job ingestion with authenticated portal scraping.

### Applications
- Kanban pipeline stages (`Saved`, `Drafting`, `Submitted`, `Interviewing`, `Offer`) with drag/drop.
- Notes, archive/restore, and recent activity timeline per card.
- Stage movement guardrails and readiness checks.
- Compact/Comfy view toggle; mobile compact hides secondary content for ultra-compact cards.

### Application Workshop
- Per-application review subpage with breadcrumbs.
- Document/response generation and regeneration with LLM provider fallback chain.
- Quality gates with auto-repair and style warnings.
- Validation checks + submission preview.
- Source-specific mapping for Greenhouse/Lever-style forms.
- Manual override preset support and audit logging.
- Browser-submit architecture with explicit approval gate and safe preview mode.

### Outreach
- Connection list with relevance ranking (degree + seniority + target match + source).
- Outreach logging/tracking by connection.
- Org chart generation via LLM.
- LinkedIn connection status and CSV-based seed import path.

### Interview Prep
- Prep session generation tied to selected applications.
- Multi-round tabs with per-round outline, cards, and format.
- Per-round date/time, logistics, and location with inline editing.
- Google Maps Places autocomplete for in-person locations.
- Round type change with regenerate/keep-existing flow.
- Drag-and-drop round reordering with pinned defaults.
- 7 question categories with AI-generated Q&A and persistent conversation history.
- Voice recording with Whisper transcription.
- Interviewer assignment from outreach connections.
- Prep checklist and session history.

### Integrations
- Clerk modal `Integrations` page:
  - LinkedIn OAuth connect/disconnect status.
  - Google Drive OAuth connect/disconnect status.
  - Drive auto-export on doc regeneration.
  - Custom Drive export destination path.
  - Google Calendar event sync.

## ML Pipelines & Optimizations

### Embedding generation
- **Model:** OpenAI `text-embedding-3-small` (1536 dimensions), configurable via `OPENAI_EMBEDDING_MODEL`.
- **Profile embeddings:** Built from structured profile text (name, location, preferred titles ×8, skills ×20, preferences JSON). Cached in `users_profile.embedding`; lazily generated on first match generation.
- **Job embeddings:** Concatenated from title + company + location + compensation + description.
- **Input limit:** 8000 chars max with whitespace normalization.

### Job matching pipeline
1. **ANN retrieval** via pgvector IVFFlat index (100 lists, cosine distance). `match_job_postings()` SQL function returns up to 180 candidates (configurable, range 20–500).
2. **Supplemental candidates** fill to 500 total from non-ANN jobs for completeness.
3. **Composite scoring** with weighted factors:
   - Semantic fit (0.55) — ANN cosine similarity, manual embedding fallback, or heuristic (`0.45 + titleFit × 0.35 + skillFit × 0.2`) when no embeddings.
   - Title fit (0.20) — exact normalized substring match → 1.0; partial → 0.35; empty → 0.45.
   - Skill fit (0.15) — `matched_count / min(preferred_skills, 6)` with special Go/Golang handling; no match → 0.2.
   - Location fit (0.10) — "remote" → 1.0; exact city → 0.9; substring → 0.75; no match → 0.25.
4. **Final score** clamped to [0.2, 0.98]. Top 100 returned (configurable via `MATCH_FINAL_LIMIT`, range 20–200).

### Embedding backfill
- Runs during match generation. Fills missing job embeddings from newest first.
- **Batch:** up to 20 jobs (configurable, range 0–200) with 2 concurrent workers (configurable, range 1–8).
- Custom `mapWithConcurrency()` loop; continues on individual failures.

### LLM document generation
- **Provider fallback chain:** OpenAI (`gpt-4o-mini`) → Anthropic (`claude-3-5-sonnet`) → deterministic template.
- Models configurable via `OPENAI_DOCS_MODEL` / `ANTHROPIC_DOCS_MODEL`.
- Temperature: 0.3, max tokens: 4096.
- **Prompt:** JSON-structured with candidate profile (9000 char resume, 3500 char excerpt), job details, output format spec, 6 style principles, 5 keyword-mirroring directives.
- **Quality gates:** resume min 140–250 words, cover letter min 120, Q&A min 3 pairs, placeholder text filter, ≥14% token overlap with source resume.
- **Auto-repair:** failed docs rebuilt via heuristic templates, re-validated; falls back to template if repair fails.
- **Style warnings:** resume 200–1400 words, cover letter 120–450, Q&A min 90 words.

### Resume parsing chain
Multi-method extraction in priority order:
1. **Mammoth** (DOCX via PK header detection) — min 100 chars for acceptance.
2. **pdftotext** (Poppler, `pdftotext -layout`) — 20s timeout, 10 MB buffer, min 100 chars.
3. **PDF stream extraction** — FlateDecode decompression, ToUnicode maps, Tj/TJ operator parsing.
4. **PDF literal extraction** — regex on parenthesized strings with octal decoding.
5. **UTF-8 raw** — readability check: ≥20% letters, ≥75% safe chars.
6. **ASCII runs** — printable byte extraction, min 4 chars per run, ≥60 total.
7. **OCR fallback** — pdftoppm → PNG at 200 DPI → `gpt-4o-mini` vision API. Enabled by default if OpenAI key present.

Best candidate selected by readability score: `(letterRatio × 0.55) + (safeRatio × 0.45) - penalties`. Minimum 0.38 for acceptance.

### Connection ranking
Composite relevance score (0–100) for outreach prioritization:
- **Connection degree:** 1st → 48, 2nd → 30, 3rd → 16.
- **Seniority:** IC → 8, Manager → 16, Director → 24, VP → 32, C-level → 40.
- **Target match:** same company → +30, same division → +20, same team → +30.
- **Source:** LinkedIn → 8, Social → 4, Import → 3, Manual → 2.

### Other LLM usage
- **Company lookup:** `gpt-4o-mini` (temp 0, 100 tokens) with Anthropic fallback.
- **Doc summary:** `gpt-4o-mini` (temp 0.2, 100 tokens) for supporting doc summarization.
- **Org chart:** `gpt-4o-mini` (temp 0.5, 4000 tokens) for team structure inference.
- **Interview Q&A:** `gpt-4o-mini` (temp 0.3–0.5, 600–1400 tokens) for question generation and feedback.

### Performance characteristics
- **No Redis or in-memory cache** — stateless API routes, Supabase as sole data layer.
- **Batch deduplication** on job ingest: 200 jobs per `.in()` query to avoid URL length limits.
- **Match insertion** uses snapshot-delete pattern (snapshot existing IDs → insert new → delete old) to minimize empty-matches window.
- **Profile embedding** cached after first generation; not re-computed unless profile changes.
- **Resume selection** for doc gen uses weighted scoring: primary status (140 pts) + content length + parse score + extracted chars.

## Architecture

- **Frontend:** Next.js 16 App Router + React 19 + Tailwind CSS v4.
- **Auth:** Clerk with middleware route protection.
- **Data + storage:** Supabase Postgres + storage buckets + pgvector (1536-dim embeddings).
- **LLM:** OpenAI/Anthropic-compatible provider routing with fallback chains.
- **Automation:** Playwright-based browser flow for application autofill/submit path.
- **Observability:** audit logs, automation runs, explicit status messages.
- **Deployment:** Railway via Dockerfile. Node 22 LTS required.

## Data model highlights

Core entities implemented across 25 migrations:
- `app_users` mapping (Clerk user → internal UUID).
- `users_profile` with preferences, personalization, and embedding.
- `job_postings` with embedding, `matches`, `applications`, `application_archive`.
- `document_versions` with source context, primary designation, and LLM summaries.
- `automation_runs` for parse jobs and ingest tracking.
- `outreach_connections`, outreach activities/threads.
- `interview_prep_sessions`, rounds, Q&A conversations, feedback.
- `job_board_targets` with validation state in `origin_context`.
- `companies` with foreign keys to jobs and connections.
- OAuth tables for LinkedIn, Google Drive, and Google Calendar.

## Setup

1. **Node version:** Node 22 LTS required (`.nvmrc` in `web/`). Turbopack panics on Node 23+.
2. **Install dependencies:**
   ```bash
   cd web && npm install
   ```
3. **Configure env:**
   ```bash
   cp web/.env.example web/.env.local
   # Fill in API keys — see .env.example for all variables with inline docs
   ```
4. **Run app:**
   ```bash
   cd web && npm run dev
   ```
5. **Apply migrations:** Run SQL files in `supabase/migrations/` (0001–0025) in order via Supabase SQL editor or psql.

## Available commands

All commands run from `web/`:

```bash
npm run dev          # Start dev server (Turbopack, auto-cleans .next cache)
npm run build        # Production build
npm run start        # Start production server
npm run lint         # ESLint (core-web-vitals + typescript rules)
npm run seed:jobs    # Seed sample job postings into Supabase
```

## Build and deploy troubleshooting

- If build appears stuck at `Finalizing page optimization ...`, check for overlapping builds: `ps aux | rg "next build"`.
- Standalone output is opt-in via `NEXT_STANDALONE_OUTPUT=true` (used in Docker builds only).
- Playwright excluded from bundle via `serverExternalPackages` in `next.config.ts`.
- `HOSTNAME=0.0.0.0` required in Railway runner stage for standalone server reachability.
- After switching Node versions, always `rm -rf node_modules && npm install`.

## Environment variables (key ones)

- **Auth/Data:** `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- **LLM:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `OPENAI_DOCS_MODEL`, `ANTHROPIC_DOCS_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_OCR_MODEL`
- **Matching/ingest:** `JOB_INGEST_MAX_PER_BOARD`, `MATCH_EMBEDDING_BACKFILL_ENABLED`, `MATCH_EMBEDDING_BACKFILL_LIMIT`, `MATCH_EMBEDDING_BACKFILL_CONCURRENCY`, `MATCH_ANN_CANDIDATE_LIMIT`, `MATCH_FINAL_LIMIT`
- **Safety:** `ENABLE_REAL_SUBMIT` (`false` recommended in dev), `RESUME_OCR_ENABLED`
- **Integrations:** LinkedIn OAuth vars, Google OAuth vars, `GOOGLE_PLACES_API_KEY`

## Product roadmap (Phase 0 → 3)

### Phase 0 (foundations) — complete
- Architecture + wireframes + scaffolding + baseline CI/dev setup.

### Phase 1 (MVP: get a job) — complete
- [x] `1.0` Ingestion/tracking: shipped.
- [x] `1.1` Sourcing/matching: shipped with board targeting + validation.
- [x] `1.2` Tailored docs/QA: shipped with provider fallback.
- [x] `1.3` Human-in-loop apply/autofill: shipped in preview/validation-first mode. Browser submission disabled by default.
- [x] `1.4` Interview prep: shipped.
- [x] `1.5` Outreach, networking & company research: shipped.
- [x] `1.6` Interview tracking & scheduling: shipped.

### Phase 2 (on-the-job assistant) — planned
- Calendar/email integrations, meeting workflows, conflict guidance, task copilots.

### Phase 3 (career growth/management) — planned
- Raise/promotion planning, hiring workflows, management/offboarding modules.

## Open backlog (pinned)

- Periodic ingest automation (scheduled run + stale-board alerts).
- Resume parser hardening for problematic PDFs.
- Document generation version UX (multi-version history and active selection).
- Interview prep advanced sort UX + interactive audio conversation mode.
- Application review contextual status surfacing per container.
- Agentic multi-page browser submission flow with user-guided confirmation.
- Rate limiting on expensive endpoints (job ingest, match generation, resume upload, Playwright submission).
- Request body validation migration to Zod across all API routes.
- God component decomposition (5 components at 700–2300 lines each).

## Changelog

See [`CHANGELOG.md`](./CHANGELOG.md).
