"""AI prompts for the compile and ask loops — provider-agnostic."""

from __future__ import annotations

# ---- vm ask prompts ----

ASK_SYSTEM_PROMPT = """You are VaultMind's query synthesis engine. You answer questions by synthesizing information from the provided wiki articles and raw sources. You must ground every claim in the input materials. Never invent facts, URLs, or conclusions not supported by the sources."""

ASK_USER_PROMPT = """Answer the question below using only the provided wiki articles and raw sources. If the materials do not contain enough information to fully answer the question, say so clearly and identify what additional knowledge would be needed.

Return ONLY valid JSON in this exact shape:
{{"answer": "your comprehensive answer in paragraph form", "gaps": ["what's still unknown or would need follow-up research"]}}

Question: {question}

Context:
{context}
"""

ASK_SELF_ASSESS_PROMPT = """Given your answer and the question, identify knowledge gaps — areas where the provided sources do not fully address what is being asked. Return only the gaps.

Return JSON:
{{"gaps": ["gap description 1", "gap description 2"]}}

Question: {question}
Answer: {answer}
"""


# ---- vm compile prompts ----

COMPILE_CONCEPT_TRIAGE_PROMPT = """You are a librarian organizing a research wiki.

Given the following RAW source documents (these are the original texts, NOT AI summaries), identify the key concepts they introduce or substantially advance.

Existing wiki concepts:
{existing_concepts}

For each concept:
- Determine if it is [NEW], [EXISTING: concept-slug], or [MERGE: concept-slug]
- Provide a one-line description of the concept
- List the source URLs that inform this concept
- If EXISTING or MERGE, use the exact concept slug provided

Respond ONLY with valid JSON in this exact shape:
{{"concepts": [{{"name": "concept name", "status": "new|existing:slug|merge:slug", "description": "one-line description", "source_urls": ["url1", "url2"], "merge_target": "slug if MERGE, else null"}}]}}

Rules:
- Slugs must be lowercase, hyphenated (e.g. "attention-mechanisms")
- Prefer EXISTING when a current concept already covers the source material
- Do not invent URLs — only use the source URLs provided below
- If multiple sources cover the same concept, mark them MERGE with the most representative slug
- Be conservative: only create a new concept if it genuinely warrants its own article
- READ the full content of each source before assigning a concept

Sources:
{new_sources}
"""


COMPILE_ARTICLE_CREATE_PROMPT = """Write a new wiki article for the concept "{concept_name}".

Description: {description}
Sources to synthesize: {source_urls}

The article should:
- Be 400-800 words
- Have clear sections: Overview, Key Ideas, Sources
- Include a Sources section with full URLs, listed in order of importance
- Be written in an encyclopedic but accessible tone
- Use wikilinks for related concepts you know about (e.g. [[attention-mechanisms]])
- Do NOT repeat exact quotes from sources — synthesize in your own words

Write the article in full markdown. No frontmatter. No code blocks unless showing code.
"""


COMPILE_ARTICLE_UPDATE_PROMPT = """You are maintaining a research wiki. Update the existing wiki article to incorporate new information from the listed sources.

Existing article:
---
{existing_content}
---

New sources to incorporate:
{new_sources}

Rules:
- Update the article to incorporate all new information
- Maintain consistent structure and tone with the existing article
- Add new backlinks to related concepts where appropriate (e.g. [[attention-mechanisms]])
- Be 400-800 words unless the new sources substantially expand the topic
- Keep the Sources section updated — add new URLs at the bottom
- Do NOT repeat exact quotes — synthesize in your own words

Write the updated article in full markdown. No frontmatter.
"""


COMPILE_INDEX_REBUILD_PROMPT = """You are maintaining a wiki index. Rebuild the master index file based on the current state of the wiki.

Existing index (for reference):
---
{existing_index}
---

Current wiki articles:
{article_summaries}

Rules:
- Keep the index concise — one line per concept with a brief description
- Note any orphan concepts (concepts with no links from other articles)
- Keep the structure clean and alphabetically ordered
- Use Obsidian wikilink syntax for article links: [[slug|display text]]
- IMPORTANT: Preserve wikilinks as [[...]] not **bold text**

Write the updated index in full markdown. No frontmatter.
"""

COMPILE_CONCEPT_DEDUP_PROMPT = """You are a librarian deduplicating a concept list. Given a list of concepts identified from raw sources, merge overlapping or near-duplicate concepts into a single canonical entry.

Concepts to deduplicate:
{concepts}

Rules:
- Merge concepts that cover the same or substantially overlapping territory
- When merging, pick the most descriptive name as the canonical name
- Combine descriptions into one coherent summary
- Combine all source_urls from merged entries
- Preserve status whenever a merged concept includes an existing or merge target
- Use status exactly as "new", "existing:slug", or "merge:slug"
- Keep genuinely distinct concepts separate
- Return ALL concepts (merged and unique) in the same JSON format

Return ONLY valid JSON:
{{"concepts": [{{"name": "canonical name", "status": "new|existing:slug|merge:slug", "description": "combined description", "source_urls": ["url1", "url2"], "merge_target": "slug if merge/existing, else null"}}]}}
"""
