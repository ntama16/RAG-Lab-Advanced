"""All prompts in one place so they are easy to read, diff and tune."""

ROUTE = """You are the intent gate of a retrieval system. Decide how to handle the user's message.

The indexed corpus contains these documents:
{corpus}

Intents:
- chitchat: greeting, thanks, or small talk. No retrieval needed.
- lookup: asks for a specific fact, name, number, code, or exact term. Keyword matching matters.
- conceptual: asks for an explanation, summary, comparison, or "how/why". Semantic matching matters.
- multi_hop: needs facts about several entities, or how entities relate, combined across passages.
- global: asks about the corpus as a whole: main themes, an overview, "what topics are covered".

Reply with JSON:
{{"intent": "chitchat|lookup|conceptual|multi_hop|global", "needs_retrieval": true|false,
  "in_scope": true|false, "reason": "<one sentence>"}}
"in_scope" is false only when the corpus clearly cannot contain the answer.

Query: {query}"""

SELF_QUERY = """Separate the user's question into (a) the semantic search query and (b) metadata filters.

Filterable metadata fields and example values:
{schema}

Rules:
- Only use fields listed above. Only add a filter when the question explicitly constrains that field.
- Operators: eq, ne, in, gt, gte, lt, lte, contains.
- Remove the filtered constraint from the search query text. If nothing applies, return an empty filter list.

Reply with JSON: {{"query": "<search text>", "filters": [{{"field": "...", "op": "...", "value": ...}}]}}

Query: {query}"""

MULTI_QUERY = """Write {n} alternative search queries for the question below. Each should approach it from a
different angle: different vocabulary, a more specific sub-question, or a broader framing.
Reply with JSON: {{"queries": ["...", "..."]}}

Query: {query}"""

HYDE = """Write a short passage (3-5 sentences) that could appear in a reference document and would directly
answer the question. Write it as plain factual prose in the style of documentation. Do not mention the question.

Query: {query}"""

GRADE = """You are grading a retrieved passage for a retrieval system.

- relevant: contains information that directly helps answer the query.
- partial: on topic, but does not contain the needed information.
- irrelevant: off topic.

Reply with JSON: {{"grade": "relevant|partial|irrelevant", "reason": "<one short sentence>"}}

Query: {query}

Passage: {passage}"""

REWRITE = """A search for the query below returned passages that did not contain the answer.
What came back was about: {seen}

Rewrite the query so a search engine is more likely to find the answer: use more specific or alternative
terminology, expand abbreviations, and drop conversational filler.
Reply with JSON: {{"query": "..."}}

Query: {query}"""

CONTEXTUALIZE = """<document title="{title}">
{document}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Write one or two sentences that situate this chunk within the overall document, to improve search retrieval
of the chunk. Reply with only that context."""

GRAPH_EXTRACT = """Extract a knowledge graph from the passage.
- entities: named things, systems, components, people, organisations, concepts that matter. Use canonical
  singular names. At most 10.
- relations: directed (source, relation, target) facts stated in the passage. Relation is a short verb phrase.

Reply with JSON:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"source": "...", "relation": "...", "target": "..."}}]}}

Passage: {passage}"""

QUERY_ENTITIES = """List the entities (named things, systems, components, concepts) that the query is about.
Reply with JSON: {{"entities": ["..."]}}

Query: {query}"""

DOC_METADATA = """Extract metadata for the document below.
Reply with JSON using only these keys (omit a key when unknown):
{{"doc_type": "<e.g. policy, runbook, spec, report, faq, notes>", "topics": ["<up to 4 short topics>"],
  "year": <int>, "audience": "<who it is for>"}}

Title: {title}

{document}"""

GENERATE_SYSTEM = """You answer questions using only the numbered context passages provided.
- Cite every claim with the passage number in square brackets, e.g. [2] or [1][3].
- If the passages do not contain the answer, say that the indexed corpus does not cover it, and say what is missing.
  Do not fall back on outside knowledge.
- Be concise and precise."""

# Phrasings that mean "I declined to answer". Used to score abstention from the answer itself
# rather than from the retrieval grader's verdict, which is a different decision (see
# raglab.pipeline.answer_abstains). Kept beside GENERATE_SYSTEM because that prompt shapes the
# wording: change one and re-check the other. Matching is substring, on a lowercased answer with
# runs of whitespace collapsed, so entries must be lowercase and contain no double spaces.
ABSTENTION_MARKERS = (
    "does not cover",
    "doesn't cover",
    "do not cover",
    "don't cover",
    "does not contain",
    "doesn't contain",
    "do not contain",
    "don't contain",
    "no supporting evidence",
    "not covered by the indexed",
    "not in the indexed corpus",
    "no information about",
    "no relevant passages",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i don't know",
    "i do not know",
)

GENERATE_USER = """Context passages:
{context}

Question: {query}"""

DIRECT_SYSTEM = "You are a helpful assistant attached to a document retrieval lab. Reply briefly."

EVALGEN = """Write one question that can be answered from the passage alone, as a real user of this
documentation would ask it (do not refer to "the passage"). Then give the answer and the exact sentence(s)
from the passage that support it, copied verbatim.
Reply with JSON: {{"question": "...", "answer": "...", "evidence": "<verbatim quote>"}}

Passage: {passage}"""

JUDGE = """Evaluate the answer.
- faithfulness: fraction (0-1) of the answer's claims that are supported by the context.
- correctness: 0-1, how well the answer agrees with the reference answer.
Reply with JSON: {{"faithfulness": <0-1>, "correctness": <0-1>, "reason": "<one sentence>"}}

Question: {query}

Reference answer: {reference}

Context:
{context}

Answer: {answer}"""

CONDENSE = """Rewrite the follow-up message as a standalone search question, using the conversation for any
missing references (pronouns, "it", "that policy", omitted subjects). If it is already standalone, return it unchanged.
Reply with JSON: {{"query": "..."}}

Conversation:
{history}

Query: {query}"""

DECOMPOSE = """If the question asks for several distinct pieces of information, split it into the smallest set of
self-contained sub-questions (each must make sense alone, repeat the subject). If it is a single question,
return it as the only item. At most {n}.
Reply with JSON: {{"sub_questions": ["...", "..."]}}

Query: {query}"""

STEP_BACK = """Write one more general "step-back" question whose answer would provide the background principle,
policy or concept needed to answer the specific question below.
Reply with JSON: {{"query": "..."}}

Query: {query}"""

REFINE = """The passage below has been split into numbered strips. Select the strips that contain information
useful for answering the query. Keep a strip if it is needed to understand another kept strip.
Reply with JSON: {{"keep": [<strip numbers>]}}. An empty list means nothing in the passage helps.

Query: {query}

Passage: {strips}"""

VERIFY = """Check an answer against its sources.
Split the answer into its individual factual claims. For each claim decide whether the context passages support it.
- supported: a passage states it (give the passage number).
- unsupported: no passage states it, or a passage contradicts it.
Ignore statements that only say the corpus lacks information.
Reply with JSON: {{"claims": [{{"claim": "...", "verdict": "supported|unsupported", "passage": <number or null>}}]}}

Context:
{context}

Answer: {answer}"""

COMMUNITY = """The entities below form a closely connected cluster in a knowledge graph built from a document corpus.
Write a 3-5 sentence report on this cluster: what it is about, how the entities relate, and the key facts.
Start with a short title on its own line.

Entities: {entities}

Relations:
{relations}

Supporting excerpts:
{excerpts}"""

EVALGEN_PARAPHRASE = """Write one question answerable from the passage alone, as a real user would ask it, but
deliberately AVOID the passage's own key terms: use everyday wording, synonyms or a described situation instead.
Then give the answer and the exact supporting sentence(s), copied verbatim.
Reply with JSON: {{"question": "...", "answer": "...", "evidence": "<verbatim quote>"}}

Passage: {passage}"""

EVALGEN_MULTI = """Write one question that can only be answered by combining BOTH passages (each contributes a
necessary fact). Give the answer and one verbatim supporting sentence from each passage.
Reply with JSON: {{"question": "...", "answer": "...", "evidence": ["<quote from passage A>", "<quote from passage B>"]}}

Passage: A: {a}

B: {b}"""

EVALGEN_UNANSWERABLE = """Here are the titles of a document corpus and one sample passage. Write one question that
sounds like it belongs to this domain but that the corpus CANNOT answer (a plausible neighbouring topic, product
or policy that is not covered). Reply with JSON: {{"question": "..."}}

Titles: {titles}

Passage: {passage}"""
