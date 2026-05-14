## Plan: SHL Conversational Recommender

Build a FastAPI service that turns the provided SHL catalog plus sample conversation patterns into a conversational assessment recommender. The core behavior should mirror the examples in the workspace: ask targeted clarification questions when required inputs are missing, then return a strict JSON object with `reply`, `recommendations`, and `end_of_conversation`. The implementation should use Python with a hybrid rules + LLM approach, call OpenRouter with `meta-llama/llama-3-8b-instruct` for LLM access, use LangChain for orchestration, support semantic retrieval with FAISS, and be validated against the supplied sample conversations.

**Steps**
1. Define the product schema and recommendation taxonomy from the catalog and examples. Normalize the JSON fields in `shl_product_catalog.json` into a clean internal model with product name, category keys, duration, languages, job levels, adaptive/remote flags, and description. Explicitly map catalog category data to the response `test_type` codes used by the evaluator, such as `K` for knowledge and `P` for personality, and preserve multi-code cases where the catalog supports them. Derive an intent taxonomy from the sample conversations in `GenAI_SampleConversations/*.md` so the system can map role type, seniority, geography/language, assessment goals, and screening stage to product filters.
2. Build the retrieval and extraction layer. Use a FAISS-backed semantic index over the catalog for catalog search and supporting context retrieval, with LangChain as the orchestration layer. Use an LLM via OpenRouter or structured prompting to convert a user message into candidate attributes, then apply rules to detect missing high-impact fields such as role family, seniority, language, location, and assessment goal. When the user asks a compare question such as “what’s the difference between X and Y?”, retrieve both catalog items from the index and answer using catalog data only, without relying on LLM priors. When critical fields are missing, ask one focused follow-up question rather than recommending too early.
3. Implement the ranking engine. Score products by relevance to role family, key coverage, seniority fit, duration, language availability, and assessment type mix. Prefer direct matches, then adjacent matches with explicit rationale, and when the catalog has no exact product return the closest catalog items only, with a caveat in the reply rather than inventing any new recommendation. Preserve the ability to return a shortlist rather than a single item when the use case needs a battery.
4. Add conversation orchestration in FastAPI. Expose `GET /health` as a readiness check and `POST /chat` as the core endpoint that accepts the full conversation history on every call and returns the agent reply plus a recommended-assessments shortlist. Keep the service stateless across requests; do not persist server-side session state. Format responses as strict JSON only, with no markdown table output, so the evaluator can parse them reliably. Support “confirm”, “remove”, and “replace” style follow-ups so the shortlist can be updated incrementally within the supplied history.
5. Add evaluation fixtures and regression checks. Convert the provided conversations into test cases that verify question-asking behavior, shortlist composition, locked-in final outputs, off-topic refusal behavior, compare/explain behavior, and prompt-injection resistance. Include at least one test for each major scenario visible in the examples: leadership, technical hiring, safety-critical roles, graduate batteries, language-constrained roles, and office productivity screening.
6. Package and document the service for Render deployment. Add setup instructions, environment variable documentation for OpenRouter, API examples, and a short note on scope boundaries so it is clear where the catalog-based recommender stops and where external models or human review are still needed. Account for Render cold starts by allowing the first `GET /health` call up to two minutes, while keeping normal request latency comfortably within the 30-second per-call limit.
7. Define turn-management and refusal behavior explicitly. Commit to a shortlist by turn 7 at the latest even if context is still incomplete, using the best available evidence and clearly signaling uncertainty in the reply. Set `end_of_conversation` to true when the agent has returned a committed shortlist or when the interaction has clearly resolved; otherwise keep it false while clarification is still needed. Refuse off-topic questions, legal/compliance interpretations, and prompt-injection attempts with a brief boundary-setting response that preserves the JSON schema.

**Relevant files**
- `shl_product_catalog.json` - primary catalog source for the product model, retrieval index, and ranking features.
- `GenAI_SampleConversations/C1.md` - leadership-selection interaction pattern and clarification style.
- `GenAI_SampleConversations/C2.md` - technical hiring flow with reasoning test and personality defaults.
- `GenAI_SampleConversations/C3.md` - language-sensitive screening flow that requires clarification before recommending.
- `GenAI_SampleConversations/C5.md` - sales-role recommendation pattern and explanation of report vs assessment distinctions.
- `GenAI_SampleConversations/C6.md` - safety-critical role pattern and knowledge vs personality tradeoff.
- `GenAI_SampleConversations/C10.md` - shortlist refinement flow with removals and replacements.

**Verification**
1. Run regression tests on the extracted sample conversations to confirm the service asks the same kind of clarifying question and returns the same recommendation shapes.
2. Validate the recommendation output on a small set of synthetic prompts covering leadership, technical, compliance, language, and productivity scenarios.
3. Check the API contract with a minimal FastAPI smoke test to ensure stateless `GET /health`, `POST /chat`, and shortlist updates work across multiple turns.
4. Review a manual sample of responses to confirm the JSON schema, turn-cap handling, refusal behavior, and Render deployment assumptions.

**Decisions**
- Use a hybrid rules + LLM design so deterministic catalog matching handles product selection while the model handles intent extraction and conversational phrasing.
- Optimize for a conversational shortlist workflow, not a one-shot search page.
- Use OpenRouter for LLM access and FAISS for semantic catalog search.
- Use LangChain as the single orchestration framework to keep the implementation simpler.
- Treat the provided catalog JSON and sample conversations as the primary source of truth, while allowing external data or tools only if they improve extraction or ranking without changing the catalog facts.
- The PDF brief could not be extracted with the available workspace tools, so this plan is anchored to the workspace files and the user-confirmed scope: FastAPI, Python, OpenRouter, Render, FAISS, LangChain/LlamaIndex, and the provided data.

**Further Considerations**
1. If you want, the ranking layer can be made fully deterministic first and the LLM used only for question generation and summarization.