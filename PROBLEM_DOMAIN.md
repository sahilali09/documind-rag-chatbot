# Problem Domain — DocuMind

## The use case

Organizations and students often have knowledge distributed across PDFs, handbooks, lecture notes, policies, and web pages. Finding a specific fact can require manually searching multiple sources.

A generic LLM may not have access to these private or recently updated documents and may generate answers that are not grounded in the user's actual sources.

**DocuMind** addresses this by retrieving passages from documents explicitly provided by the user and supplying those passages to an LLM for answer generation.

Supported sources include:

- PDF documents
- TXT files
- Markdown files
- Web pages

The system is designed to produce answers that are:

- **Grounded** — generated with retrieved document passages as context. This reduces reliance on unsupported model knowledge but does not guarantee that every generated statement is correct.
- **Source-aware** — retrieved chunks are displayed with available source metadata, such as filename and page number or web source.
- **Conversational** — follow-up questions can be reformulated using previous conversation history before retrieval.
- **Abstention-capable** — when retrieved relevance is below the configured threshold, the system can abstain instead of generating from weak context. This is a configurable heuristic, not a perfect relevance classifier.

## Representative scenarios

- A student uploads lecture material and textbook chapters and asks study questions.
- A new employee provides company handbooks and onboarding documents and asks policy questions.
- A researcher provides several papers and asks questions about their contents.
- A user provides product documentation URLs and asks integration-related questions.

## Why RAG instead of only using a larger LLM

- **Freshness** — the indexed knowledge base can contain documents that are newer than the model's training data or were never publicly available.
- **Verifiability** — retrieved source chunks allow users to inspect the information used as context.
- **Cost and flexibility** — new documents can be indexed without fine-tuning the LLM, and the LLM and embedding providers can be changed independently.
- **Scope control** — retrieval provides domain-specific context to the generation step, which can reduce unsupported answers.