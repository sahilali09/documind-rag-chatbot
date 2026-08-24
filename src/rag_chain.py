from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Generator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from src import vectorstore as vectorstore_module
from src.exceptions import RetrievalError


CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given the conversation history and a follow-up user question, rephrase "
    "the follow-up question into a standalone question that can be understood "
    "without the conversation history. Do NOT answer the question. "
    "If the question is already standalone, return it unchanged."
)

QA_CITATION_INSTRUCTIONS = (
    "Answer ONLY from the provided context. "
    "Use inline citations in exactly this format: [1], [2], [3]. "
    "The numbers refer only to the source labels in the provided context. "
    "Never generate citations such as 【1†L1-L4】, [1†L1-L4], (Source 1), "
    "Markdown links, URLs, or invented citation formats. "
    "Do not cite information that is not supported by the provided context. "
    "When a claim is supported by multiple sources, cite each relevant "
    "source, for example [1][3]."
)

ABSTENTION_MESSAGE = (
    "I couldn't find enough relevant information in the provided documents "
    "to answer that question confidently."
)


def trim_history(
    chat_history: list[BaseMessage],
    max_turns: int | None = None,
) -> list[BaseMessage]:
    """Keep the most recent complete conversation turns."""
    effective_turns = (
        config.MAX_HISTORY_TURNS
        if max_turns is None
        else max_turns
    )

    if effective_turns < 0:
        raise RetrievalError(
            f"max_turns must be >= 0, got {effective_turns}."
        )

    if effective_turns == 0 or not chat_history:
        return []

    max_messages = effective_turns * 2
    trimmed = chat_history[-max_messages:]

    if len(trimmed) % 2 != 0:
        trimmed = trimmed[1:]

    return trimmed


@dataclass
class RetrievalDiagnostics:
    """Diagnostics describing one retrieval and generation cycle."""

    original_question: str
    standalone_question: str
    retrieved_count: int
    scores: list[float] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float | None = None
    abstained: bool = False
    score_threshold: float | None = None


class ConversationalRAGChain:
    """Multi-turn RAG chain with hybrid retrieval and abstention."""

    def __init__(
        self,
        vectorstore: Chroma,
        llm: BaseChatModel,
        k: int | None = None,
        score_threshold: float | None = ...,
        system_prompt: str | None = None,
    ):
        if vectorstore is None:
            raise RetrievalError("A vector store is required.")

        if llm is None:
            raise RetrievalError("An LLM is required.")

        self.vectorstore = vectorstore
        self.llm = llm

        self.k = (
            config.RETRIEVER_K
            if k is None
            else k
        )

        if self.k <= 0:
            raise RetrievalError(
                f"Retriever k must be > 0, got {self.k}."
            )

        self.score_threshold = (
            config.RETRIEVAL_SCORE_THRESHOLD
            if score_threshold is ...
            else score_threshold
        )

        if self.score_threshold is not None and not (
            -1.0 <= self.score_threshold <= 1.0
        ):
            raise RetrievalError(
                "score_threshold must be between -1 and 1, "
                "or None to disable abstention."
            )

        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else config.SYSTEM_PROMPT
        ).strip()

        if not self.system_prompt:
            raise RetrievalError("system_prompt must not be empty.")

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self._contextualize_chain = (
            contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        qa_system_prompt = (
            f"{self.system_prompt}\n\n"
            f"{QA_CITATION_INSTRUCTIONS}\n\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self._qa_chain = (
            qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def _validate_question(self, question: str) -> str:
        if not isinstance(question, str):
            raise RetrievalError("Question must be a string.")

        question = question.strip()

        if not question:
            raise RetrievalError("Question must not be empty.")

        return question

    def _reformulate_question(
        self,
        question: str,
        chat_history: list[BaseMessage],
    ) -> str:
        """Convert a follow-up question into a standalone retrieval query."""
        if not chat_history:
            return question

        try:
            standalone = self._contextualize_chain.invoke(
                {
                    "input": question,
                    "chat_history": chat_history,
                }
            ).strip()
        except Exception as exc:
            raise RetrievalError(
                f"Question reformulation failed: {exc}"
            ) from exc

        if not standalone:
            raise RetrievalError(
                "Question reformulation returned an empty query."
            )

        return standalone

    def _retrieve_with_scores(
        self,
        query: str,
    ) -> tuple[list[Document], list[float]]:
        try:
            results = vectorstore_module.hybrid_search_with_relevance(
                self.vectorstore,
                query,
                k=self.k,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Hybrid document retrieval failed: {exc}"
            ) from exc

        documents = [document for document, _, _ in results]

        semantic_scores = [
            float(semantic_score)
            for _, _, semantic_score in results
            if semantic_score >= -1.0
        ]

        return documents, semantic_scores

    def _should_abstain(
        self,
        scores: list[float],
    ) -> bool:
        if self.score_threshold is None:
            return False

        if not scores:
            return True

        return max(scores) < self.score_threshold

    @staticmethod
    def _format_context(
        documents: list[Document],
    ) -> str:
        """Format retrieved chunks with stable source numbers."""
        blocks: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            label = (
                document.metadata.get("display_source")
                or document.metadata.get("source")
                or f"source {index}"
            )

            content = document.page_content.strip()

            if not content:
                continue

            blocks.append(
                f"[{index}] ({label})\n{content}"
            )

        return (
            "\n\n".join(blocks)
            if blocks
            else "No relevant context was found."
        )

    @staticmethod
    def _clean_generated_citations(text: str) -> str:
        """Normalize unsupported citation formats to the source-number form."""
        text = re.sub(
            r"【\s*(\d+)\s*†[^】]*】",
            r"[\1]",
            text,
        )

        text = re.sub(
            r"\[\s*(\d+)\s*†[^\]]*\]",
            r"[\1]",
            text,
        )

        return text

    def _retrieve_and_diagnose(
        self,
        question: str,
        chat_history: list[BaseMessage],
    ) -> tuple[list[Document], RetrievalDiagnostics]:
        start = time.perf_counter()

        standalone_question = self._reformulate_question(
            question,
            chat_history,
        )

        retrieved_docs, scores = self._retrieve_with_scores(
            standalone_question
        )

        retrieval_latency_ms = (
            time.perf_counter() - start
        ) * 1000

        diagnostics = RetrievalDiagnostics(
            original_question=question,
            standalone_question=standalone_question,
            retrieved_count=len(retrieved_docs),
            scores=scores,
            retrieval_latency_ms=retrieval_latency_ms,
            score_threshold=self.score_threshold,
        )

        return retrieved_docs, diagnostics

    def invoke(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> dict:
        """Retrieve context and generate one complete answer."""
        question = self._validate_question(question)

        history = trim_history(
            chat_history or []
        )

        retrieved_docs, diagnostics = (
            self._retrieve_and_diagnose(
                question,
                history,
            )
        )

        if self._should_abstain(
            diagnostics.scores
        ):
            diagnostics.abstained = True

            return {
                "answer": ABSTENTION_MESSAGE,
                "source_documents": [],
                "standalone_question": diagnostics.standalone_question,
                "diagnostics": diagnostics,
            }

        context = self._format_context(
            retrieved_docs
        )

        start = time.perf_counter()

        try:
            answer = self._qa_chain.invoke(
                {
                    "input": question,
                    "chat_history": history,
                    "context": context,
                }
            )
            answer = self._clean_generated_citations(answer)
        except Exception as exc:
            raise RetrievalError(
                f"Answer generation failed: {exc}"
            ) from exc
        finally:
            diagnostics.generation_latency_ms = (
                time.perf_counter() - start
            ) * 1000

        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "standalone_question": diagnostics.standalone_question,
            "diagnostics": diagnostics,
        }

    def stream(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> tuple[
        Generator[str, None, None],
        list[Document],
        RetrievalDiagnostics,
    ]:
        """Retrieve eagerly and stream only the generated answer."""
        question = self._validate_question(question)

        history = trim_history(
            chat_history or []
        )

        retrieved_docs, diagnostics = (
            self._retrieve_and_diagnose(
                question,
                history,
            )
        )

        if self._should_abstain(
            diagnostics.scores
        ):
            diagnostics.abstained = True

            def abstain_stream():
                yield ABSTENTION_MESSAGE

            return (
                abstain_stream(),
                [],
                diagnostics,
            )

        context = self._format_context(
            retrieved_docs
        )

        try:
            raw_stream = self._qa_chain.stream(
                {
                    "input": question,
                    "chat_history": history,
                    "context": context,
                }
            )
        except Exception as exc:
            raise RetrievalError(
                f"Answer streaming failed: {exc}"
            ) from exc

        def timed_stream():
            start = time.perf_counter()
            buffer: list[str] = []

            try:
                for token in raw_stream:
                    buffer.append(token)

                answer = self._clean_generated_citations(
                    "".join(buffer)
                )

                yield answer
            except Exception as exc:
                raise RetrievalError(
                    f"Answer streaming failed: {exc}"
                ) from exc
            finally:
                diagnostics.generation_latency_ms = (
                    time.perf_counter() - start
                ) * 1000

        return (
            timed_stream(),
            retrieved_docs,
            diagnostics,
        )