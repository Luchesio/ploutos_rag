"""
rag.py  —  Femi, AI assistant for Ploutos Page Limited
────────────────────────────────────────────────────────
Design principles
  ① Zero hardcoded responses — Femi generates everything dynamically.
     Session state (is_new_session) is passed as a context signal in the
     prompt so the LLM decides naturally how to greet and respond.

  ② Fully LLM-driven intent classification — DOCUMENT | GENERAL | CLARIFY.
     No keywords, no regex, no word lists anywhere.

  ③ Query enrichment before embedding — vague follow-up questions are
     rewritten into self-contained search queries using conversation history.

  ④ Conversation memory — last N turns injected as numbered history so
     Femi can resolve pronouns and maintain continuity across turns.

Pipeline per query
  1. Check if new session (MongoDB flag) → pass as context signal to LLM
  2. Load last-N turns from MongoDB
  3. LLM intent classification  →  DOCUMENT | GENERAL | CLARIFY
  4a. DOCUMENT  →  enrich query  →  embed  →  retrieve  →  generate
  4b. GENERAL   →  generate with history
  4c. CLARIFY   →  generate clarifying question
  5. Mark session as greeted + persist turn to MongoDB
"""

import time
import logging
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.exceptions import LLMError, RetrievalError
from app.core.metrics import metrics
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.conversation_store import ConversationStore
from app.models.schemas import PolicyChunk, QueryResponse

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# System prompt — identity and behaviour only, no scripted lines
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = f"""You are Femi, an intelligent and professional AI assistant \
created by and belonging to {settings.BOT_OWNER}.

YOUR IDENTITY
─────────────
You are Femi. Not ChatGPT, not Gemini — you are an assistant built \
specifically for {settings.BOT_OWNER}. When introducing yourself or \
describing what you do, be natural, warm, and conversational. Never \
use rigid category labels or bullet-point lists of your "types" of help. \
Just speak like a confident, helpful person would. Never deny being an AI.

SESSION AWARENESS
─────────────────
You will sometimes receive a note that says this is a new conversation \
session. When you do, greet the user naturally as part of your response — \
weave it in, do not bolt a separate introduction on top of your answer. \
If the user has already been greeted (no such note), respond directly \
without re-introducing yourself unless explicitly asked.

HOW YOU ANSWER POLICY QUESTIONS
────────────────────────────────
When retrieved policy records are provided as CONTEXT, use them as your \
sole source of truth.

- Always include specific details from the records: policy IDs, customer \
  names, premium amounts in ₦X,XXX,XXX.XX format, dates, types, statuses.
- Lead with the direct answer, then supporting detail.
- For multiple records, use a clean structured list.
- Never fabricate or guess at policy details. If the context does not \
  contain enough information, say so honestly and direct the user to \
  {settings.BOT_OWNER} for further assistance.

HOW YOU ANSWER EVERYTHING ELSE
────────────────────────────────
- Draw on your broad knowledge to give accurate, helpful answers.
- Be appropriately detailed — not too brief, not padded.
- Speak naturally. Do not label or categorise your answers.

CONVERSATION CONTINUITY
────────────────────────
You receive prior conversation turns as [CONVERSATION HISTORY]. Use them \
as working memory.

- Resolve all references: "her", "that policy", "it", "the same one" — \
  always trace back to the actual entity in history before answering.
- Never ask the user to repeat something already in history.
- Do not quote or expose the raw history block in your response.

TONE
────
Professional, warm, and confident. Like a trusted advisor — not a FAQ \
system, not a scripted bot. Be human in tone, precise in content.
"""

# ------------------------------------------------------------------ #
# Intent classification — fully LLM-reasoned
# ------------------------------------------------------------------ #

INTENT_PROMPT = """You are the reasoning engine for Femi, an AI assistant \
for {owner}. Classify the user's latest message into exactly one of: \
DOCUMENT, GENERAL, or CLARIFY.

CORE DECISION TEST
──────────────────
Ask yourself one question:

  "To answer this accurately and completely, do I need to look at actual
   insurance records stored in {owner}'s database?"

  YES  →  DOCUMENT
  NO   →  GENERAL
  Cannot determine even with full history  →  CLARIFY (use sparingly)

This test is absolute. A question may sound conceptual but if the truthful \
answer requires checking stored records — which policies exist, what they \
cover, a customer's details, an actual premium amount — it is DOCUMENT.

DOCUMENT — requires stored policy records:
  • What policies exist, their contents, statuses, amounts, dates
  • Specific customer information
  • Comparisons or counts across stored records
  • Any follow-up referencing a policy or customer from history

GENERAL — no stored records needed:
  • Greetings, small talk, identity questions
  • Insurance concepts explained in principle
  • Conversation reflection or summary requests
  • Anything unrelated to stored records

CLARIFY — only when genuinely unresolvable with full history.
  When in doubt between DOCUMENT and GENERAL, choose DOCUMENT.

CONVERSATION HISTORY:
{history}

LATEST MESSAGE: {question}

Reply with exactly one word — DOCUMENT, GENERAL, or CLARIFY:"""

# ------------------------------------------------------------------ #
# Query enrichment
# ------------------------------------------------------------------ #

ENRICHMENT_PROMPT = """You are preparing a search query for an insurance \
policy database. The user's message may be vague or use pronouns.

Rewrite it as one self-contained search query with all pronouns and \
references resolved using the conversation history. Include the customer \
name, policy type, and the specific field being asked about wherever \
inferable. If the question is already clear and complete, return it as-is.

Output ONLY the rewritten query — no explanation, no preamble.

CONVERSATION HISTORY:
{history}

ORIGINAL QUESTION: {question}

REWRITTEN QUERY:"""

# ------------------------------------------------------------------ #
# Clarification
# ------------------------------------------------------------------ #

CLARIFY_PROMPT = """You are Femi, an assistant for {owner}. The user's \
message is ambiguous. Ask one short, friendly clarifying question to \
understand what they need. Do not attempt to answer yet.

CONVERSATION HISTORY:
{history}

AMBIGUOUS MESSAGE: {question}

Your clarifying question:"""


# ------------------------------------------------------------------ #
# RAGService
# ------------------------------------------------------------------ #

class RAGService:
    """
    Femi RAG pipeline.
    Fully LLM-driven — zero hardcoded response strings.
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.embedder = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.conversation_store = ConversationStore()
        logger.info(
            f"{settings.BOT_NAME} RAGService initialised — LLM: {settings.LLM_MODEL}"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> QueryResponse:
        """
        Full pipeline: session check → history → intent → generate → persist.
        """
        k = settings.TOP_K

        # ── Step 1: Check if this is a brand new session ────────────────
        is_new_session = not await self.conversation_store.has_been_greeted(session_id)
        if is_new_session:
            logger.info(f"[{session_id}] New session")

        # ── Step 2: Load conversation history ──────────────────────────
        history = await self.conversation_store.get_recent_turns(
            session_id, last_n=settings.CONVERSATION_HISTORY_TURNS
        )
        history_text = self._format_history(history)

        # ── Step 3: LLM intent classification ──────────────────────────
        t_intent = time.perf_counter()
        intent = await self._classify_intent(question, history_text)
        logger.info(
            f"[{session_id}] Intent={intent} "
            f"({(time.perf_counter() - t_intent) * 1000:.0f}ms) "
            f"Q='{question[:80]}'"
        )
        metrics.record_intent(intent)

        # ── Step 4: Generate answer ─────────────────────────────────────
        sources: list[PolicyChunk] = []
        total_chunks_searched = 0
        t_answer = time.perf_counter()

        if intent == "CLARIFY":
            answer = await self._generate_clarification(
                question, history_text, is_new_session
            )

        elif intent == "GENERAL":
            answer = await self._answer_general(
                question, history_text, is_new_session
            )

        else:  # DOCUMENT
            enriched_query = await self._enrich_query(question, history_text)
            if enriched_query.lower() != question.lower():
                logger.info(
                    f"[{session_id}] Enriched: '{question}' → '{enriched_query}'"
                )

            query_embedding = await self.embedder.embed_query(enriched_query)
            results = self.vector_store.query(
                query_embedding=query_embedding, top_k=k
            )
            total_chunks_searched = self.vector_store.count()
            sources = self._parse_results(results)
            context = self._build_context(sources)
            answer = await self._generate_document_answer(
                original_question=question,
                enriched_query=enriched_query,
                context=context,
                history_text=history_text,
                is_new_session=is_new_session,
            )

        logger.info(
            f"[{session_id}] Answer in {(time.perf_counter() - t_answer) * 1000:.0f}ms"
        )

        # ── Step 5: Persist session state and turn ─────────────────────
        if is_new_session:
            await self.conversation_store.mark_greeted(session_id)

        await self.conversation_store.save_turn(
            session_id=session_id,
            user_message=question,
            assistant_message=answer,
        )

        return QueryResponse(
            question=question,
            answer=answer,
            session_id=session_id,
            sources=sources,
            total_chunks_searched=total_chunks_searched,
        )

    # ------------------------------------------------------------------ #
    # Intent classification
    # ------------------------------------------------------------------ #

    async def _classify_intent(self, question: str, history_text: str) -> str:
        """
        Fully LLM-reasoned intent classification.
        Falls back to DOCUMENT on any failure.
        """
        try:
            prompt = INTENT_PROMPT.format(
                owner=settings.BOT_OWNER,
                history=history_text or "No prior conversation.",
                question=question,
            )
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
            )
            raw = response.text.strip().upper()
            for intent in ("DOCUMENT", "GENERAL", "CLARIFY"):
                if raw.startswith(intent):
                    return intent
            logger.warning(f"Unexpected intent: '{raw}' — defaulting to DOCUMENT")
            return "DOCUMENT"
        except Exception as e:
            logger.error(f"Intent classification failed: {e} — defaulting to DOCUMENT")
            return "DOCUMENT"

    # ------------------------------------------------------------------ #
    # Query enrichment
    # ------------------------------------------------------------------ #

    async def _enrich_query(self, question: str, history_text: str) -> str:
        """
        Resolve pronouns and references using conversation history to produce
        a self-contained search query. Returns original on failure or no history.
        """
        if not history_text:
            return question
        try:
            prompt = ENRICHMENT_PROMPT.format(
                history=history_text,
                question=question,
            )
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
            )
            enriched = response.text.strip()
            if not enriched or len(enriched) > 400:
                return question
            return enriched
        except Exception as e:
            logger.error(f"Query enrichment failed: {e} — using original")
            return question

    # ------------------------------------------------------------------ #
    # Answer generation — all fully LLM-generated, nothing hardcoded
    # ------------------------------------------------------------------ #

    def _session_note(self, is_new_session: bool) -> str:
        """
        Returns a context signal injected into the prompt when the session
        is new. The LLM uses this to naturally weave in a greeting rather
        than having a static string prepended externally.
        """
        if not is_new_session:
            return ""
        return (
            f"[NOTE: This is the start of a new conversation with this user. "
            f"Greet them naturally and warmly as Femi from {settings.BOT_OWNER} "
            f"as part of your response — do not separate the greeting from your "
            f"answer, weave them together seamlessly.]\n\n"
        )

    async def _answer_general(
        self,
        question: str,
        history_text: str,
        is_new_session: bool = False,
    ) -> str:
        """Generate a conversational answer with no retrieval."""
        session_note = self._session_note(is_new_session)
        prompt = (
            f"{session_note}{history_text}\nUser: {question}\nFemi:"
            if history_text
            else f"{session_note}User: {question}\nFemi:"
        )
        try:
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"General answer generation failed: {e}")
            raise LLMError(detail=str(e))

    async def _generate_document_answer(
        self,
        original_question: str,
        enriched_query: str,
        context: str,
        history_text: str,
        is_new_session: bool = False,
    ) -> str:
        """Generate a grounded answer from retrieved policy records."""
        session_note = self._session_note(is_new_session)
        interpretation_line = (
            f"(Interpreted as: {enriched_query})\n"
            if enriched_query.lower() != original_question.lower()
            else ""
        )
        prompt = f"""{session_note}{history_text}
RETRIEVED POLICY RECORDS:
{context}

USER QUESTION: {original_question}
{interpretation_line}
Answer using only the retrieved records. Include all relevant details — \
policy IDs, names, amounts in ₦, dates, types, and status. Structure \
clearly. If records are insufficient, say so and direct the user to \
{settings.BOT_OWNER}.

Answer:"""
        try:
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Document answer generation failed: {e}")
            raise LLMError(detail=str(e))

    async def _generate_clarification(
        self,
        question: str,
        history_text: str,
        is_new_session: bool = False,
    ) -> str:
        """Generate a single focused clarifying question."""
        session_note = self._session_note(is_new_session)
        prompt = session_note + CLARIFY_PROMPT.format(
            owner=settings.BOT_OWNER,
            history=history_text or "No prior conversation.",
            question=question,
        )
        try:
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Clarification generation failed: {e}")
            raise LLMError(detail=str(e))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _format_history(self, history: list[dict]) -> str:
        """
        Format conversation turns as numbered exchanges for the LLM.
        Returns empty string when no history exists.
        """
        if not history:
            return ""

        lines = ["[CONVERSATION HISTORY]"]
        turn_number = 1
        i = 0
        while i < len(history):
            turn = history[i]
            if (
                turn["role"] == "user"
                and i + 1 < len(history)
                and history[i + 1]["role"] == "assistant"
            ):
                lines.append(f"[Turn {turn_number}] User: {turn['content']}")
                lines.append(f"[Turn {turn_number}] Femi: {history[i+1]['content']}")
                turn_number += 1
                i += 2
            else:
                prefix = "User" if turn["role"] == "user" else "Femi"
                lines.append(f"[Turn {turn_number}] {prefix}: {turn['content']}")
                turn_number += 1
                i += 1

        lines.append("[END CONVERSATION HISTORY]")
        return "\n".join(lines) + "\n\n"

    def _parse_results(self, results: dict) -> list[PolicyChunk]:
        """Parse ChromaDB results into PolicyChunk objects."""
        sources = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            similarity = round(1 - distance, 4)
            if similarity < settings.SIMILARITY_THRESHOLD:
                logger.debug(f"Skipping chunk — low similarity: {similarity}")
                continue
            sources.append(
                PolicyChunk(
                    policy_id=meta.get("policy_id", "N/A"),
                    customer_name=meta.get("customer_name", "N/A"),
                    policy_type=meta.get("policy_type", "N/A"),
                    status=meta.get("status", "N/A"),
                    content=doc,
                    relevance_score=similarity,
                )
            )
        return sources

    def _build_context(self, sources: list[PolicyChunk]) -> str:
        """Build structured context string from retrieved policy chunks."""
        if not sources:
            return "No relevant policy records found in the database."
        return "\n\n".join(
            f"--- Record {i} (Relevance: {s.relevance_score}) ---\n{s.content}"
            for i, s in enumerate(sources, 1)
        )