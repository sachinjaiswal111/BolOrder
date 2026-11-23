"""
AgentLLMService — the ONE place that talks to the LLM for the voice-order agent.

agent_node.py calls this service; it must not create its own LLM client.

Supported providers (set MODEL_PROVIDER in .env):
    OpenAI-compatible chat-completions API (one code path):
      - "openai"     : OpenAI            (gpt-4o-mini, gpt-4o, ...)
      - "openrouter" : OpenRouter        (openai/gpt-4o-mini, ...)
      - "groq"       : Groq              (llama-3.3-70b-versatile, ...)
      - "gemini"     : Google Gemini     (via Google's OpenAI-compatible endpoint)
    Anthropic Messages API (native SDK, its own code path):
      - "claude"     : Anthropic Claude  (claude-sonnet-5-5, claude-haiku-5-5, claude-opus-5-5, ...)
                       "anthropic" is accepted as an alias.

Auto-detection: an API_KEY starting with "sk-or-" is treated as OpenRouter even if
MODEL_PROVIDER says otherwise.

The service forces exactly one tool call per turn and always returns
{ "tool_name": str, "arguments": dict } — it never raises.
"""

import json
from functools import lru_cache

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_OPENAI_COMPATIBLE_BASE_URLS = {
    "openai":     None,  # SDK default (api.openai.com)
    "openrouter": "https://openrouter.ai/api/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai/",
}
_CLAUDE_NAMES = {"claude", "anthropic"}

LLM_TIMEOUT_SECONDS    = 20
CLAUDE_TIMEOUT_SECONDS = 30   # Claude 5.x models think before answering
LLM_MAX_RETRIES        = 1
LLM_TEMPERATURE        = 0.1
CLAUDE_MAX_TOKENS      = 8000  # includes thinking tokens; the tool call itself is small

# Claude 5.5 models reject tool_choice "any"/"tool" and non-default temperature
# with a 400, so for Claude we use tool_choice "auto" and steer with the prompt.
_CLAUDE_TOOL_INSTRUCTION = (
    "\n\nYou MUST respond by calling exactly one of the provided tools. "
    "Never answer with plain text."
)


class AgentLLMService:
    """Runs one turn of the tool-calling conversation agent (async)."""

    def __init__(self):
        provider = settings.MODEL_PROVIDER.lower()
        if settings.API_KEY.startswith("sk-or-"):
            provider = "openrouter"
        if provider in _CLAUDE_NAMES:
            provider = "claude"

        if provider != "claude" and provider not in _OPENAI_COMPATIBLE_BASE_URLS:
            raise ValueError(
                f"Unknown MODEL_PROVIDER='{provider}'. Use one of: "
                f"{', '.join(list(_OPENAI_COMPATIBLE_BASE_URLS) + ['claude'])}."
            )

        self._provider = provider
        self._model    = settings.MODEL_NAME

        if provider == "claude":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(
                api_key=settings.API_KEY,
                timeout=CLAUDE_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        else:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=settings.API_KEY,
                base_url=_OPENAI_COMPATIBLE_BASE_URLS[provider],
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        logger.info(f"AgentLLMService ready | provider={self._provider} | model={self._model}")

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Send the full message list (system + state + catalog + history + user)
        with the agent tools and get exactly one tool call back.

        Always returns. Never raises.
        Returns: { "tool_name": str, "arguments": dict }
        """
        try:
            logger.info(
                f"Agent LLM call | provider={self._provider} | model={self._model} | "
                f"messages={len(messages)}"
            )
            if self._provider == "claude":
                result = await self._run_claude(messages, tools)
            else:
                result = await self._run_openai_compatible(messages, tools)

            logger.info(
                f"Agent tool selected: {str(result['tool_name'])[:60]} | "
                f"args_keys={list(result['arguments'].keys())}"
            )
            return result

        except Exception as e:
            err = str(e)[:300].encode("ascii", "replace").decode("ascii")
            logger.error(f"Agent LLM failed [{type(e).__name__}]: {err}")
            return self._fallback()

    # ── OpenAI-compatible providers ───────────────────────────────────────────

    async def _run_openai_compatible(self, messages: list[dict], tools: list[dict]) -> dict:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=LLM_TEMPERATURE,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            logger.warning("LLM returned no tool call — falling back to respondMessage")
            return self._text_reply(message.content)

        tool_call = message.tool_calls[0]
        try:
            arguments = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"LLM sent invalid JSON arguments for tool {tool_call.function.name}")
            arguments = {}
        return {"tool_name": tool_call.function.name, "arguments": arguments}

    # ── Anthropic Claude ──────────────────────────────────────────────────────

    async def _run_claude(self, messages: list[dict], tools: list[dict]) -> dict:
        system, claude_messages = self._to_claude_messages(messages)
        claude_tools = [
            {
                "name":         t["function"]["name"],
                "description":  t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

        # No temperature / top_p (400 on Claude 5.5 models) and no forced tool_choice
        # (400 on Opus 5.5 / Sonnet 5.5): tool_choice "auto" + the system instruction.
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system + _CLAUDE_TOOL_INSTRUCTION,
            messages=claude_messages,
            tools=claude_tools,
            tool_choice={"type": "auto"},
        )

        if response.stop_reason == "refusal":
            logger.warning("Claude refused this request (stop_reason=refusal)")
            return self._fallback()

        for block in response.content:
            if block.type == "tool_use":
                return {"tool_name": block.name, "arguments": dict(block.input or {})}

        logger.warning("Claude returned no tool call — falling back to respondMessage")
        text = next((b.text for b in response.content if b.type == "text"), None)
        return self._text_reply(text)

    @staticmethod
    def _to_claude_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """
        Convert the OpenAI-style list (several system messages + history) to
        Claude's shape: one `system` string + user/assistant messages that
        start with a user turn. Consecutive same-role messages are allowed
        by the API (it combines them).
        """
        system_parts: list[str] = []
        chat: list[dict] = []
        for m in messages:
            role, content = m.get("role"), (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                chat.append({"role": role, "content": content})

        # History can be truncated mid-conversation and start with an assistant turn.
        while chat and chat[0]["role"] != "user":
            chat.pop(0)
        if not chat:
            chat = [{"role": "user", "content": "(no input)"}]

        return "\n\n".join(system_parts), chat

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _text_reply(text) -> dict:
        return {
            "tool_name": "respondMessage",
            "arguments": {
                "message_en":    text or "I didn't understand. Please repeat.",
                "message_local": "Samajh nahi aaya. Dobara boliye.",
            },
        }

    @staticmethod
    def _fallback() -> dict:
        """Safe default when the LLM call fails — never crashes the graph."""
        return {
            "tool_name": "respondMessage",
            "arguments": {
                "message_en":    "Sorry, I'm having trouble. Please try again.",
                "message_local": "Kuch dikkat aa rahi hai. Dobara try kariye.",
            },
        }


@lru_cache(maxsize=1)
def get_agent_llm() -> AgentLLMService:
    """One shared service (and HTTP client) for the whole process."""
    return AgentLLMService()
