from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from weather_tool import get_weather


model = ChatOllama(
    model="llama3.2",
    validate_model_on_init=True,
    temperature=0.2,
)

GENERAL_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's question directly and naturally. "
    "If the user shares personal context such as their name, remember it within the active session when memory is enabled. "
    "Do not mention tools or weather unless the user is asking about weather."
)

WEATHER_AGENT_PROMPT = (
    "You are a helpful assistant. "
    "Call get_weather only for weather-related questions that include or imply a city. "
    "For non-weather questions, answer directly without tools. "
    "If the user asks a compound question, answer every part of it. "
    "Use remembered conversation context when it is available, such as the user's name. "
    "If the weather tool has no data for a city, say so clearly while still answering any other parts of the question."
)

checkpointer = InMemorySaver()
weather_agent = create_agent(
    model=model,
    tools=[get_weather],
    checkpointer=checkpointer,
    system_prompt=WEATHER_AGENT_PROMPT,
)


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "".join(chunks).strip()

    return str(content)


def _is_weather_query(user_query: str) -> bool:
    text = user_query.lower()
    weather_keywords = (
        "weather",
        "temperature",
        "forecast",
        "rain",
        "raining",
        "sunny",
        "cloudy",
        "humidity",
        "wind",
        "hot",
        "cold",
    )
    return any(keyword in text for keyword in weather_keywords)


def _asks_for_name(user_query: str) -> bool:
    text = user_query.lower()
    return "what is my name" in text or "my name" in text


def _extract_user_name(memory_history: list[dict[str, str]] | None) -> str | None:
    for item in reversed(memory_history or []):
        if item.get("role") != "human":
            continue

        content = item.get("content", "")
        lower_content = content.lower()
        marker = "my name is "
        if marker not in lower_content:
            continue

        start_index = lower_content.index(marker) + len(marker)
        extracted = content[start_index:].strip().strip(".?!, ")
        if extracted:
            return extracted

    return None


def _build_messages(
    user_query: str,
    memory_history: list[dict[str, str]] | None,
    system_prompt: str,
) -> list:
    messages = [SystemMessage(content=system_prompt)]

    for item in memory_history or []:
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "ai":
            messages.append(AIMessage(content=item["content"]))

    messages.append(HumanMessage(content=user_query))
    return messages


def _append_history(
    memory_enabled: bool,
    memory_history: list[dict[str, str]] | None,
    user_query: str,
    final_answer: str,
) -> list[dict[str, str]]:
    if not memory_enabled:
        return []

    updated_history = list(memory_history or [])
    updated_history.append({"role": "human", "content": user_query})
    updated_history.append({"role": "ai", "content": final_answer})
    return updated_history


def format_memory(
    memory_enabled: bool, memory_history: list[dict[str, str]] | None
) -> str:
    if not memory_enabled:
        return "Memory is disabled."

    if not memory_history:
        return "No information stored in memory yet."

    lines = []
    for item in memory_history:
        role = item.get("role", "unknown").capitalize()
        content = item.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def run_chat(user_query: str) -> str:
    if not user_query or not user_query.strip():
        return "Please enter a message."

    if _is_weather_query(user_query):
        response = weather_agent.invoke(
            {
                "messages": _build_messages(
                    user_query,
                    [],
                    WEATHER_AGENT_PROMPT,
                )
            },
            config={"configurable": {"thread_id": str(uuid4())}},
        )

        for message in reversed(response["messages"]):
            if getattr(message, "type", "") == "ai" and not getattr(
                message, "tool_calls", None
            ):
                return _message_content_to_text(message.content)

        return "I could not generate a response."

    response = model.invoke(_build_messages(user_query, [], GENERAL_PROMPT))
    return _message_content_to_text(response.content)


def run_chat_with_audit(
    user_query: str,
    show_audit: bool,
    memory_enabled: bool,
    session_state: dict | None,
) -> tuple[str, str, str, dict]:
    session_state = dict(session_state or {})
    thread_id = session_state.get("thread_id")
    memory_history = session_state.get("memory_history", [])

    if memory_enabled and not thread_id:
        thread_id = str(uuid4())
    if not memory_enabled:
        thread_id = str(uuid4())
        memory_history = []

    if not user_query or not user_query.strip():
        session_state["thread_id"] = thread_id
        session_state["memory_history"] = list(memory_history or [])
        memory_text = format_memory(memory_enabled, memory_history)
        return "Please enter a message.", "", memory_text, session_state

    audit_entries = [f"User query: {user_query}"]
    any_tool_called = False
    final_answer = "I could not generate a response."
    weather_result = None
    weather_city = None

    if not _is_weather_query(user_query):
        response = model.invoke(
            _build_messages(
                user_query,
                memory_history if memory_enabled else [],
                GENERAL_PROMPT,
            )
        )
        final_answer = _message_content_to_text(response.content)
        audit_entries.append("Tool called: no")
        audit_text = "\n".join(audit_entries) if show_audit else ""
        updated_history = _append_history(
            memory_enabled, memory_history, user_query, final_answer
        )
        session_state["thread_id"] = thread_id
        session_state["memory_history"] = updated_history
        memory_text = format_memory(memory_enabled, updated_history)
        return final_answer, audit_text, memory_text, session_state

    events = weather_agent.stream(
        {
            "messages": _build_messages(
                user_query,
                memory_history if memory_enabled else [],
                WEATHER_AGENT_PROMPT,
            )
        },
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    )

    for event in events:
        for update in event.values():
            for message in update.get("messages", []):
                tool_calls = getattr(message, "tool_calls", None) or []
                if tool_calls:
                    for tool_call in tool_calls:
                        any_tool_called = True
                        tool_args = tool_call.get("args", {})
                        audit_entries.append(
                            f"Tool requested: {tool_call.get('name')} | args: {tool_args}"
                        )
                        if tool_call.get("name") == "get_weather":
                            weather_city = tool_args.get("city")

                if getattr(message, "type", "") == "tool":
                    any_tool_called = True
                    weather_result = _message_content_to_text(message.content)
                    audit_entries.append(
                        f"Tool executed: yes | result: {weather_result}"
                    )

                if getattr(message, "type", "") == "ai" and not tool_calls:
                    final_answer = _message_content_to_text(message.content)

    if _asks_for_name(user_query):
        remembered_name = _extract_user_name(memory_history if memory_enabled else [])
        if remembered_name:
            if weather_result == "Weather data unavailable for this city.":
                location = weather_city or "that city"
                final_answer = (
                    f"Your name is {remembered_name}. "
                    f"I couldn't get weather data for {location}."
                )
            elif weather_result and weather_city:
                final_answer = (
                    f"Your name is {remembered_name}. "
                    f"The weather in {weather_city} is {weather_result}."
                )
            elif not _is_weather_query(user_query):
                final_answer = f"Your name is {remembered_name}."
        elif weather_result == "Weather data unavailable for this city.":
            location = weather_city or "that city"
            final_answer = (
                "You haven't told me your name yet. "
                f"I couldn't get weather data for {location}."
            )

    audit_entries.append(f"Tool called: {'yes' if any_tool_called else 'no'}")
    audit_text = "\n".join(audit_entries) if show_audit else ""
    updated_history = _append_history(
        memory_enabled, memory_history, user_query, final_answer
    )
    session_state["thread_id"] = thread_id
    session_state["memory_history"] = updated_history
    memory_text = format_memory(memory_enabled, updated_history)
    return final_answer, audit_text, memory_text, session_state