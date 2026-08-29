import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict

import discord
from openai import OpenAI

ENV_FILE = Path(__file__).with_name(".env")


def load_env_file():
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key, sep, value = line.partition("=")
        if not sep:
            continue

        os.environ.setdefault(key.strip(), value.strip().strip("\"'").strip())


load_env_file()

# Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN") or os.getenv("token")
if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN in environment or .env file.")

# LM Studio defaults to localhost:1234. No API key required, but a placeholder is needed.
LM_STUDIO_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
BOT_NAME = "Alita"
MEMORY_FILE = Path(__file__).with_name("conversation_memory.json")
MAX_HISTORY_MESSAGES = 12
MAX_TOOL_CALL_STEPS = 4

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def load_conversation_memory():
    if not MEMORY_FILE.exists():
        return defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as memory_file:
            raw_memory = json.load(memory_file)
    except (OSError, json.JSONDecodeError):
        return defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))

    memory = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
    for channel_id, messages in raw_memory.items():
        memory[str(channel_id)] = deque(messages[-MAX_HISTORY_MESSAGES:], maxlen=MAX_HISTORY_MESSAGES)
    return memory


def save_conversation_memory(memory):
    try:
        serializable_memory = {
            channel_id: list(messages)
            for channel_id, messages in memory.items()
        }
        with MEMORY_FILE.open("w", encoding="utf-8") as memory_file:
            json.dump(serializable_memory, memory_file, ensure_ascii=True, indent=2)
    except OSError as error:
        print(f"Error saving conversation memory: {error}")


def format_conversation_history(history):
    if not history:
        return "No prior conversation history."

    return "\n".join(
        f"{entry['author']}: {entry['content']}"
        for entry in history
    )


def build_gatekeeper_context(channel_name, history, message):
    return (
        f"Channel: #{channel_name}\n"
        f"Recent conversation history:\n{format_conversation_history(history)}\n"
        f"Current message: {message.author.display_name}: {message.content}"
    )


def build_response_context(channel_name, message):
    return (
        f"Channel: #{channel_name}\n"
        f"Current message: {message.author.display_name}: {message.content}\n"
        f"If you need recent conversation history or a memory note, use the available tools instead of guessing."
    )


def build_tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_recent_channel_history",
                "description": "Return the recent conversation history for the current channel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of recent messages to return.",
                            "minimum": 1,
                            "maximum": MAX_HISTORY_MESSAGES,
                            "default": MAX_HISTORY_MESSAGES,
                        }
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_memory_note",
                "description": "Store a short memory note for the current channel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "A concise note to remember later.",
                        }
                    },
                    "required": ["note"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def execute_tool_call(tool_name: str, tool_arguments: Dict[str, Any], channel_history):
    if tool_name == "get_recent_channel_history":
        limit = tool_arguments.get("limit", MAX_HISTORY_MESSAGES)
        limit = max(1, min(int(limit), MAX_HISTORY_MESSAGES))
        recent_messages = list(channel_history)[-limit:]
        return {
            "messages": recent_messages,
            "count": len(recent_messages),
        }

    if tool_name == "save_memory_note":
        note = tool_arguments.get("note", "").strip()
        if not note:
            return {"saved": False, "error": "note cannot be empty"}

        channel_history.append(
            {
                "author": BOT_NAME,
                "content": f"Memory note: {note}",
            }
        )
        return {"saved": True, "note": note}

    return {"error": f"Unknown tool: {tool_name}"}


conversation_memory = load_conversation_memory()

def query_lm_studio(system_prompt, user_prompt, temperature=0.7):
    """Sends a completion request to the active model loaded in LM Studio."""
    try:
        completion = LM_STUDIO_CLIENT.chat.completions.create(
            # LM Studio automatically uses whatever model you currently have loaded active
            model="local-model", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            # max_tokens=150 # Keeps generation short, snappy, and fast
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with LM Studio: {e}")
        return ""


def query_lm_studio_with_tools(system_prompt, user_prompt, channel_history, temperature=0.8):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tool_definitions = build_tool_definitions()

    for _ in range(MAX_TOOL_CALL_STEPS):
        try:
            completion = LM_STUDIO_CLIENT.chat.completions.create(
                model="local-model",
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
                temperature=temperature,
            )
        except Exception as error:
            print(f"Error communicating with LM Studio: {error}")
            return ""

        assistant_message = completion.choices[0].message
        assistant_entry = assistant_message.model_dump(exclude_none=True)
        messages.append(assistant_entry)

        tool_calls = assistant_entry.get("tool_calls") or []
        if not tool_calls:
            return (assistant_entry.get("content") or "").strip()

        print(f"Tool calls requested: {tool_calls}")

        for tool_call in tool_calls:
            try:
                tool_arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                tool_arguments = {}

            tool_result = execute_tool_call(tool_call["function"]["name"], tool_arguments, channel_history)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=True),
                }
            )

    return ""

@client.event
async def on_ready():
    print(f"Successfully logged in as {client.user}!")

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself to prevent infinite chat loops
    if message.author == client.user:
        return

    channel_id = str(message.channel.id)
    channel_name = getattr(message.channel, "name", "direct-message")
    channel_history = conversation_memory[channel_id]

    # Package the incoming chat message details with recent context
    gatekeeper_context = build_gatekeeper_context(channel_name, channel_history, message)
    response_context = build_response_context(channel_name, message)

    # PASS 1: The Gatekeeper Prompt (Low temperature for strict logic decision)
    gatekeeper_system = (
        f"You are the background logic for an AI named {BOT_NAME}. Your job is to decide "
        f"if {BOT_NAME} should reply to the current message using the conversation history. "
        f"Reply YES if: the user directly mentions '{BOT_NAME}', asks a question the AI could answer, "
        f"or if the recent conversation makes a reply feel natural, relevant, or socially appropriate. "
        f"Reply NO if: it is generic small talk, spam, a command, or boring text that does not need a response. "
        f"CRITICAL: You must only output the single word 'YES' or 'NO'. Do not explain your reasoning."
    )

    decision = query_lm_studio(gatekeeper_system, gatekeeper_context, temperature=0.1)
    print(f"[{message.author.display_name}]: {message.content} -> Gatekeeper Decision: {decision}")

    # PASS 2: Character Generation (Only triggers if Gatekeeper says YES)
    if "YES" in decision.upper():
        async with message.channel.typing():
            character_system = (
                f"You are {BOT_NAME}, and you are a cute AI VTuber streamer "
                f"inspired by Neuro-sama. Keep your responses short, funny, and conversational while remembering to be serious when needed. "
                f"Never use emojis. Do not break character. "
                f"You may use tools when they help you answer accurately. "
                f"If you need more context, call get_recent_channel_history. "
                f"If you want to remember something important, call save_memory_note."
            )
            
            ai_response = query_lm_studio_with_tools(
                character_system,
                response_context,
                channel_history,
                temperature=0.8,
            )
            
            if ai_response:
                await message.channel.send(ai_response)

                channel_history.append(
                    {
                        "author": BOT_NAME,
                        "content": ai_response,
                    }
                )

    channel_history.append(
        {
            "author": message.author.display_name,
            "content": message.content,
        }
    )
    save_conversation_memory(conversation_memory)


if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
