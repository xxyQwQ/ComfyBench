# ∞ Simple Chat
## Documentation
- Class name: `SimpleChat`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

SimpleChat facilitates one-on-one chat simulations between two conversable agents, allowing for the initiation of dialogues, the application of summary methods to chat histories, and the option to clear or maintain chat history for subsequent interactions. It abstracts the complexities of chat initiation and management, providing a straightforward interface for simulating realistic chat interactions.
## Input types
### Required
- **`initiator`**
    - Represents the initiating conversable agent in the chat, setting the stage for the dialogue's commencement.
    - Comfy dtype: `AGENT`
    - Python dtype: `ConversableAgent`
- **`recipient`**
    - Denotes the recipient conversable agent, who responds to the initiator's messages, completing the two-way communication setup.
    - Comfy dtype: `AGENT`
    - Python dtype: `ConversableAgent`
- **`summary_method`**
    - Specifies the method used to summarize the chat history, impacting the way the conversation's essence is captured and presented.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clear_history`**
    - A boolean flag indicating whether to clear the chat history before starting a new chat session, affecting the continuity of conversations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`init_message`**
    - The initial message to start the chat with, setting the tone and context for the ensuing dialogue.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_turns`**
    - Limits the number of turns the chat can take, controlling the conversation's length and flow.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`chat_history`**
    - Comfy dtype: `STRING`
    - The compiled history of messages exchanged during the chat, presented in a readable format.
    - Python dtype: `str`
- **`summary`**
    - Comfy dtype: `STRING`
    - A concise summary of the chat, reflecting the conversation's key points and outcomes.
    - Python dtype: `str`
- **`initiator`**
    - Comfy dtype: `AGENT`
    - Returns the initiating agent after the chat, potentially with updated state or information.
    - Python dtype: `ConversableAgent`
- **`recipient`**
    - Comfy dtype: `AGENT`
    - Returns the recipient agent after the chat, potentially with updated state or information.
    - Python dtype: `ConversableAgent`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SimpleChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initiator": ("AGENT",),
                "recipient": ("AGENT",),
                "summary_method": ([
                    "last_msg",
                    "reflection_with_llm",
                ],),
                # whether to clear the chat history with the agent
                "clear_history": ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "init_message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "max_turns": ("INT", {"default": 10}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "AGENT", "AGENT",)
    RETURN_NAMES = ("chat_history", "summary", "initiator", "recipient",)

    FUNCTION = "start_chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Chat"

    def start_chat(self, initiator, recipient, summary_method, clear_history, init_message=None, max_turns=None):
        initiator = clone_conversable_agent(initiator)
        recipient = clone_conversable_agent(recipient)

        chat_result = initiator.initiate_chat(
            recipient,
            message=init_message,
            max_turns=max_turns,
            summary_method=summary_method,
            clear_history=clear_history,
        )
        summary = chat_result.summary
        chat_history = ""
        for message in chat_result.chat_history:
            if message["content"] is not None:
                chat_history += "- " + message["content"] + "\n\n"
        return (chat_history, summary, initiator, recipient,)

```
