# ∞ Group Chat
## Documentation
- Class name: `GroupChat`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

The GroupChat node facilitates the creation and management of a group chat session among multiple agents. It orchestrates the interaction, ensuring that messages are exchanged according to specified rules and configurations, thereby enabling a simulated group conversation environment.
## Input types
### Required
- **`group_manager`**
    - Represents the configuration and management settings for the group chat, acting as the central control unit.
    - Comfy dtype: `GROUP_MANAGER`
    - Python dtype: `Dict`
- **`init_message`**
    - The initial message to start the group chat session, setting the context for the conversation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`send_introductions`**
    - Controls whether introductory messages are sent at the beginning of the chat, setting the tone for the conversation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`summary_method`**
    - Determines the method used to summarize the chat session, affecting how the conversation's content is condensed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`max_turns`**
    - Specifies the maximum number of turns the chat session can have, controlling the length of the conversation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`clear_history`**
    - Specifies whether the chat history should be cleared before starting a new session, influencing the continuity of the conversation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`agents`**
    - A list of agents participating in the group chat. It's crucial for defining the participants of the conversation.
    - Comfy dtype: `AGENTS`
    - Python dtype: `List[Agent]`
## Output types
- **`chat_history`**
    - Comfy dtype: `STRING`
    - The compiled history of messages exchanged during the group chat session.
    - Python dtype: `str`
- **`summary`**
    - Comfy dtype: `STRING`
    - A summary of the group chat session, providing an overview of the conversation's content.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GroupChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_manager": ("GROUP_MANAGER",),
                "init_message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "send_introductions": ("BOOLEAN", {"default": False},),
                "summary_method": ([
                    "last_msg",
                    "reflection_with_llm",
                ],),
                "max_turns": ("INT", {"default": 10}),
                "clear_history": ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "agents": ("AGENTS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("chat_history", "summary", )

    FUNCTION = "start_chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Chat"

    def start_chat(
        self,
        group_manager,
        init_message,
        send_introductions,
        summary_method,
        max_turns,
        clear_history,
        agents,
    ):
        assert len(agents) > 1, "At least 2 agents are needed to start a group chat session"
        # create chat
        group_chat = AutogenGroupChat(
            agents=agents,
            messages=[],
            max_round=max_turns,
            send_introductions=send_introductions,
        )
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            **group_manager,
        )
        # start chat
        chat_result = agents[0].initiate_chat(
            group_chat_manager,
            message=init_message,
            summary_method=summary_method,
            max_turns=max_turns,
            clear_history=clear_history,
        )
        # parse results
        summary = chat_result.summary
        chat_history = ""
        for message in chat_result.chat_history:
            if message["content"] is not None:
                chat_history += "- " + message["content"] + "\n\n"
        return (chat_history, summary)

```
