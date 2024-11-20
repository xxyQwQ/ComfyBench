# ∞ Group Chat (Adv)
## Documentation
- Class name: `GroupChatAdvanced`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

The GroupChatAdvanced node facilitates advanced group chat simulations among multiple agents, incorporating features like message filtering, speaker selection, and customizable chat introductions. It enables the creation and management of dynamic, multi-agent conversations for various simulation and interaction scenarios.
## Input types
### Required
- **`group_manager`**
    - Specifies the manager of the group chat, orchestrating the flow and rules of the conversation.
    - Comfy dtype: `GROUP_MANAGER`
    - Python dtype: `Dict`
- **`init_message`**
    - The initial message to start the chat, setting the context for the conversation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`select_speaker_message_template`**
    - Template for customizing the message that introduces the speaker selection process, guiding the narrative flow.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`select_speaker_prompt_template`**
    - Customizes the select speaker prompt, guiding the LLM in selecting the next agent to speak.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`summary_method`**
    - Determines the method for generating a chat summary, affecting the analysis and insights derived from the conversation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_turns`**
    - Specifies the maximum number of message exchanges allowed in the chat, controlling the conversation's length.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`func_call_filter`**
    - Determines if the next speaker is chosen based on function call suggestions, influencing the flow of conversation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`speaker_selection_method`**
    - Specifies the method used to select the next speaker, affecting the dynamics of the conversation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`allow_repeat_speaker`**
    - Allows or disallows the same speaker to be chosen consecutively, impacting the variety of conversation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`send_introductions`**
    - Controls whether introductory messages are sent at the beginning of the chat, setting the stage for the conversation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`role_for_select_speaker_messages`**
    - Defines the role used in select speaker messages, guiding the context for speaker selection.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clear_history`**
    - Indicates whether the chat history should be cleared before starting a new session, impacting the continuity of conversations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`agents`**
    - A list of agents participating in the chat. It's crucial for simulating the group chat dynamics and interactions among different entities.
    - Comfy dtype: `AGENTS`
    - Python dtype: `List[Agent]`
## Output types
- **`chat_history`**
    - Comfy dtype: `STRING`
    - The compiled history of messages exchanged during the chat, providing a complete record of the conversation.
    - Python dtype: `str`
- **`summary`**
    - Comfy dtype: `STRING`
    - A summary of the chat, offering insights or an overview of the conversation's content and outcomes.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GroupChatAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_manager": ("GROUP_MANAGER",),
                "init_message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                # select_speaker_message_template: customize the select speaker message (used in "auto" speaker selection), which appears first in the message context and generally includes the agent descriptions and list of agents. The string value will be converted to an f-string, use "{roles}" to output the agent's and their role descriptions and "{agentlist}" for a comma-separated list of agent names in square brackets.
                "select_speaker_message_template": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "You are in a role play game. The following roles are available:\n{roles}.\nRead the following conversation.\nThen select the next role from {agentlist} to play. Only return the role."
                }),
                # select_speaker_prompt_template: customize the select speaker prompt (used in "auto" speaker selection), which appears last in the message context and generally includes the list of agents and guidance for the LLM to select the next agent. The string value will be converted to an f-string, use "{agentlist}" for a comma-separated list of agent names in square brackets.
                "select_speaker_prompt_template": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
                }),
                "summary_method": ([
                    "last_msg",
                    "reflection_with_llm",
                ],),
                "max_turns": ("INT", {"default": 10}),
                # When set to True and when a message is a function call suggestion,
                # the next speaker will be chosen from an agent which contains the corresponding function name
                # in its `function_map`
                "func_call_filter": ("BOOLEAN", {"default": True},),
                # speaker_selection_method: the method for selecting the next speaker.
                # Could be any of the following (case insensitive), will raise ValueError if not recognized:
                # - "auto": the next speaker is selected automatically by LLM.
                # - "manual": the next speaker is selected manually by user input.
                # - "random": the next speaker is selected randomly.
                # - "round_robin": the next speaker is selected in a round robin fashion, i.e., iterating in the same order as provided in `agents`.
                "speaker_selection_method": ([
                    "auto",
                    # "manual",
                    "random",
                    "round_robin",
                ],),
                # whether to allow the same speaker to speak consecutively.
                "allow_repeat_speaker": ("BOOLEAN", {"default": True},),
                # send_introductions: send a round of introductions at the start of the group chat, so agents know who they can speak to
                "send_introductions": ("BOOLEAN", {"default": False},),
                # role_for_select_speaker_messages: sets the role name for speaker selection when in 'auto' mode, typically 'user' or 'system'.
                "role_for_select_speaker_messages": ([
                    "system",
                    "user",
                ],),
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
        select_speaker_message_template,
        select_speaker_prompt_template,
        func_call_filter,
        speaker_selection_method,
        allow_repeat_speaker,
        send_introductions,
        role_for_select_speaker_messages,
        summary_method,
        max_turns,
        clear_history,
        agents,
    ):
        assert len(agents) != 1, "At least 1 agent is needed to start a group chat session"
        # create chat
        group_chat = AutogenGroupChat(
            agents=agents,
            messages=[],
            max_round=max_turns,
            func_call_filter=func_call_filter,
            select_speaker_prompt_template=select_speaker_prompt_template,
            speaker_selection_method=speaker_selection_method,
            allow_repeat_speaker=allow_repeat_speaker,
            send_introductions=send_introductions,
            role_for_select_speaker_messages=role_for_select_speaker_messages,
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
            chat_history += "- " + str(message["content"]) + "\n\n"
        return (chat_history, summary)

```
