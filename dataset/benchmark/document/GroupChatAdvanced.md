- `GroupChatAdvanced`: The GroupChatAdvanced node facilitates advanced group chat simulations among multiple agents, incorporating features like message filtering, speaker selection, and customizable chat introductions. It enables the creation and management of dynamic, multi-agent conversations for various simulation and interaction scenarios.
    - Inputs:
        - `group_manager` (Required): Specifies the manager of the group chat, orchestrating the flow and rules of the conversation. Type should be `GROUP_MANAGER`.
        - `init_message` (Required): The initial message to start the chat, setting the context for the conversation. Type should be `STRING`.
        - `select_speaker_message_template` (Required): Template for customizing the message that introduces the speaker selection process, guiding the narrative flow. Type should be `STRING`.
        - `select_speaker_prompt_template` (Required): Customizes the select speaker prompt, guiding the LLM in selecting the next agent to speak. Type should be `STRING`.
        - `summary_method` (Required): Determines the method for generating a chat summary, affecting the analysis and insights derived from the conversation. Type should be `COMBO[STRING]`.
        - `max_turns` (Required): Specifies the maximum number of message exchanges allowed in the chat, controlling the conversation's length. Type should be `INT`.
        - `func_call_filter` (Required): Determines if the next speaker is chosen based on function call suggestions, influencing the flow of conversation. Type should be `BOOLEAN`.
        - `speaker_selection_method` (Required): Specifies the method used to select the next speaker, affecting the dynamics of the conversation. Type should be `COMBO[STRING]`.
        - `allow_repeat_speaker` (Required): Allows or disallows the same speaker to be chosen consecutively, impacting the variety of conversation. Type should be `BOOLEAN`.
        - `send_introductions` (Required): Controls whether introductory messages are sent at the beginning of the chat, setting the stage for the conversation. Type should be `BOOLEAN`.
        - `role_for_select_speaker_messages` (Required): Defines the role used in select speaker messages, guiding the context for speaker selection. Type should be `COMBO[STRING]`.
        - `clear_history` (Required): Indicates whether the chat history should be cleared before starting a new session, impacting the continuity of conversations. Type should be `BOOLEAN`.
        - `agents` (Optional): A list of agents participating in the chat. It's crucial for simulating the group chat dynamics and interactions among different entities. Type should be `AGENTS`.
    - Outputs:
        - `chat_history`: The compiled history of messages exchanged during the chat, providing a complete record of the conversation. Type should be `STRING`.
        - `summary`: A summary of the chat, offering insights or an overview of the conversation's content and outcomes. Type should be `STRING`.