- `LMStudioPrompt`: LMStudioPrompt is designed to interface with the LM Studio API, offering a streamlined way to generate prompts by leveraging the capabilities of this specific API. It mirrors the functionality of a similar node but specifically utilizes LM Studio's services for prompt generation, emphasizing its integration with LM Studio's unique features and API structure.
    - Inputs:
        - `input_prompt` (Required): The 'input_prompt' parameter represents the initial prompt or question provided by the user, serving as the basis for generating contextually relevant responses through the LM Studio API. Type should be `STRING`.
        - `mode` (Required): The 'mode' parameter determines the operational mode of the LM Studio API, potentially affecting the style or approach of the generated responses. Type should be `COMBO[STRING]`.
        - `custom_history` (Required): The 'custom_history' parameter allows for the inclusion of a custom interaction history, enhancing the context for the generated response by the LM Studio API. Type should be `STRING`.
        - `server_address` (Required): The 'server_address' parameter specifies the network address of the LM Studio server, required for API communication. Type should be `STRING`.
        - `server_port` (Required): The 'server_port' parameter indicates the network port of the LM Studio server, essential for establishing a connection. Type should be `INT`.
        - `seed` (Required): The 'seed' parameter is used to initialize the random number generator within the LM Studio API, ensuring reproducibility of the generated responses. Type should be `INT`.
    - Outputs:
        - `text`: The output 'text' contains the generated prompt from the LM Studio API, encapsulating the result of the prompt generation process. Type should be `STRING`.
