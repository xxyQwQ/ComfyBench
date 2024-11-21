- `CreateTaxonomy`: The CreateTaxonomy node is designed to automate the process of updating and generating a new taxonomy based on provided product descriptions and updates. It leverages a language model to suggest modifications to an existing taxonomy or create a new one, aiming to enhance the organization and categorization of products within a marketplace.
    - Inputs:
        - `path_to_descriptions` (Required): Specifies the file path to the product descriptions, which are used as input for generating or updating the taxonomy. Type should be `STRING`.
        - `updates` (Required): Indicates the number of updates to be applied to the taxonomy during the generation process. Type should be `INT`.
        - `batch_size` (Required): Determines the number of product descriptions processed in each batch for taxonomy generation or update. Type should be `INT`.
        - `update_rate` (Required): Specifies the frequency at which the taxonomy is updated based on the provided product descriptions. Type should be `INT`.
        - `fix_request` (Required): A template for the request to fix issues in the taxonomy, used by the language model to suggest specific changes. Type should be `STRING`.
        - `prompt` (Required): A template for the prompt used by the language model to categorize products and suggest taxonomy updates. Type should be `STRING`.
        - `llm` (Required): The language model used for generating taxonomy suggestions and updates. Type should be `LLM_MODEL`.
        - `path_to_taxonomy` (Optional): Optional. Specifies the file path to the existing taxonomy, which can be updated or used as a reference for generating a new taxonomy. Type should be `STRING`.
    - Outputs:
        - `taxonomy`: The updated or newly generated taxonomy as a result of the node's execution. Type should be `STRING`.
        - `old_taxonomy`: The original taxonomy before any updates were applied, used for comparison or rollback purposes. Type should be `STRING`.