- `AV_AwsBedrockClaudeApi`: This node is designed to facilitate the integration with AWS Bedrock Claude API, providing a streamlined way to access Claude's language model capabilities through AWS. It abstracts the complexity of authentication and API communication, enabling users to easily leverage Claude's AI for various applications.
    - Inputs:
        - `aws_access_key_id` (Required): The AWS access key ID for authentication with AWS services. It's crucial for establishing a secure connection to AWS. Type should be `STRING`.
        - `aws_secret_access_key` (Required): The AWS secret access key for authentication. Together with the access key ID, it forms the credentials needed for accessing AWS resources securely. Type should be `STRING`.
        - `aws_session_token` (Required): An optional session token for temporary credentials that grant access to AWS services. It's used in conjunction with temporary access keys. Type should be `STRING`.
        - `region` (Required): The AWS region where the Bedrock Claude API is hosted. It determines the geographical location of the API endpoint being accessed. Type should be `COMBO[STRING]`.
        - `version` (Required): Specifies the version of the Bedrock Claude API to be used. It ensures that the API's features and capabilities are compatible with the user's requirements. Type should be `COMBO[STRING]`.
    - Outputs:
        - `llm_api`: The initialized Claude API object, ready for making requests to the Claude language model. Type should be `LLM_API`.