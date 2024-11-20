---
tags:
- LoRA
---

# AWS Bedrock Claude API
## Documentation
- Class name: `AV_AwsBedrockClaudeApi`
- Category: `ArtVenture/LLM`
- Output node: `False`

This node is designed to facilitate the integration with AWS Bedrock Claude API, providing a streamlined way to access Claude's language model capabilities through AWS. It abstracts the complexity of authentication and API communication, enabling users to easily leverage Claude's AI for various applications.
## Input types
### Required
- **`aws_access_key_id`**
    - The AWS access key ID for authentication with AWS services. It's crucial for establishing a secure connection to AWS.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`aws_secret_access_key`**
    - The AWS secret access key for authentication. Together with the access key ID, it forms the credentials needed for accessing AWS resources securely.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`aws_session_token`**
    - An optional session token for temporary credentials that grant access to AWS services. It's used in conjunction with temporary access keys.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`region`**
    - The AWS region where the Bedrock Claude API is hosted. It determines the geographical location of the API endpoint being accessed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`version`**
    - Specifies the version of the Bedrock Claude API to be used. It ensures that the API's features and capabilities are compatible with the user's requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`llm_api`**
    - Comfy dtype: `LLM_API`
    - The initialized Claude API object, ready for making requests to the Claude language model.
    - Python dtype: `ClaudeApi`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AwsBedrockClaudeApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aws_access_key_id": ("STRING", {"multiline": False}),
                "aws_secret_access_key": ("STRING", {"multiline": False}),
                "aws_session_token": ("STRING", {"multiline": False}),
                "region": (aws_regions, {"default": aws_regions[0]}),
                "version": (bedrock_anthropic_versions, {"default": bedrock_anthropic_versions[0]}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, aws_access_key_id, aws_secret_access_key, aws_session_token, region, version):
        if not aws_access_key_id or aws_access_key_id == "":
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
        if not aws_secret_access_key or aws_secret_access_key == "":
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        if not aws_session_token or aws_session_token == "":
            aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

        if not aws_access_key_id or not aws_secret_access_key:
            raise Exception("AWS credentials is required.")

        return (
            AwsBedrockClaudeApi(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region=region,
                version=version,
            ),
        )

```
