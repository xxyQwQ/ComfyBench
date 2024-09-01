from bs4 import BeautifulSoup

from agent.code_gen_agent.utils.state import AgentState
from agent.code_gen_agent.utils.function import safe_extract_from_soup


refiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to design their own workflows.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

The key points behind the requirements and the expected paradigm of the workflow is analyzed as follows:

{analysis}

## Reference

According to the analysis, we have retrieved relevant workflows. Here are some example workflows that may be helpful:

{reference}

## Workspace

The code and description of the current workflow you are working on is presented as follows:

{workspace}

## Refinement

Based on the current working progress, your step-by-step plan is presented as follows:

{planning}

To realize the first step of your plan, the code and description of your updated workflow are presented as in the workspace.
However, an error occurred when running your code. This may be caused by nested calls, missing parameters, or other issues. The detailed error message is presented as follows:

{refinement}

First, Try to explain why this error occurred.
Your explanation should be enclosed with "<explanation>" tag. For example: <explanation> The error occurred because the input parameter of node_1 is missing. </explanation>.

After that, correct the error and provide your Python code again.
Your code should be enclosed with "<code>" tag. For example: <code> output_1 = node_1() </code>.

Finally, provide a brief description of the updated workflow and the expected effects as in the example.
Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now, provide your explanation, code, and description with the required format.
'''


def get_refiner_agent_prompt(state: AgentState, planning: str, refinement: str):
    query_content = state.query
    analysis_content = state.analysis

    reference_content = ''
    for reference in state.reference:
        reference_content += f'- Example: {reference.metadata["name"]}\n\n'
        with open(reference.metadata['code'], 'r') as code_file:
            code = code_file.read()
        reference_content += f'<code>\n{code}\n</code>\n\n'
        with open(reference.metadata['description'], 'r') as desc_file:
            description = desc_file.read()
        reference_content += f'<description>\n{description}\n</description>\n\n'

    workspace_content = f'<code>\n{state.workspace["code"]}\n</code>\n\n'
    workspace_content += f'<description>\n{state.workspace["description"]}\n</description>'
    planning_content = planning
    refinement_content = refinement

    prompt_text = refiner_prompt.format(
        query=query_content,
        analysis=analysis_content,
        reference=reference_content,
        workspace=workspace_content,
        planning=planning_content,
        refinement=refinement_content
    )
    return prompt_text


def parse_refiner_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    explanation = safe_extract_from_soup(soup, 'explanation')
    code = safe_extract_from_soup(soup, 'code')
    description = safe_extract_from_soup(soup, 'description')
    return explanation, code, description