import json

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from utils.parser import parse_code_to_prompt
from utils.model import ReferenceStorage, invoke_completion


class CRPAgentState(object):
    def __init__(self, instruction: str, reference_list: list[Document]):
        self.instruction = instruction
        self.reference_list = reference_list
        self.current_step = 0
        self.history_list = []
        self.workspace_code = ''

    def update_history(self, command: str):
        self.current_step += 1
        self.history_list.append({
            'step': self.current_step,
            'command': command
        })

    def update_workspace(self, code: str):
        self.workspace_code = code


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


user_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are a community user familiar with ComfyUI. Your task is to design workflows by commanding the intelligent assistant to generate corresponding Python code.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Reference

According to the requirements, we have retrieved some relevant workflows which may be helpful:

{reference}

## History

Here is a recent history of your commands. The most recent command is at the bottom:

{history}

## Workspace

The code of the current workflow is presented as follows:

{workspace}

## Format

Based on the history and workspace, you should provide your command describing how to modify the workflow in the next step. Your command should be enclosed using "<cmd>" tag. For example: <cmd>Refer to "reference_name" to add a module.</cmd> and <cmd>Change the factor to 0.5 and rewrite the prompt.</cmd>.

Now, provide your command with the required format.
'''


def format_user_agent_prompt(state: CRPAgentState) -> str:
    instruction_content = state.instruction

    reference_content = ''
    for reference in state.reference_list:
        reference_content += f'- Workflow: {reference.metadata["name"]}\n\n'
        reference_content += f'<code>\n{reference.metadata["code"]}\n</code>\n\n'
        reference_content += f'<function>\n{reference.metadata["function"]}\n</function>\n\n'
        reference_content += f'<principle>\n{reference.metadata["principle"]}\n</principle>\n\n'

    history_content = ''
    if len(state.history_list) == 0:
        history_content += '- The history is empty.'
    else:
        for history in state.history_list:
            history_content += f'- Step: {history["step"]}\n\n'
            history_content += f'<cmd>\n{history["command"]}\n</cmd>\n\n'

    workspace_content = ''
    if state.workspace_code == '':
        workspace_content += '- The workspace is empty.'
    else:
        workspace_content += f'<code>\n{state.workspace_code}\n</code>\n\n'

    prompt_content = user_prompt.format(
        instruction=instruction_content.strip(),
        reference=reference_content.strip(),
        history=history_content.strip(),
        workspace=workspace_content.strip()
    )
    return prompt_content


def parse_user_agent_answer(answer: str) -> str:
    soup = BeautifulSoup(answer, 'html.parser')
    command = safe_extract_from_soup(soup, 'cmd')
    return command


assistant_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an intelligent assistant familiar with ComfyUI. Your task is to design workflows by following the commands of the community user to generate corresponding Python code.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Reference

According to the requirements, we have retrieved some relevant workflows which may be helpful:

{reference}

## Command

The command given by the community user to modify the workflow is as follows:

{command}

## Workspace

The code of the current workflow is presented as follows:

{workspace}

## Format

Based on the command and workspace, you should provide your Python code to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your code with the required format.
'''


def format_assistant_agent_prompt(state: CRPAgentState) -> str:
    instruction_content = state.instruction

    reference_content = ''
    for reference in state.reference_list:
        reference_content += f'- Workflow: {reference.metadata["name"]}\n\n'
        reference_content += f'<code>\n{reference.metadata["code"]}\n</code>\n\n'
        reference_content += f'<function>\n{reference.metadata["function"]}\n</function>\n\n'
        reference_content += f'<principle>\n{reference.metadata["principle"]}\n</principle>\n\n'

    command_content = f'<cmd>\n{state.history_list[-1]["command"]}\n</cmd>'

    workspace_content = ''
    if state.workspace_code == '':
        workspace_content += '- The workspace is empty.'
    else:
        workspace_content += f'<code>\n{state.workspace_code}\n</code>\n\n'

    prompt_content = assistant_prompt.format(
        instruction=instruction_content.strip(),
        reference=reference_content.strip(),
        command=command_content.strip(),
        workspace=workspace_content.strip()
    )
    return prompt_content


def parse_assistant_agent_answer(answer: str) -> str:
    soup = BeautifulSoup(answer, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    return code


class CRPPipeline(object):
    def __init__(self, num_references: int = 5, num_rounds: int = 3):
        self.num_references = num_references
        self.num_rounds = num_rounds

    def __call__(self, instruction: str) -> dict:
        # start pipeline
        print(' Task Instruction '.center(80, '-'))
        print(instruction)
        print()

        # retrieve reference
        storage = ReferenceStorage()
        query = f'Instruction: {instruction}'
        reference_list = storage.retrieve(query, count=self.num_references)
        print(' Retrieved Reference '.center(80, '-'))
        for reference in reference_list:
            print(reference.page_content)
            print()
        state = CRPAgentState(instruction, reference_list)

        # generate workflow
        for round_index in range(self.num_rounds):
            print(f' Round {round_index + 1} '.center(80, '-'))
            print()
            message = format_user_agent_prompt(state)
            print(' User Prompt '.center(80, '-'))
            print(message)
            print()
            answer, usage = invoke_completion(message)
            print(' User Answer'.center(80, '-'))
            print(answer)
            print(usage)
            print()
            command = parse_user_agent_answer(answer)
            state.update_history(command)

            message = format_assistant_agent_prompt(state)
            print(f' Assistant Prompt '.center(80, '-'))
            print(message)
            print()
            answer, usage = invoke_completion(message)
            print(f' Assistant Answer'.center(80, '-'))
            print(answer)
            print(usage)
            print()
            code = parse_assistant_agent_answer(answer)
            state.update_workspace(code)

        # parse code
        try:
            prompt = parse_code_to_prompt(code)
        except Exception as error:
            raise RuntimeError('failed to parse code') from error
        print(' Parsed Workflow '.center(80, '-'))
        print(json.dumps(prompt, indent=4))
        print()

        return prompt
