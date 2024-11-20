import json

from bs4 import BeautifulSoup

from utils.parser import parse_code_to_prompt
from utils.model import ReferenceStorage, invoke_completion


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


generator_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Example

Here are some example workflows for your reference:

{example}

## Format

First, you should provide a plan, including which workflows you will refer to and how you will modify and compose them. Your plan should be enclosed using "<plan>" tag. For example: <plan>I will create my workflow based on "text-to-image" and "image_super_resolution". I will cascade them and rewrite the prompt text.</plan>.

After that, you should provide your Python code as in the example to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your plan and code with the required format.
'''


def format_generator_agent_prompt(instruction: str, example_list: list) -> str:
    instruction_content = instruction

    example_content = ''
    for example in example_list:
        example_content += f'- Workflow: {example.metadata["name"]}\n\n'
        example_content += f'<code>\n{example.metadata["code"]}\n</code>\n\n'
        example_content += f'<function>\n{example.metadata["function"]}\n</function>\n\n'
        example_content += f'<principle>\n{example.metadata["principle"]}\n</principle>\n\n'

    prompt_content = generator_prompt.format(
        instruction=instruction_content.strip(),
        example=example_content.strip()
    )
    return prompt_content


def parse_generator_agent_answer(answer: str) -> tuple[str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    plan = safe_extract_from_soup(soup, 'plan')
    code = safe_extract_from_soup(soup, 'code')
    return plan, code


aggregator_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Example

Here are some example workflows for your reference:

{example}

## Solution

Some possible solutions have been generated as follows:

{solution}

## Format

First, you should consider which solution is most likely to finish the task. Your thought should be enclosed using "<thought>" tag. For example: <thought>The first solution is the best because other solutions contain syntax errors.</thought>.

After that, you should choose the best solution by specifying the index. Your choice should be enclosed using "<choice>" tag. For example: <choice>1</choice>.

Now, provide your thought and choice with the required format.
'''


def format_aggregator_agent_prompt(instruction: str, example_list: list, solution_dict: dict) -> str:
    instruction_content = instruction

    example_content = ''
    for example in example_list:
        example_content += f'- Workflow: {example.metadata["name"]}\n\n'
        example_content += f'<code>\n{example.metadata["code"]}\n</code>\n\n'
        example_content += f'<function>\n{example.metadata["function"]}\n</function>\n\n'
        example_content += f'<principle>\n{example.metadata["principle"]}\n</principle>\n\n'

    solution_content = ''
    for index, solution in solution_dict.items():
        solution_content += f'Solution {index}:\n\n'
        solution_content += f'<plan>\n{solution["plan"]}\n</plan>\n\n'
        solution_content += f'<code>\n{solution["code"]}\n</code>\n\n'

    prompt_content = aggregator_prompt.format(
        instruction=instruction_content.strip(),
        example=example_content.strip(),
        solution=solution_content.strip()
    )
    return prompt_content


def parse_aggregator_agent_answer(answer: str) -> tuple[str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    thought = safe_extract_from_soup(soup, 'thought')
    choice = safe_extract_from_soup(soup, 'choice')
    return thought, choice


def parse_aggregator_agent_choice(choice: str) -> int:
    index = int(choice.strip())
    return index


class CoTSCPipeline(object):
    def __init__(self, num_examples: int = 3, num_trajectories: int = 3):
        self.num_examples = num_examples
        self.num_trajectories = num_trajectories

    def __call__(self, instruction: str) -> dict:
        # start pipeline
        print(' Task Instruction '.center(80, '-'))
        print(instruction)
        print()

        # create example
        storage = ReferenceStorage()
        example_list = storage.document_list[:self.num_examples]
        print(' Created Example '.center(80, '-'))
        for example in example_list:
            print(example.page_content)
            print()

        # generate solution
        solution_dict = {}
        message = format_generator_agent_prompt(instruction, example_list)
        print(' Generator Prompt '.center(80, '-'))
        print(message)
        print()
        for index in range(1, self.num_trajectories + 1):
            generation, usage = invoke_completion(message)
            print(' Generator Answer '.center(80, '-'))
            print(generation)
            print(usage)
            print()
            plan, code = parse_generator_agent_answer(generation)
            print(' Parsed Plan '.center(80, '-'))
            print(plan)
            print()
            print(' Parsed Code '.center(80, '-'))
            print(code)
            print()
            solution_dict[index] = {'plan': plan, 'code': code}

        # aggregate solution
        message = format_aggregator_agent_prompt(instruction, example_list, solution_dict)
        print(' Aggregator Prompt '.center(80, '-'))
        print(message)
        print()
        aggregation, usage = invoke_completion(message)
        print(' Aggregator Answer '.center(80, '-'))
        print(aggregation)
        print(usage)
        print()
        thought, choice = parse_aggregator_agent_answer(aggregation)
        print(' Parsed Thought '.center(80, '-'))
        print(thought)
        print()
        print(' Parsed Choice '.center(80, '-'))
        print(choice)
        print()
        index = parse_aggregator_agent_choice(choice)
        print(' Parsed Index '.center(80, '-'))
        print(index)
        print()
        plan = solution_dict[index]['plan']
        print(' Chosen Plan '.center(80, '-'))
        print(plan)
        print()
        code = solution_dict[index]['code']
        print(' Chosen Code '.center(80, '-'))
        print(code)
        print()

        # parse code
        try:
            prompt = parse_code_to_prompt(code)
        except Exception as error:
            raise RuntimeError('failed to parse code') from error
        print(' Parsed Workflow '.center(80, '-'))
        print(json.dumps(prompt, indent=4))
        print()

        return prompt
