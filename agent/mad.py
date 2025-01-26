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

## Reference

According to the requirements, we have retrieved some relevant workflows which may be helpful:

{reference}

## Format

You should provide your Python code as in the reference to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your code with the required format.
'''


def format_generator_agent_prompt(instruction: str, reference_list: list) -> str:
    instruction_content = instruction

    reference_content = ''
    for reference in reference_list:
        reference_content += f'- Workflow: {reference.metadata["name"]}\n\n'
        reference_content += f'<code>\n{reference.metadata["code"]}\n</code>\n\n'
        reference_content += f'<function>\n{reference.metadata["function"]}\n</function>\n\n'
        reference_content += f'<principle>\n{reference.metadata["principle"]}\n</principle>\n\n'

    prompt_content = generator_prompt.format(
        instruction=instruction_content.strip(),
        reference=reference_content.strip()
    )
    return prompt_content


def parse_generator_agent_answer(answer: str) -> str:
    soup = BeautifulSoup(answer, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    return code


solver_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping to design workflows according to user requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Reference

Here are some workflows created by other solvers. They may be correct or incorrect, but you can refer to them, which may help correct or improve your own workflow.

{reference}

## Workspace

The code of your own workflow is presented as follows:

{code}

## Format

You should provide your Python code to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your code with the required format.
'''


def format_solver_agent_prompt(instruction: str, reference_list: list, code: str) -> str:
    instruction_content = instruction

    reference_content = ''
    for reference in reference_list:
        reference_content += f'<code>\n{reference}\n</code>\n\n'

    code_content = f'<code>\n{code}\n</code>\n\n'

    prompt_content = solver_prompt.format(
        instruction=instruction_content.strip(),
        reference=reference_content.strip(),
        code=code_content.strip()
    )
    return prompt_content


def parse_solver_agent_answer(answer: str) -> str:
    soup = BeautifulSoup(answer, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    return code


aggregator_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping to design workflows according to user requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Solution

After debating, there are {count} solutions provided by different solvers. You should consider these solutions and choose the best one to finalize the workflow. The workflows are presented as follows:

{solution}

## Format

You should provide your Python code to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your code with the required format.
'''


def format_aggregator_agent_prompt(instruction: str, count: int, solution_list: list) -> str:
    instruction_content = instruction
    count_content = str(count)

    solution_content = ''
    for index, solution in enumerate(solution_list):
        solution_content += f'- Solution {index + 1}\n\n'
        solution_content += f'<code>\n{solution}\n</code>\n\n'

    prompt_content = aggregator_prompt.format(
        instruction=instruction_content.strip(),
        count=count_content.strip(),
        solution=solution_content.strip()
    )
    return prompt_content


def parse_aggregator_agent_answer(answer: str) -> str:
    soup = BeautifulSoup(answer, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    return code


class MADPipeline(object):
    def __init__(self, num_references: int = 5, num_solvers: int = 3, num_rounds: int = 1):
        self.num_references = num_references
        self.num_solvers = num_solvers
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

        # generate workflow
        code_list = []
        print(' Round 0 '.center(80, '-'))
        print()
        for solver_index in range(self.num_solvers):
            message = format_generator_agent_prompt(instruction, reference_list)
            print(f' Solver-{solver_index + 1} Prompt '.center(80, '-'))
            print(message)
            print()
            generation, usage = invoke_completion(message)
            print(f' Solver-{solver_index + 1} Answer'.center(80, '-'))
            print(generation)
            print(usage)
            print()
            code = parse_generator_agent_answer(generation)
            code_list.append(code)

        # simulate debate
        for round_index in range(self.num_rounds):
            update_list = []
            print(f' Round {round_index + 1} '.center(80, '-'))
            print()
            for solver_index in range(self.num_solvers):
                reference_list = []
                for reference_index, reference_code in enumerate(code_list):
                    if reference_index != solver_index:
                        reference_list.append(reference_code)
                message = format_solver_agent_prompt(instruction, reference_list, code_list[solver_index])
                print(f' Solver-{solver_index + 1} Prompt '.center(80, '-'))
                print(message)
                print()
                solution, usage = invoke_completion(message)
                print(f' Solver-{solver_index + 1} Answer'.center(80, '-'))
                print(solution)
                print(usage)
                print()
                code = parse_solver_agent_answer(solution)
                update_list.append(code)
            code_list = update_list

        # aggregate solution
        message = format_aggregator_agent_prompt(instruction, self.num_solvers, code_list)
        print(' Aggregator Prompt '.center(80, '-'))
        print(message)
        print()
        aggregation, usage = invoke_completion(message)
        print(' Aggregator Answer'.center(80, '-'))
        print(aggregation)
        print()
        code = parse_aggregator_agent_answer(aggregation)

        # parse code
        try:
            prompt = parse_code_to_prompt(code)
        except Exception as error:
            raise RuntimeError('failed to parse code') from error
        print(' Parsed Workflow '.center(80, '-'))
        print(json.dumps(prompt, indent=4))
        print()

        return prompt
