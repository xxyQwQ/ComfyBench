import re
import ast
import json

from markdown_to_json import dictify


def extract_key_value_pair(text: str) -> tuple[str, str]:
    key, value = text.split(':', 1)
    return key.strip(), value.strip()


def fetch_type_by_name(archive: dict, name: str) -> str:
    for key, value in archive.items():
        if value['identifier'] == name:
            return key
    return None


def fetch_slot_by_name(archive: list, name: str) -> int:
    for index, value in enumerate(archive):
        if value['name'] == name:
            return index
    return None


def extract_document_with_template(name: str, content: str) -> tuple[str, dict]:
    header_pattern = re.compile(r'---\n.*?\n---', re.DOTALL)
    content = header_pattern.sub('', content).strip()

    lines = content.split('\n')
    node_name = name
    node_description = lines[6]
    node_document = dictify('\n'.join(lines[7:]))

    input_list = []
    output_list = []

    # inputs (required)
    if isinstance(node_document['Input types'], dict) and 'Required' in node_document['Input types']:
        inputs_document = node_document['Input types']['Required']

        if isinstance(inputs_document, list):
            for index in range(0, len(inputs_document), 2):
                input_name = inputs_document[index].strip('*')
                input_name = input_name.strip('`')
                input_document = inputs_document[index + 1]
                if len(input_document) == 3:
                    input_description = input_document[0]
                    _, input_type = extract_key_value_pair(input_document[1])
                    input_type = input_type.strip('`')
                    input_list.append({
                        'name': input_name,
                        'type': input_type,
                        'description': input_description,
                        'required': True
                    })

    # inputs (optional)
    if isinstance(node_document['Input types'], dict) and 'Optional' in node_document['Input types']:
        inputs_document = node_document['Input types']['Optional']

        if isinstance(inputs_document, list):
            for index in range(0, len(inputs_document), 2):
                input_name = inputs_document[index].strip('*')
                input_name = input_name.strip('`')
                input_document = inputs_document[index + 1]
                if len(input_document) == 3:
                    input_description = input_document[0]
                    _, input_type = extract_key_value_pair(input_document[1])
                    input_type = input_type.strip('`')
                    input_list.append({
                        'name': input_name,
                        'type': input_type,
                        'description': input_description,
                        'required': False
                    })

    # outputs
    outputs_document = node_document['Output types']
    if isinstance(outputs_document, list):
        for index in range(0, len(outputs_document), 2):
            output_name = outputs_document[index].strip('*')
            output_name = output_name.strip('`')
            output_document = outputs_document[index + 1]
            if len(output_document) == 3:
                output_description = output_document[1]
                _, output_type = extract_key_value_pair(output_document[0])
                output_type = output_type.strip('`')
                output_list.append({
                    'name': output_name,
                    'type': output_type,
                    'description': output_description
                })

    document = f'- `{node_name}`: {node_description}\n'
    document += '    - Inputs:\n'
    for input_info in input_list:
        if input_info['required']:
            document += f'        - `{input_info["name"]}` (Required): '
        else:
            document += f'        - `{input_info["name"]}` (Optional): '
        document += f'{input_info["description"]} Type should be `{input_info["type"]}`.\n'
    document += '    - Outputs:\n'
    for output_info in output_list:
        document += f'        - `{output_info["name"]}`: '
        document += f'{output_info["description"]} Type should be `{output_info["type"]}`.\n'
    identifier = re.sub(r'[^A-Za-z0-9_]+', '_', name)
    archive = {
        'identifier': identifier,
        'description': node_description,
        'inputs': input_list,
        'outputs': output_list
    }

    return document, archive


def parse_prompt_to_code(prompt: dict, verbose: bool = False):
    code = ''
    type_list = []
    node_dict = {}

    with open('./dataset/benchmark/document/meta.json', 'r') as meta_file:
        node_meta = json.load(meta_file)

    for node_id, node_info in prompt.items():
        node_type = node_info['class_type']
        type_list.append(node_type)
        assert node_type in node_meta, f'node {node_type} not found'

        node_dict[node_id] = {
            'type': node_type,
            'name': node_meta[node_type]['identifier'],
            'inputs': node_info['inputs'],
            'outputs': [],
            'visited': False
        }

    for _ in range(len(node_dict)):
        for node_id, node_info in node_dict.items():
            if node_info['visited']:
                continue

            invalid = False
            for input_value in node_info['inputs'].values():
                if isinstance(input_value, list):
                    input_node, _ = input_value
                    if not node_dict[input_node]['visited']:
                        invalid = True
                        break
            if invalid:
                continue

            # the node is valid
            node_info['visited'] = True

            parameter_list = []
            for input_name, input_value in node_info['inputs'].items():
                if isinstance(input_value, list):
                    output_node, output_slot = input_value
                    input_value = f'{node_dict[output_node]["outputs"][output_slot]}'
                elif isinstance(input_value, str):
                    input_value = f'"""{input_value}"""'
                parameter_list.append(f'{input_name}={input_value}')

            return_list = []
            for output_info in node_meta[node_info['type']]['outputs']:
                return_name = f'{output_info["name"].replace(" ", "_").lower()}_{node_id}'
                node_info['outputs'].append(return_name)
                return_list.append(return_name)
            if not return_list:
                return_list.append('_')

            code += f'{", ".join(return_list)} = {node_info["name"]}'
            code += f'({", ".join(parameter_list)})\n'

    if verbose:
        extra = {'type_list': type_list}
        return code, extra
    else:
        return code


def parse_code_to_prompt(code: str, verbose: bool = False):
    node_count = 0
    type_list = []
    node_dict = {}

    with open('./dataset/benchmark/document/meta.json', 'r') as meta_file:
        node_meta = json.load(meta_file)
    variable_record = {}

    tree_root = ast.parse(code)
    for tree_node in tree_root.body:
        code_line = ast.unparse(tree_node).strip()
        assert isinstance(tree_node, ast.Assign), f'unexpected node type {type(tree_node)} in code line: {code_line}'

        node_name = tree_node.value.func.id
        node_type = fetch_type_by_name(node_meta, node_name)
        type_list.append(node_type)
        assert node_type, f'node type for {node_name} is not found'

        # create node
        node_count += 1
        node_id = str(node_count)
        node_info = {'class_type': node_type, 'inputs': {}}

        # process parameters
        for keyword in tree_node.value.keywords:
            if isinstance(keyword.value, ast.Constant):
                node_info['inputs'][keyword.arg] = keyword.value.value
            elif isinstance(keyword.value, ast.Name):
                node_info['inputs'][keyword.arg] = variable_record[keyword.value.id]

        # process returns
        for target in tree_node.targets:
            if isinstance(target, ast.Name):
                variable_record[target.id] = [node_id, 0]
            elif isinstance(target, ast.Tuple):
                for index, element in enumerate(target.elts):
                    variable_record[element.id] = [node_id, index]

        node_dict[node_id] = node_info

    prompt = node_dict
    if verbose:
        extra = {'type_list': type_list}
        return prompt, extra
    else:
        return prompt


def parse_prompt_to_markdown(prompt: dict, verbose: bool = False):
    markdown = ''
    type_list = []
    node_dict = {}

    with open('./dataset/benchmark/document/meta.json', 'r') as meta_file:
        node_meta = json.load(meta_file)

    for node_id, node_info in prompt.items():
        node_type = node_info['class_type']
        type_list.append(node_type)
        assert node_type in node_meta, f'node {node_type} not found'

        node_dict[node_id] = {
            'type': node_type,
            'name': f'N{node_id}',
            'inputs': node_info['inputs'],
            'outputs': node_meta[node_type]['outputs']
        }

    for node_id, node_info in node_dict.items():
        markdown += f'- {node_info["name"]}: {node_info["type"]}\n'

        for input_name, input_value in node_info['inputs'].items():
            if isinstance(input_value, list):
                output_node, output_slot = input_value
                output_name = node_dict[output_node]['outputs'][output_slot]['name']
                input_value = f'({node_dict[output_node]["name"]}.{output_name})'
            elif isinstance(input_value, str):
                input_value = input_value.replace('\n', ' ')
                input_value = f'"{input_value}"'
            markdown += f'    - {input_name}: {input_value}\n'

    if verbose:
        extra = {'type_list': type_list}
        return markdown, extra
    else:
        return markdown


def parse_markdown_to_prompt(markdown: str, verbose: bool = False):
    type_list = []
    node_dict = {}

    with open('./dataset/benchmark/document/meta.json', 'r') as meta_file:
        node_meta = json.load(meta_file)

    for line in markdown.split('\n'):
        if line.startswith('- '):
            node_name, node_type = line.strip('- ').split(': ')
            node_id = node_name.strip('N')
            node_dict[node_id] = {'class_type': node_type, 'inputs': {}}

    for line in markdown.split('\n'):
        if line.startswith('- '):
            node_name, node_type = line.strip('- ').split(': ')
            node_id = node_name.strip('N')
        elif line.startswith('    - '):
            input_name, input_value = line.strip('- ').split(': ')
            if input_value.startswith('(') and input_value.endswith(')'):
                output_node, output_name = input_value.strip('()').split('.')
                output_id = output_node.strip('N')
                output_type = node_dict[output_id]['class_type']
                output_slot = fetch_slot_by_name(node_meta[output_type]['outputs'], output_name)
                node_dict[node_id]['inputs'][input_name] = [output_id, output_slot]
            elif input_value.startswith('"') and input_value.endswith('"'):
                node_dict[node_id]['inputs'][input_name] = input_value.strip('"')
            else:
                node_dict[node_id]['inputs'][input_name] = eval(input_value)

    prompt = node_dict
    if verbose:
        extra = {'type_list': type_list}
        return prompt, extra
    else:
        return prompt
