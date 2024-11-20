import os
import json
import yaml
import uuid
import time

import websocket
from urllib.parse import urlencode
from urllib.request import Request, urlopen


with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    if config['proxy']['http_proxy'] is not None:
        os.environ['http_proxy'] = config['proxy']['http_proxy']
    if config['proxy']['https_proxy'] is not None:
        os.environ['https_proxy'] = config['proxy']['https_proxy']
    CLIENT_ID = str(uuid.uuid4())
    SERVER_ADDRESS = config['comfyui']['comfyui_address']
    EXECUTION_TIMEOUT = config['comfyui']['execution_timeout']


def queue_prompt(prompt: dict) -> dict:
    request = Request(
        url=f'http://{SERVER_ADDRESS}/prompt',
        data=json.dumps({
            'prompt': prompt,
            'client_id': CLIENT_ID
        }).encode('utf-8')
    )
    with urlopen(request) as response:
        feedback = json.loads(response.read())
    return feedback


def interrupt_prompt() -> dict:
    request = Request(
        url=f'http://{SERVER_ADDRESS}/interrupt',
        method='POST'
    )
    with urlopen(request) as response:
        feedback = response.status
    return {'status': feedback}


def fetch_history(prompt_id: str) -> dict:
    with urlopen(f'http://{SERVER_ADDRESS}/history/{prompt_id}') as response:
        history = json.loads(response.read())
    return history


def fetch_output(filename: str, subfolder: str) -> bytes:
    parameter = urlencode({
        'filename': filename,
        'subfolder': subfolder,
        'type': 'output'
    })
    with urlopen(f'http://{SERVER_ADDRESS}/view?{parameter}') as response:
        output = response.read()
    return output


def execute_prompt(prompt: dict) -> tuple[dict, dict]:
    outputs = {}

    socket = websocket.WebSocket()
    socket.connect(f'ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}')
    prompt_id = queue_prompt(prompt)['prompt_id']

    socket.settimeout(EXECUTION_TIMEOUT)
    timeout = time.time() + EXECUTION_TIMEOUT
    while True:
        data = socket.recv()
        if isinstance(data, str):
            message = json.loads(data)
            if message['type'] == 'executing':
                message = message['data']
                if message['node'] is None and message['prompt_id'] == prompt_id:
                    break
        if time.time() > timeout:
            interrupt_prompt()
            raise TimeoutError('execution timeout')

    history = fetch_history(prompt_id)[prompt_id]
    for node_output in history['outputs'].values():
        for type_output in node_output.values():
            for spec_output in type_output:
                if isinstance(spec_output, dict) and spec_output['type'] == 'output':
                    output = fetch_output(spec_output['filename'], spec_output['subfolder'])
                    outputs[spec_output['filename']] = output

    status = history['status']
    return status, outputs
