import sys


RESET = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'


class Logger:
    def __init__(self, path: str):
        self.console = sys.stdout
        self.file = open(path, 'w')

    def write(self, message: str):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        pass


def bold(text: str) -> str:
    return f'{BOLD}{text}{RESET}'


def underline(text: str) -> str:
    return f'{UNDERLINE}{text}{RESET}'


def red(text: str) -> str:
    return f'{RED}{text}{RESET}'


def green(text: str) -> str:
    return f'{GREEN}{text}{RESET}'


def yellow(text: str) -> str:
    return f'{YELLOW}{text}{RESET}'
