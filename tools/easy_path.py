import os

_MARKER = 'thesis_workspace2.code-workspace'

# tools/ lives directly inside WS_ROOT, so go up two levels from this file.
WS_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

assert os.path.exists(os.path.join(WS_ROOT, _MARKER)), (
    f"easy_path.py is misplaced — expected {_MARKER} in {WS_ROOT}"
)


def find_ws_root(start: str, marker: str = _MARKER) -> str:
    d = start
    while True:
        if os.path.exists(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f'Workspace root not found (looked for {marker})')
        d = parent
