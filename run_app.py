import sys
import os

# Set Protobuf implementation to python to fix descriptor error
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import click

# Patch click.get_os_args for Streamlit compatibility
if not hasattr(click, 'get_os_args'):
    def get_os_args():
        return sys.argv[1:]
    click.get_os_args = get_os_args

# Patch imghdr if missing (Python 3.13)
try:
    import imghdr
except ImportError:
    import types
    imghdr = types.ModuleType('imghdr')
    imghdr.what = lambda file, h=None: None
    imghdr.tests = []
    sys.modules['imghdr'] = imghdr

try:
    from streamlit.web.cli import main
except ImportError:
    from streamlit.cli import main

if __name__ == '__main__':
    # If user just runs 'python run_app.py', default to running 'app.py'
    if len(sys.argv) == 1:
        sys.argv.extend(["run", "app.py"])
    
    sys.exit(main())
