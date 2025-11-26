
# Patch for Python 3.13 where imghdr is removed
# Streamlit depends on it.

def what(file, h=None):
    return None

def tests():
    return []
