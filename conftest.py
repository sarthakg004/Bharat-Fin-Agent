"""Root conftest — makes `import finagent` resolve when pytest runs straight
from a checkout (CI does not `pip install -e .`): pytest prepends this file's
directory to sys.path during collection.
"""
