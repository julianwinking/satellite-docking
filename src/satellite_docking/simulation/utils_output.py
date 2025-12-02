import os

def out_dir(folder: str) -> str:
    """
    Returns the output directory for the given exercise.
    """
    return os.path.abspath(folder)
