import os

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def load_prompt(name, **kwargs):
    """Load a prompt template from the prompts/ directory.

    Args:
        name: Filename (with or without .txt extension).
        **kwargs: Placeholders to substitute, e.g. load_prompt("x", text=val).

    Returns:
        Prompt string with placeholders replaced.
    """
    if not name.endswith(".txt"):
        name += ".txt"
    filepath = os.path.join(_PROMPTS_DIR, name)
    with open(filepath, "r", encoding="utf-8") as f:
        prompt = f.read()
    for key, value in kwargs.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt
