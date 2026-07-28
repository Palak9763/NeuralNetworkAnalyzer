"""
hello_world_not_ml.py

A simple non-ML Python script with no neural network structure.
Used to verify that Feature 2 (Custom / Raw-Code AST Pattern-Matching)
cleanly raises ModelParsingError / FrameworkNotSupportedError instead of
crashing or producing a nonsense graph result.
"""


def calculate_greeting(name: str) -> str:
    message = f"Hello, {name}!"
    print(message)
    return message


if __name__ == "__main__":
    calculate_greeting("World")
