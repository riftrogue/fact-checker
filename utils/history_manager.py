import json
import os

def load_history(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(filename, messages):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
                                                                