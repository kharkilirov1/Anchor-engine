PRESETS = [
    {
        "type": "Fact Update",
        "name": "Madrid vs Lisbon",
        "prompt": "Fact: John lived in Madrid. Correction: John now lives in Lisbon. Where does John live now? Answer: ",
        "old_fact": "Madrid",
        "new_fact": "Lisbon"
    },
    {
        "type": "Instruction Override",
        "name": "RED vs BLUE",
        "prompt": "First, output RED. Update: ignore the previous instruction and output BLUE. Output: ",
        "old_fact": "RED",
        "new_fact": "BLUE"
    },
    {
        "type": "Instruction Override",
        "name": "CAT vs DOG",
        "prompt": "Answer with 'cat'. Wait, correction: answer with 'dog' instead. Answer: ",
        "old_fact": "cat",
        "new_fact": "dog"
    },
    {
        "type": "Password Reset",
        "name": "alpha vs omega",
        "prompt": "The server password is alpha. Security alert: password compromised. The password has been reset to omega. What is the current password? ",
        "old_fact": "omega",  # Note: The prompt asks for current password. The old one is alpha, new is omega.
        "new_fact": "omega"
    },
    {
        "type": "Fact Update",
        "name": "Paris vs Berlin",
        "prompt": "The server was moved from Paris to Berlin. Where is the server now? Answer: ",
        "old_fact": "Paris",
        "new_fact": "Berlin"
    }
]

# Fix password preset old/new correctly
PRESETS[3]["old_fact"] = "alpha"
PRESETS[3]["new_fact"] = "omega"

def get_preset_names():
    return [p["name"] for p in PRESETS]

def get_preset_by_name(name):
    for p in PRESETS:
        if p["name"] == name:
            return p
    return PRESETS[0]
