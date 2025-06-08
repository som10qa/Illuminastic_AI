import json

def summarize_agent_path(kb_history_path):
    """Print stats about the agent's run."""
    with open(kb_history_path, "r") as f:
        steps = 0
        positions = []
        for line in f:
            snap = json.loads(line)
            positions.append(tuple(snap["kb"]["user_position"]))
            steps += 1
    print(f"Total steps: {steps}")
    print(f"Path: {positions}")
    print(f"Unique cells visited: {len(set(positions))}")
