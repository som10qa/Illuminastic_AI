import random
import time
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import json

# --------- ENVIRONMENT (with wall/moving obstacles hooks) ---------
class Environment:
    def __init__(self, size, obstacles, wet_floors, walls=None, moving_obstacles=None):
        self.size = size
        self.obstacles = set(obstacles)
        self.wet_floors = set(wet_floors)
        self.walls = set(walls) if walls else set()
        self.moving_obstacles = set(moving_obstacles) if moving_obstacles else set()

    def is_obstacle(self, pos):
        return pos in self.obstacles or pos in self.moving_obstacles

    def is_wet(self, pos):
        return pos in self.wet_floors

    def is_wall(self, pos):
        return pos in self.walls

    def is_within(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def update_moving_obstacles(self):
        # Placeholder: Move each moving obstacle randomly (for demo only)
        updated = set()
        for pos in self.moving_obstacles:
            choices = [(pos[0]+dx, pos[1]+dy) for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]]
            valid = [c for c in choices if self.is_within(c) and c not in self.obstacles and c not in self.wet_floors and c not in self.walls]
            if valid:
                updated.add(random.choice(valid))
            else:
                updated.add(pos)
        self.moving_obstacles = updated

# --------- KNOWLEDGE BASE ---------
class KnowledgeBase:
    def __init__(self):
        self.data = {
            'user_position': (0, 0),
            'goal_position': None,
            'obstacles': set(),
            'wet_floors': set(),
            'walls': set(),
            'uncertainty': {},
            'last_update': 0,
            'status_flags': {'uncertainty': False, 'sensor_error': False}
        }

    def update(self, observation):
        for key, value in observation.items():
            if key in ['obstacles', 'wet_floors', 'walls']:
                self.data[key].update(value)
            else:
                self.data[key] = value
        self.data['last_update'] = time.time()

    def flag_uncertainty(self, msg):
        self.data['status_flags']['uncertainty'] = True
        self.data['uncertainty'][time.time()] = msg

    def clear_flags(self):
        self.data['status_flags'] = {'uncertainty': False, 'sensor_error': False}

# --------- SENSOR SIMULATION (Extensible) ---------
class Sensors:
    def __init__(self, env, kb, noise=0.1):
        self.env = env
        self.kb = kb
        self.noise = noise  # probability of sensor error

    def sense(self, pos):
        # Baseline: perceive nearby obstacles and wet floors
        observations = {'obstacles': set(), 'wet_floors': set(), 'walls': set(), 'user_position': pos}
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                neighbor = (pos[0]+dx, pos[1]+dy)
                if self.env.is_within(neighbor):
                    # Obstacles
                    if self.env.is_obstacle(neighbor):
                        if random.random() > self.noise:
                            observations['obstacles'].add(neighbor)
                    else:
                        if random.random() < self.noise/4:
                            observations['obstacles'].add(neighbor)
                    # Wet floors
                    if self.env.is_wet(neighbor):
                        if random.random() > self.noise:
                            observations['wet_floors'].add(neighbor)
                    else:
                        if random.random() < self.noise/4:
                            observations['wet_floors'].add(neighbor)
                    # Walls (new sensor type example)
                    if self.env.is_wall(neighbor):
                        if random.random() > self.noise:
                            observations['walls'].add(neighbor)
        # -- Example: Add more sensors here (e.g., thermal, sound, etc.)
        # observations['thermal'] = self.sense_thermal(pos)
        return observations

    # Example sensor: thermal hazard (stub, for future extension)
    def sense_thermal(self, pos):
        # Returns high if a 'thermal hazard' cell nearby (for RL/fusion experiments)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (pos[0]+dx, pos[1]+dy)
                # Define a custom logic for "thermal" cell detection
                # For demo, return random
                if random.random() < 0.1:
                    return "HOT"
        return "SAFE"

# --------- POLICY LEARNING HOOK (stub) ---------
class PolicyLearner:
    def __init__(self):
        pass  # Replace with RL or learned policy setup

    def choose_action(self, kb, available_moves):
        # Placeholder: random move (replace with RL policy later)
        return random.choice(available_moves) if available_moves else None

# --------- PLANNER (A* with FRESH REPLAN) ---------
class Planner:
    def __init__(self, kb, env):
        self.kb = kb
        self.env = env
        self.current_plan = []

    def plan_to_goal(self, start_override=None):
        from queue import PriorityQueue
        start = start_override if start_override else self.kb.data['user_position']
        goal = self.kb.data['goal_position']
        obstacles = self.kb.data['obstacles'] | self.kb.data.get('walls', set())
        wet_floors = self.kb.data['wet_floors']

        pq = PriorityQueue()
        pq.put((0, start, []))
        visited = set()
        while not pq.empty():
            cost, node, path = pq.get()
            if node == goal:
                self.current_plan = path + [node]
                return
            if node in visited:
                continue
            visited.add(node)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                neighbor = (node[0]+dx, node[1]+dy)
                if not self.env.is_within(neighbor): continue
                if neighbor in obstacles: continue
                if neighbor in wet_floors: continue
                if neighbor in visited: continue
                h = abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                pq.put((cost+1+h, neighbor, path+[node]))
        self.current_plan = []

    def next_action(self, current=None):
        if not self.current_plan or (current and self.current_plan[0] != current):
            self.plan_to_goal(start_override=current)
        if len(self.current_plan) < 2:
            return None
        return self.current_plan[1]

# --------- VISUALIZATION WITH SAVE ---------
def draw_grid(user_pos, goal, obstacles, wet_floors, walls, size, step, path=None, save_dir=None):
    grid = np.ones((size, size, 3), dtype=float)
    for (x, y) in obstacles:
        grid[x, y] = [1, 0, 0]    # Red for obstacles
    for (x, y) in wet_floors:
        grid[x, y] = [0, 0, 1]    # Blue for wet floor
    for (x, y) in walls:
        grid[x, y] = [0.5, 0.5, 0.5] # Grey for walls
    grid[goal[0], goal[1]] = [0, 1, 0]  # Green for goal
    grid[user_pos[0], user_pos[1]] = [1, 1, 0]  # Yellow for user

    plt.clf()
    plt.imshow(grid, interpolation='none')
    plt.title(f"Illuminastick Navigation\nStep: {step}, User:{user_pos}, Goal:{goal}")
    plt.axis('off')

    # Annotate grid
    for i in range(size):
        for j in range(size):
            txt = ""
            if (i, j) == (0, 0):
                txt = "S"  # Start
            elif (i, j) == goal:
                txt = "G"  # Goal
            elif (i, j) == user_pos:
                txt = "A"  # Agent
            elif (i, j) in obstacles:
                txt = "O"  # Obstacle
            elif (i, j) in wet_floors:
                txt = "W"  # Wet Floor
            elif (i, j) in walls:
                txt = "||" # Wall
            else:
                txt = ""
            if txt:
                plt.text(j, i, txt, ha='center', va='center', fontsize=12, weight='bold', color='black', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))
    if path is not None and len(path) > 1:
        xs, ys = zip(*path)
        plt.plot(ys, xs, color='cyan', linewidth=2, marker='o', markersize=5, alpha=0.7, label="Path")
        plt.legend(loc="lower left")
    plt.tight_layout()
    plt.pause(0.05)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/frame_{step:03d}.png")

# --------- MAIN AGENT LOOP ---------
def serialize_kb(kb):
    out = {}
    for k, v in kb.items():
        if isinstance(v, set):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = {str(time): msg for time, msg in v.items()}
        else:
            out[k] = v
    return out

def main():
    # Folders
    if not os.path.exists("logs"): os.makedirs("logs")
    if not os.path.exists("frames"): os.makedirs("frames")
    logging.basicConfig(filename='logs/run1.log', level=logging.INFO, format='%(asctime)s %(message)s')

    # Environment setup
    size = 10
    random.seed(42)
    np.random.seed(42)
    goal = (8, 9)
    avoid = {(0, 0), goal}
    obstacles = set()
    while len(obstacles) < 10:
        c = (random.randint(1,8), random.randint(1,8))
        if c not in avoid: obstacles.add(c)
    wet_floors = set()
    while len(wet_floors) < 4:
        c = (random.randint(1,8), random.randint(1,8))
        if c not in avoid and c not in obstacles: wet_floors.add(c)
    walls = set()
    while len(walls) < 5:
        c = (random.randint(1,8), random.randint(1,8))
        if c not in avoid and c not in obstacles and c not in wet_floors: walls.add(c)
    moving_obstacles = {(5, 5)} # Example: single moving obstacle

    env = Environment(size, obstacles, wet_floors, walls=walls, moving_obstacles=moving_obstacles)
    kb = KnowledgeBase()
    kb.data['goal_position'] = goal
    kb.data['walls'] = walls
    sensors = Sensors(env, kb)
    planner = Planner(kb, env)
    policy = PolicyLearner() # Placeholder for RL policies

    user_pos = (0, 0)
    kb.data['user_position'] = user_pos
    planner.plan_to_goal(start_override=user_pos)

    step = 0
    path_history = [user_pos]
    kb_history = []

    plt.ion()
    plt.figure(figsize=(6,6))
    print("\n--- Illuminastick Navigation Start ---\n")
    while user_pos != goal and step < 100:
        print(f"Step {step}: User at {user_pos}")
        logging.info(f"Step {step}: User at {user_pos}, KB: {kb.data}")

        # Update moving obstacles (if any)
        env.update_moving_obstacles()

        draw_grid(user_pos, goal, env.obstacles | env.moving_obstacles, env.wet_floors, env.walls, size, step, path=path_history, save_dir="frames")
        kb_history.append({"step": step, "kb": serialize_kb(kb.data)})

        # Sense environment (extend here for more sensors)
        observations = sensors.sense(user_pos)
        kb.update(observations)

        # Wet floor warning
        if any(n in kb.data['wet_floors'] for n in [(user_pos[0]+dx,user_pos[1]+dy) for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]]):
            print("Warning: Wet floor nearby! Alert user!")
            logging.info("Warning: Wet floor nearby! Alert user!")

        # --- POLICY LEARNING HOOK: choose action (stub: use A* planner, can replace with RL) ---
        # next_pos = policy.choose_action(kb.data, list(neighbor_positions))  # for RL
        next_pos = planner.next_action(current=user_pos)  # for classical planner

        if next_pos is None:
            print("No path found to goal. Stopping.")
            logging.info("No path found to goal. Stopping.")
            break
        if next_pos in kb.data['obstacles'] or next_pos in kb.data['wet_floors'] or next_pos in kb.data.get('walls', set()):
            kb.flag_uncertainty("Obstacle/wet floor/wall unexpectedly detected!")
            print("Unexpected obstacle! Handling... Replanning.")
            logging.info("Unexpected obstacle! Handling... Replanning.")
            planner.plan_to_goal(start_override=user_pos)
            continue
        print(f"Moving from {user_pos} -> {next_pos}")
        logging.info(f"Moving from {user_pos} -> {next_pos}")
        user_pos = next_pos
        kb.data['user_position'] = user_pos
        path_history.append(user_pos)
        if user_pos == goal:
            print("\nGOAL REACHED!")
            logging.info("GOAL REACHED!")
            break
        step += 1
        time.sleep(0.05)
    else:
        print("\nMax steps reached or unable to reach goal.")
        logging.info("Max steps reached or unable to reach goal.")
    plt.ioff()
    plt.show()
    print("\n--- Navigation End ---\n")
    print("Final Knowledge Base:")
    for k, v in kb.data.items():
        print(f"{k}: {v}")

    # Save KB history as JSON Lines
    with open("kb_history.jsonl", "w") as f:
        for snapshot in kb_history:
            json.dump(snapshot, f)
            f.write("\n")

if __name__ == "__main__":
    main()
