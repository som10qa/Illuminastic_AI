import random
import time
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import imageio.v2 as imageio

# ========= HELPERS FOR PATHS =========
def find_file(filename, search_dirs=[".", "config", "data"]):
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return filename

# ====== CONFIG/ARGUMENTS ======
def load_config():
    parser = argparse.ArgumentParser(description="Multi-Agent Navigation Simulator")
    parser.add_argument('--config', type=str, default="config/config.json", help="JSON config file")
    parser.add_argument('--size', type=int, default=10, help="Grid size (NxN)")
    parser.add_argument('--agents', type=int, default=2, help="Number of agents")
    parser.add_argument('--num_obstacles', type=int, default=10, help="Static obstacles")
    parser.add_argument('--num_wet', type=int, default=4, help="Wet floors")
    parser.add_argument('--num_walls', type=int, default=5, help="Walls")
    parser.add_argument('--moving_obstacles', type=int, default=2, help="Moving obstacles")
    parser.add_argument('--max_steps', type=int, default=100, help="Max steps per episode")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--rl_agent', type=str, default="0", help="Comma-separated RL agent indices")
    parser.add_argument('--make_gif', action='store_true', help="Make GIF after simulation")
    parser.add_argument('--share_every', type=int, default=1, help="Agents share knowledge every N steps")
    parser.add_argument('--init_states', type=str, default="data/initial_states.json", help="Initial agent states JSON (optional)")
    args = parser.parse_args()

    # If config file provided, load/overwrite from config directory
    config_path = find_file(args.config)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            d = json.load(f)
        for k, v in d.items():
            setattr(args, k, v)
    return args

args = load_config()

# ====== ENVIRONMENT ======
class Environment:
    """Grid world with static, moving obstacles, wet floors, and walls."""
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
        """Randomly move each moving obstacle."""
        updated = set()
        for pos in self.moving_obstacles:
            choices = [(pos[0]+dx, pos[1]+dy) for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]]
            valid = [c for c in choices if self.is_within(c) and c not in self.obstacles and c not in self.wet_floors and c not in self.walls]
            if valid:
                updated.add(random.choice(valid))
            else:
                updated.add(pos)
        self.moving_obstacles = updated

# ====== KNOWLEDGE BASE ======
class KnowledgeBase:
    def __init__(self, start, goal):
        self.data = {
            'user_position': start,
            'goal_position': goal,
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

# ====== SENSOR MODEL ======
class Sensors:
    def __init__(self, env, kb, noise=0.1):
        self.env = env
        self.kb = kb
        self.noise = noise
    def sense(self, pos):
        observations = {'obstacles': set(), 'wet_floors': set(), 'walls': set(), 'user_position': pos}
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                neighbor = (pos[0]+dx, pos[1]+dy)
                if self.env.is_within(neighbor):
                    if self.env.is_obstacle(neighbor):
                        if random.random() > self.noise:
                            observations['obstacles'].add(neighbor)
                    else:
                        if random.random() < self.noise/4:
                            observations['obstacles'].add(neighbor)
                    if self.env.is_wet(neighbor):
                        if random.random() > self.noise:
                            observations['wet_floors'].add(neighbor)
                    else:
                        if random.random() < self.noise/4:
                            observations['wet_floors'].add(neighbor)
                    if self.env.is_wall(neighbor):
                        if random.random() > self.noise:
                            observations['walls'].add(neighbor)
        return observations

# ====== AGENT COMMUNICATION ======
def communicate(kb_list):
    for i, kba in enumerate(kb_list):
        for j, kbb in enumerate(kb_list):
            if i != j:
                kbb.data['obstacles'].update(kba.data['obstacles'])
                kbb.data['wet_floors'].update(kba.data['wet_floors'])
                kbb.data['walls'].update(kba.data['walls'])

# ====== RL AGENT TEMPLATE ======
class RLAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
    def choose_action(self, kb, available_moves, goal):
        return random.choice(available_moves) if available_moves else None

# ====== PLANNER (A*) ======
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

# ====== UTILS ======
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

def reward_fn(agent_pos, goal, kb):
    if agent_pos == goal: return 100
    elif agent_pos in kb['obstacles'] or agent_pos in kb['wet_floors']: return -10
    else: return -1

def choose_agent_color(i):
    colors = ['yellow', 'magenta', 'orange', 'cyan', 'lime', 'pink']
    return colors[i % len(colors)]

# ====== VISUALIZATION ======
def draw_grid(agent_positions, agent_goals, obstacles, wet_floors, walls, moving_obstacles, size, step, path_histories=None, save_dir=None, kb_list=None, show_belief=False):
    grid = np.ones((size, size, 3), dtype=float)
    for (x, y) in obstacles: grid[x, y] = [1, 0, 0]
    for (x, y) in wet_floors: grid[x, y] = [0, 0, 1]
    for (x, y) in walls: grid[x, y] = [0.5, 0.5, 0.5]
    for (x, y) in moving_obstacles: grid[x, y] = [0, 0, 0]
    plt.clf()
    plt.imshow(grid, interpolation='none')
    plt.title(f"Multi-Agent Navigation\nStep: {step}")
    plt.axis('off')
    for i in range(size):
        for j in range(size):
            for a, pos in enumerate(agent_positions):
                if (i, j) == pos:
                    txt = f"A{a+1}"
                    plt.text(j, i, txt, ha='center', va='center', fontsize=11, weight='bold', color=choose_agent_color(a), bbox=dict(facecolor='white', alpha=0.7))
            for a, goal in enumerate(agent_goals):
                if (i, j) == goal:
                    plt.text(j, i, f"G{a+1}", ha='center', va='center', fontsize=12, weight='bold', color='green', bbox=dict(facecolor='white', alpha=0.7))
            if (i, j) in obstacles:
                plt.text(j, i, "O", ha='center', va='center', fontsize=10, color='red', alpha=0.8)
            elif (i, j) in wet_floors:
                plt.text(j, i, "W", ha='center', va='center', fontsize=10, color='blue', alpha=0.8)
            elif (i, j) in walls:
                plt.text(j, i, "||", ha='center', va='center', fontsize=10, color='black', alpha=0.8)
            elif (i, j) in moving_obstacles:
                plt.text(j, i, "M", ha='center', va='center', fontsize=10, color='black', alpha=0.8)
            if kb_list and show_belief:
                # Show if any agent *believes* this cell is a hazard
                known = any((i, j) in k['obstacles'] or (i, j) in k['wet_floors'] or (i, j) in k['walls'] for k in kb_list)
                if known:
                    plt.text(j, i, "!", ha='center', va='bottom', color='purple', alpha=0.5, fontsize=7)
    if path_histories:
        for a, path in enumerate(path_histories):
            if len(path) > 1:
                xs, ys = zip(*path)
                plt.plot(ys, xs, color=choose_agent_color(a), linewidth=2, marker='o', markersize=4, alpha=0.6)
    plt.tight_layout()
    plt.pause(0.01)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/frame_{step:03d}.png")

# ====== MAIN LOOP ======
def main():
    logs_dir = "logs"
    frames_dir = "frames"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(logs_dir, 'run1.log'), level=logging.INFO, format='%(asctime)s %(message)s')
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Load initial states/config ---
    agent_starts, agent_goals, agent_types, positions = [], [], [], set()
    num_agents = args.agents

    init_states_path = find_file(getattr(args, 'init_states', 'data/initial_states.json'))
    use_init_states = os.path.exists(init_states_path)
    if use_init_states:
        with open(init_states_path, "r") as f:
            init_data = json.load(f)
        if "agents" in init_data:
            for agent in init_data["agents"]:
                agent_starts.append(tuple(agent["start"]))
                agent_goals.append(tuple(agent["goal"]))
                agent_types.append(agent.get("type", "classic"))
                positions.add(tuple(agent["start"]))
                positions.add(tuple(agent["goal"]))
            num_agents = len(agent_starts)
        else:
            agent_starts = [tuple(pos) for pos in init_data.get("agent_starts",[])]
            agent_goals = [tuple(pos) for pos in init_data.get("agent_goals",[])]
            agent_types = ["classic" for _ in agent_starts]
            positions.update(agent_starts)
            positions.update(agent_goals)
            num_agents = len(agent_starts)
        if num_agents == 0 or len(agent_starts) != num_agents or len(agent_goals) != num_agents:
            raise RuntimeError(f"Initial states in {init_states_path} do not match number of agents")
    else:
        tries = 0
        max_tries = 1000
        agent_types = []
        for i in range(num_agents):
            while True:
                tries += 1
                if tries > max_tries:
                    raise RuntimeError("Unable to allocate enough unique start/goal positions. Try increasing grid size or reducing agents/obstacles.")
                start = (random.randint(0, args.size-1), random.randint(0, args.size-1))
                goal = (random.randint(0, args.size-1), random.randint(0, args.size-1))
                if start not in positions and goal not in positions and start != goal:
                    agent_starts.append(start)
                    agent_goals.append(goal)
                    positions.add(start)
                    positions.add(goal)
                    agent_types.append("classic")
                    break

    # --- RL agent selection: from config (preferred) or CLI ---
    if use_init_states and "agents" in locals() and "rl_agent_indices" in init_data:
        rl_agents_set = set(init_data["rl_agent_indices"])
    else:
        rl_agents_set = set(int(i) for i in str(args.rl_agent).split(",") if str(i).strip().isdigit())

    def random_locs(n):
        out = set()
        while len(out) < n:
            c = (random.randint(0, args.size-1), random.randint(0, args.size-1))
            if c not in positions: out.add(c)
        positions.update(out)
        return out

    obstacles = random_locs(int(args.num_obstacles))
    wet_floors = random_locs(int(args.num_wet))
    walls = random_locs(int(args.num_walls))
    moving_obstacles = random_locs(int(args.moving_obstacles))
    env = Environment(int(args.size), obstacles, wet_floors, walls=walls, moving_obstacles=moving_obstacles)

    agents, sensors_list, planners, policies, kbs, path_histories = [], [], [], [], [], []
    kb_histories = [[] for _ in range(num_agents)]
    for i in range(num_agents):
        kb = KnowledgeBase(agent_starts[i], agent_goals[i])
        kb.data['walls'] = walls
        kbs.append(kb)
        sensors_list.append(Sensors(env, kb))
        planners.append(Planner(kb, env))
        # Both by type or explicit RL index
        if (i in rl_agents_set) or (len(agent_types) > i and agent_types[i] == "rl"):
            policies.append(RLAgent(i))
        else:
            policies.append(None)
        path_histories.append([agent_starts[i]])

    plt.ion()
    plt.figure(figsize=(7,7))
    print("\n--- Multi-Agent Navigation Start ---\n")
    done = [False for _ in range(num_agents)]
    steps = 0
    while not all(done) and steps < int(args.max_steps):
        env.update_moving_obstacles()
        draw_grid(
            [kb.data['user_position'] for kb in kbs],
            agent_goals,
            env.obstacles,
            env.wet_floors,
            env.walls,
            env.moving_obstacles,
            int(args.size),
            steps,
            path_histories=path_histories,
            save_dir=frames_dir,
            kb_list=[kb.data for kb in kbs],
            show_belief=True
        )
        for i, kb in enumerate(kbs):
            kb_histories[i].append({"step": steps, "kb": serialize_kb(kb.data)})
        if int(args.share_every) > 0 and steps % int(args.share_every) == 0:
            communicate(kbs)
        for i in range(num_agents):
            if done[i]: continue
            pos = kbs[i].data['user_position']
            goal = kbs[i].data['goal_position']
            planner = planners[i]
            # RL or A* planner
            if policies[i] is not None:
                available_moves = []
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    npos = (pos[0]+dx,pos[1]+dy)
                    if env.is_within(npos) and npos not in kbs[i].data['obstacles'] and npos not in kbs[i].data['wet_floors'] and npos not in kbs[i].data['walls']:
                        available_moves.append(npos)
                next_pos = policies[i].choose_action(kbs[i].data, available_moves, goal)
            else:
                next_pos = planner.next_action(current=pos)
            if next_pos is None or next_pos == pos:
                done[i] = True
                continue
            obs = sensors_list[i].sense(next_pos)
            kbs[i].update(obs)
            if next_pos in kbs[i].data['obstacles'] or next_pos in kbs[i].data['wet_floors'] or next_pos in kbs[i].data['walls']:
                kbs[i].flag_uncertainty("Hazard/wall encountered! Replanning.")
                planners[i].plan_to_goal(start_override=pos)
                continue
            kbs[i].data['user_position'] = next_pos
            path_histories[i].append(next_pos)
            _ = reward_fn(next_pos, goal, kbs[i].data)
            if next_pos == goal:
                done[i] = True
        steps += 1
        time.sleep(0.01)
    plt.ioff()
    plt.show()

    print("\n--- Simulation End ---\n")
    for i, kb in enumerate(kbs):
        print(f"Agent {i+1} final KB:")
        for k, v in kb.data.items():
            print(f"  {k}: {v}")
    for i in range(num_agents):
        kb_file = os.path.join(logs_dir, f"kb_history_agent{i+1}.jsonl")
        with open(kb_file, "w") as f:
            for snap in kb_histories[i]:
                json.dump(snap, f)
                f.write("\n")

    # ===== Assemble GIF at end =====
    if args.make_gif:
        print("\n[INFO] Assembling simulation.gif from frames/ ...")
        images = []
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")])
        for fname in frame_files:
            images.append(imageio.imread(os.path.join(frames_dir, fname)))
        gif_path = os.path.join(frames_dir, "simulation.gif")
        imageio.mimsave(gif_path, images, duration=0.12)
        print(f"[INFO] simulation.gif saved at {gif_path}")

if __name__ == "__main__":
    main()