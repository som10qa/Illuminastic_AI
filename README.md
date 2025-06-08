# Illuminastick: A Knowledge-Based Embodied AI Navigation Simulator in Dynamic Grid Environments

**Abstract:**  
*Illuminastick* is a research-driven simulation framework for studying knowledge-based embodied agents in partially observable, dynamic environments. Agents operate within a configurable 2D grid world, tasked with autonomous navigation from a designated start location to a goal, while effectively avoiding randomly placed obstacles and hazardous “wet floor” zones. The simulator models realistic uncertainty via noisy sensors, supports stepwise visualization, detailed knowledge-base tracking, and rigorous evaluation of planning under uncertainty.

---

## Key Features and Research Components

- **Dynamic Grid Environment:**  
  The simulator generates a 10x10 (or configurable) grid containing randomly distributed obstacles, hazardous cells, and walls, providing a testbed for robust navigation and knowledge integration.

- **Structured Knowledge Base:**  
  Each agent maintains an explicit, updatable knowledge base capturing its position, the goal location, detected obstacles, wet floor regions, walls, and uncertainty flags. The knowledge base tracks the agent’s belief about the environment, enabling symbolic knowledge representation and reasoning.

- **Noisy Perception & Sensor Adaptation:**  
  Sensor models reflect real-world uncertainty, introducing both false positives and negatives in detection. The simulator now **supports sensor adaptation/learning**: sensor noise decreases as the agent gains experience, mimicking improved reliability through adaptation or learning.

- **BLE-based Localization:**  
  Agents periodically “re-localize” using simulated BLE signals, correcting their position to a legal (non-obstacle/wall) grid cell. This models real-world ground-truth correction, and supports research on integrating external localization signals in mobile robotics.

- **Ontology-Based Inference & Shared Belief:**  
  Agents construct and update an explicit **belief map** using ontological labels (e.g., obstacle, wall, wet floor). At each step, the simulator computes the intersection of all agents’ beliefs—identifying hazardous or critical cells recognized by every agent—enabling collective situation assessment and symbolic reasoning.

- **Reactive and Deliberative Planning:**  
  Navigation uses an A*-based planner, which replans dynamically as new information is perceived or received. The framework supports switching between knowledge-based (symbolic) and reinforcement learning (RL) policies for agent action selection.

- **Stepwise Visualization & Annotation:**  
  Every simulation step is visualized with annotated grid cells (agent, goal, obstacles, wet floors, walls), with agent paths and belief overlays. Frames can be exported as a GIF for post-analysis.

- **Comprehensive Logging & Data Export:**  
  The simulator logs agent state and knowledge base after every move, producing JSONL and plain-text logs for research, reproducibility, or reporting.

---

## Recent Research Extensions (2024–2025)

- **Sensor Adaptation/Learning:**  
  Agents adapt their sensors over time, decreasing noise (error rate) based on navigation progress or hazardous events.

- **BLE/External Localization Integration:**  
  Agents periodically update their location using simulated BLE fixes, mimicking periodic corrections via indoor localization systems.

- **Ontology-Enabled Reasoning:**  
  All agents’ belief maps are merged every step, with cells recognized as hazardous by every agent being reported as critical knowledge—a foundation for advanced ontological reasoning or situation awareness.

- **Multi-Agent Communication:**  
  Agents can share and merge their knowledge, simulating explicit communication or shared memory architectures.

---

## Code Structure and Customization

- **illuminastick_sim.py** / **illuminastick_sim_parameterized.py:**  
  Main simulation scripts—implement the environment, agent, sensors, planner, ontology, knowledge base, and visualization.

- **Feature extension points:**  
  - **Sensors:** Extend or adapt the `Sensors` class for new modalities or adaptive/learning behavior.
  - **Environment:** Add new obstacle types, structured layouts, or moving hazards.
  - **Ontology/Belief:** Plug in symbolic reasoning, logic, or even formal OWL/Description Logic modules for more advanced AI.
  - **RL/Policy Learning:** Implement RL agents in the RLAgent template and connect to learning pipelines.
  - **BLE Integration:** Connect `get_ble_location()` to real-world APIs or simulated positioning systems.

---

## How to Run

```bash
# Standard run (single agent, default settings)
python3 illuminastick_sim.py

# Multi-agent, advanced features enabled
python3 illuminastick_sim_parameterized.py --agents 3 --make_gif --config config/config.json

#Example Output
[BLE] Agent 2 re-localized via BLE at (4, 2)
[Sensor Adaptation] Sensor noise decreased to 0.15
[Ontology] Step 12: All agents believe these cells are hazardous: [(2, 3), (5, 4), (7, 2)]