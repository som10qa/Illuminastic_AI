# Illuminastick: A Knowledge-Based Embodied AI Navigation Simulator in Dynamic Grid Environments

**Abstract:**  
*Illuminastick* is a research-driven simulation framework for studying knowledge-based embodied agents in partially observable, dynamic environments. The agent operates within a configurable 2D grid world, tasked with autonomous navigation from a designated start location to a goal, while effectively avoiding randomly placed obstacles and hazardous “wet floor” zones. The simulator models realistic uncertainty via noisy sensors, and supports stepwise visualization, detailed knowledge-base tracking, and rigorous evaluation of planning under uncertainty.

---

## Key Features and Research Components

- **Dynamic Grid Environment:**  
  The simulator generates a 10x10 (or configurable) grid containing randomly distributed obstacles and hazardous cells, providing a testbed for robust navigation and knowledge integration.

- **Structured Knowledge Base:**  
  The agent maintains an explicit, updatable knowledge base capturing its position, the goal location, detected obstacles, wet floor regions, and uncertainty flags. This supports research on symbolic knowledge representation and update mechanisms in embodied systems.

- **Noisy Perception via Sensors:**  
  Sensor models are parameterized to reflect real-world uncertainty, introducing both false positives and false negatives in obstacle/hazard detection. This facilitates experimentation with robust decision-making and reasoning under partial observability.

- **Reactive and Deliberative Planning:**  
  Navigation employs an A*-based planner, which replans paths dynamically as new information is perceived. This allows examination of the integration between reactive sensing and symbolic planning in dynamic environments.

- **Stepwise Visualization & Annotation:**  
  Each time step is visualized with annotated grid cells indicating agent (A), start (S), goal (G), obstacles (O), wet floors (W), and (optionally) walls (`||`). The agent’s traversed path is overlaid and each frame is saved for post-experiment analysis.

- **Comprehensive Logging & Data Export:**  
  The simulator logs agent state and knowledge base after every move, producing both JSONL and plain-text logs for subsequent research, reproducibility, or reporting.

---

## Extensibility and Future Directions

The Illuminastick framework is designed for modular extensibility. Potential research and coursework extensions include (but are not limited to):

- **Introducing New Sensor Modalities:**  
  Add custom sensor types (e.g., range-finders, vision, tactile sensors) to investigate sensor fusion or multi-modal perception in embodied agents.

- **Complex Environment Generation:**  
  Incorporate structured layouts (e.g., walls, corridors), moving hazards, or multi-agent scenarios to test advanced planning and coordination strategies.

- **Learning-Based and Policy Optimization Approaches:**  
  Integrate reinforcement learning or imitation learning algorithms to evolve policies for navigation under uncertainty, enabling direct comparison with knowledge-based (symbolic) approaches.

- **Adaptive and Hierarchical Planning:**  
  Explore integration of high-level task planners, conditional planning under uncertainty, or hybrid symbolic-subsymbolic architectures.

---

## Code Structure and Customization

- **illuminastick_sim.py:**  
  Main simulation file implementing the environment, agent, sensors, planner, knowledge base, and visualization.

- **Adding new features:**  
  - **Sensor Types:** Extend the `Sensors` class (e.g., add `sense_thermal`, `sense_sound`).
  - **Complex Environments:** Add more wall and moving obstacle logic in the `Environment` class.
  - **Policy Learning:** Implement RL in the `PolicyLearner` class and update the main loop for RL-based decision-making.
  - **Multi-Agent:** Instantiate multiple agents with separate or shared knowledge bases.

---

## How to Run

```bash
python3 illuminastick_sim.py

or

```bash
python3 illuminastick_sim_parameterized.py --agents 3 --make_gif --config config/config.json
