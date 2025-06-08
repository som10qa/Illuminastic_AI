
import matplotlib.pyplot as plt
import numpy as np

def plot_belief_map(kb, grid_size=10, save_path=None):
    """
    Plot agent's current knowledge (obstacles, wet floors, etc.) as a heatmap.
    """
    grid = np.zeros((grid_size, grid_size))
    for x, y in kb['obstacles']:
        grid[x, y] = 1
    for x, y in kb['wet_floors']:
        grid[x, y] = 0.5
    for x, y in kb.get('walls', []):
        grid[x, y] = 0.8
    plt.imshow(grid, cmap='coolwarm', interpolation='nearest')
    plt.title("Agent Belief Map")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.show()
