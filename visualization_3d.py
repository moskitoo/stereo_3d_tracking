import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ObjectVisualizer:
    def __init__(self, figsize=(10, 8)):
        """
        Initialize 3D visualization with Matplotlib
        """
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('3D Object Tracking')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Store object plot references
        self.object_plots = {}
        self.object_trajectories = {}

    def update_visualization(self, tracked_objects):
        """
        Update 3D visualization with current tracked objects
        """
        # Clear previous plots
        for plot in list(self.object_plots.values()) + list(self.object_trajectories.values()):
            if plot:
                plot.remove()
        
        self.object_plots.clear()
        self.object_trajectories.clear()
        
        # Plot each object
        for obj_id, obj in tracked_objects.items():
            positions = np.array(obj.position)
            
            # Separate x, y, z coordinates
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

            if obj.id == 1:
                print(f"ID: {obj.id}, Estimated 3D position: {positions[-1]}")

            
            # Convert color to matplotlib format (normalized 0-1 RGB)
            color = tuple(c/255.0 for c in obj.color)
            
            # Plot current point
            current_point = self.ax.scatter(
                x[-1], y[-1], z[-1], 
                color=color, 
                s=100, 
                marker='o', 
                label=f'Object {obj_id}'
            )
            
            # Plot trajectory
            trajectory = self.ax.plot3D(x, y, z, color=color, linewidth=2, alpha=0.5)
            
            # Store plot references
            self.object_plots[obj_id] = current_point
            self.object_trajectories[obj_id] = trajectory[0]
        
        # Update legend
        self.ax.legend()
        
        # Automatically adjust plot limits
        self.ax.autoscale()

    def render(self, tracked_objects):
        """
        Update and render the visualization
        """
        self.update_visualization(tracked_objects)
        plt.draw()
        plt.pause(0.01)  # Small pause to update the plot

    def close(self):
        """
        Close the visualization
        """
        plt.ioff()
        plt.close(self.fig)