import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ObjectVisualizer:
    def __init__(self, figsize=(16, 12), xlim=(-60, 20), ylim=(-20, 20), zlim=(-10, 30)):
        """
        Initialize 3D visualization with Matplotlib
        """
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=figsize)
        
        # 3D plot
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_3d.set_title('3D Object Tracking')
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Z')
        self.ax_3d.set_zlabel('Y')
        self.ax_3d.set_xlim(xlim)
        self.ax_3d.set_ylim(zlim)
        self.ax_3d.set_zlim(ylim)
        
        # XY projection
        self.ax_xy = self.fig.add_subplot(222)
        self.ax_xy.set_title('XY Projection')
        self.ax_xy.set_xlabel('X')
        self.ax_xy.set_ylabel('Y')
        self.ax_xy.set_xlim(xlim)
        self.ax_xy.set_ylim(ylim)
        
        # XA projection
        self.ax_xa = self.fig.add_subplot(223)
        self.ax_xa.set_title('XA Projection')
        self.ax_xa.set_xlabel('X')
        self.ax_xa.set_ylabel('Z')
        self.ax_xa.set_xlim(xlim)
        self.ax_xa.set_ylim(zlim)
        
        # YX projection
        self.ax_yx = self.fig.add_subplot(224)
        self.ax_yx.set_title('YX Projection')
        self.ax_yx.set_xlabel('Y')
        self.ax_yx.set_ylabel('Z')
        self.ax_yx.set_xlim(ylim)
        self.ax_yx.set_ylim(zlim)
        
        # Store object plot references
        self.object_plots_3d = {}
        self.object_trajectories_3d = {}
        self.object_projections = {'xy': {}, 'xa': {}, 'yx': {}}
        self.annotations = {'3d': {}, 'xy': {}, 'xa': {}, 'yx': {}}

    def update_visualization(self, tracked_objects):
        """
        Update 3D visualization with current tracked objects and their projections
        """
        # Clear previous plots
        for obj_id in list(self.object_plots_3d.keys()):
            if self.object_plots_3d[obj_id]:
                self.object_plots_3d[obj_id].remove()
            if self.object_trajectories_3d[obj_id]:
                self.object_trajectories_3d[obj_id][0].remove()
        self.object_plots_3d.clear()
        self.object_trajectories_3d.clear()
        
        for projection in self.object_projections.values():
            for obj_id, plots in list(projection.items()):
                if plots[0]:
                    plots[0].remove()
                if plots[1]:
                    plots[1].remove()
            projection.clear()
        
        for annotation_set in self.annotations.values():
            for obj_id, annotation in list(annotation_set.items()):
                if annotation:
                    annotation.remove()
            annotation_set.clear()
        
        # Plot each object
        for obj_id, obj in tracked_objects.items():
            positions = np.array(obj.world_3d_position)
            x, y, z = positions[0], positions[1], positions[2]
            color = tuple(c / 255.0 for c in reversed(obj.color))
            
            # 3D plot
            self.object_plots_3d[obj_id] = self.ax_3d.scatter(
                x, z, y, color=color, s=200, marker='o'  # Increased marker size
            )
            self.annotations['3d'][obj_id] = self.ax_3d.text(
                x, z, y, f'{obj_id}', color='white', ha='center', va='center', fontsize=8
            )
            self.object_trajectories_3d[obj_id] = self.ax_3d.plot3D(x, z, y, color=color, linewidth=2, alpha=0.5)
            
            # 2D projections
            self.object_projections['xy'][obj_id] = [
                self.ax_xy.scatter(x, y, color=color, s=200),  # Increased marker size
                self.ax_xy.plot(x, y, color=color, linewidth=2, alpha=0.5)[0]
            ]
            self.annotations['xy'][obj_id] = self.ax_xy.text(
                x, y, f'{obj_id}', color='white', ha='center', va='center', fontsize=8
            )
            
            self.object_projections['xa'][obj_id] = [
                self.ax_xa.scatter(x, z, color=color, s=200),  # Increased marker size
                self.ax_xa.plot(x, z, color=color, linewidth=2, alpha=0.5)[0]
            ]
            self.annotations['xa'][obj_id] = self.ax_xa.text(
                x, z, f'{obj_id}', color='white', ha='center', va='center', fontsize=8
            )
            
            self.object_projections['yx'][obj_id] = [
                self.ax_yx.scatter(y, z, color=color, s=200),  # Increased marker size
                self.ax_yx.plot(y, z, color=color, linewidth=2, alpha=0.5)[0]
            ]
            self.annotations['yx'][obj_id] = self.ax_yx.text(
                y, z, f'{obj_id}', color='white', ha='center', va='center', fontsize=8
            )
        
        # Update legends
        self.ax_3d.legend()



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
