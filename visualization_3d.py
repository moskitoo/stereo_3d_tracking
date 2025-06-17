
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg
class ObjectVisualizer:
    def __init__(self, 
                 figsize=(16, 12), 
                 xlim=(-15, 20), 
                 ylim=(-2, 2), 
                 zlim=(-10, 60),
                 display_plots=None):
        """
        Initialize 3D visualization with Matplotlib, with configurable plot display
        
        Parameters:
        -----------
        figsize : tuple, optional
            Size of the main figure
        xlim : tuple, optional
            X-axis limits
        ylim : tuple, optional
            Y-axis limits
        zlim : tuple, optional
            Z-axis limits
        display_plots : dict, optional
            Dictionary controlling which plots are displayed
            Default: All plots enabled
            Example: {
                'main_4subplot': True,
                '3d_separate': True,
                'xy_separate': True,
                'xz_separate': True,
                'yx_separate': True
            }
        """
        # Default display configuration
        self.default_display_plots = {
            'main_4subplot': True,
            '3d_separate': True,
            'xy_separate': True,
            'xz_separate': True,
            'yx_separate': True
        }
        
        # Update display configuration
        self.display_plots = self.default_display_plots.copy()
        if display_plots is not None:
            self.display_plots.update(display_plots)
        
        plt.ion()  # Interactive mode
        
        # Initialize figures and axes only for enabled plots
        self.figs = {}
        self.axes = {}
        
        # Main figure with 4 subplots
        if self.display_plots['main_4subplot']:
            self.figs['main'] = plt.figure(figsize=figsize)
            
            # 3D plot
            self.axes['3d'] = self.figs['main'].add_subplot(221, projection='3d')
            self.axes['3d'].set_title('3D Object Tracking')
            self.axes['3d'].set_xlabel('X')
            self.axes['3d'].set_ylabel('Z')
            self.axes['3d'].set_zlabel('Y')
            self.axes['3d'].set_xlim(xlim)
            self.axes['3d'].set_ylim(zlim)
            self.axes['3d'].set_zlim(ylim)
            
            # XY projection
            self.axes['xy'] = self.figs['main'].add_subplot(222)
            self.axes['xy'].set_title('XY Projection (side plane)')
            self.axes['xy'].set_xlabel('X')
            self.axes['xy'].set_ylabel('Y')
            self.axes['xy'].set_xlim(xlim)
            self.axes['xy'].set_ylim(ylim)
            
            # XZ projection
            self.axes['xz'] = self.figs['main'].add_subplot(223)
            self.axes['xz'].set_title('XZ Projection (ground plane)')
            self.axes['xz'].set_xlabel('X')
            self.axes['xz'].set_ylabel('Z')
            self.axes['xz'].set_xlim(xlim)
            self.axes['xz'].set_ylim(zlim)
            
            # YX projection
            self.axes['yx'] = self.figs['main'].add_subplot(224)
            self.axes['yx'].set_title('YX Projection (front plane)')
            self.axes['yx'].set_xlabel('Y')
            self.axes['yx'].set_ylabel('Z')
            self.axes['yx'].set_xlim(ylim)
            self.axes['yx'].set_ylim(zlim)
        
        # Separate windows
        if self.display_plots['3d_separate']:
            self.figs['3d_separate'] = plt.figure(figsize=(10, 8))
            self.axes['3d_separate'] = self.figs['3d_separate'].add_subplot(111, projection='3d')
            self.axes['3d_separate'].set_title('3D Object Tracking')
            self.axes['3d_separate'].set_xlabel('X')
            self.axes['3d_separate'].set_ylabel('Z')
            self.axes['3d_separate'].set_zlabel('Y')
            self.axes['3d_separate'].set_xlim(xlim)
            self.axes['3d_separate'].set_ylim(zlim)
            self.axes['3d_separate'].set_zlim(ylim)
        
        if self.display_plots['xy_separate']:
            self.figs['xy_separate'] = plt.figure(figsize=(8, 6))
            self.axes['xy_separate'] = self.figs['xy_separate'].add_subplot(111)
            self.axes['xy_separate'].set_title('XY Projection (side plane)')
            self.axes['xy_separate'].set_xlabel('X')
            self.axes['xy_separate'].set_ylabel('Y')
            self.axes['xy_separate'].set_xlim(xlim)
            self.axes['xy_separate'].set_ylim(ylim)
        
        if self.display_plots['xz_separate']:
            self.figs['xz_separate'] = plt.figure(figsize=(8, 6))
            self.axes['xz_separate'] = self.figs['xz_separate'].add_subplot(111)
            self.axes['xz_separate'].set_title('XZ Projection (ground plane)')
            self.axes['xz_separate'].set_xlabel('X')
            self.axes['xz_separate'].set_ylabel('Z')
            self.axes['xz_separate'].set_xlim(xlim)
            self.axes['xz_separate'].set_ylim(zlim)
        
        if self.display_plots['yx_separate']:
            self.figs['yx_separate'] = plt.figure(figsize=(8, 6))
            self.axes['yx_separate'] = self.figs['yx_separate'].add_subplot(111)
            self.axes['yx_separate'].set_title('YX Projection (front plane)')
            self.axes['yx_separate'].set_xlabel('Y')
            self.axes['yx_separate'].set_ylabel('Z')
            self.axes['yx_separate'].set_xlim(ylim)
            self.axes['yx_separate'].set_ylim(zlim)
        
        # Store object plot references
        self.object_plots = {
            key: {} for key in [
                '3d', '3d_separate', 
                'xy', 'xy_separate', 
                'xz', 'xz_separate', 
                'yx', 'yx_separate'
            ]
        }
        
        self.object_trajectories = {
            key: {} for key in [
                '3d', '3d_separate', 
                'xy', 'xy_separate', 
                'xz', 'xz_separate', 
                'yx', 'yx_separate'
            ]
        }
        
        self.annotations = {
            key: {} for key in [
                '3d', '3d_separate', 
                'xy', 'xy_separate', 
                'xz', 'xz_separate', 
                'yx', 'yx_separate'
            ]
        }

        self.icons = {
            key: {} for key in [
                '3d', '3d_separate', 
                'xy', 'xy_separate', 
                'xz', 'xz_separate', 
                'yx', 'yx_separate'
            ]
        }

    def update_visualization(self, tracked_objects):
        """
        Update 3D visualization with current tracked objects and their projections
        """
        # Clear previous plots for all enabled plots
        for plot_type in self.object_plots:
            if plot_type in self.axes:
                for obj_id in list(self.object_plots[plot_type].keys()):
                    # Remove scatter plot
                    if self.object_plots[plot_type].get(obj_id):
                        self.object_plots[plot_type][obj_id].remove()
                    
                    # Remove trajectory plot
                    if self.object_trajectories[plot_type].get(obj_id):
                        self.object_trajectories[plot_type][obj_id][0].remove()
                    
                    # Remove annotation
                    if self.annotations[plot_type].get(obj_id):
                        self.annotations[plot_type][obj_id].remove()

                    # Remove annotation
                    if self.icons[plot_type].get(obj_id):
                        self.icons[plot_type][obj_id].remove()
                
                # Clear dictionaries
                self.object_plots[plot_type].clear()
                self.object_trajectories[plot_type].clear()
                self.annotations[plot_type].clear()
                self.icons[plot_type].clear()
        
        # Plot each object
        for obj_id, obj in tracked_objects.items():
            # positions = np.array(obj.world_3d_position[-1])
            positions = np.array(obj.position_3d[-1])
            x, y, z = positions[0], positions[1], positions[2]
            color = tuple(c / 255.0 for c in reversed(obj.color))
            
            # Plotting function for each plot type
            def plot_object(plot_type, ax):
                # Determine coordinates based on plot type
                if plot_type in ['3d', '3d_separate']:
                    scatter_x = x
                    scatter_y = z
                    scatter_z = y
                    self.object_plots[plot_type][obj_id] = ax.scatter(
                        scatter_x, scatter_y, scatter_z, 
                        color=color, alpha=0.7, s=200
                    )
                    
                    # 3D plot requires x, y, z, and s arguments
                    self.annotations[plot_type][obj_id] = ax.text(
                        x, z, y, 
                        f'{obj_id}', 
                        color='white', ha='center', va='center', fontsize=8
                    )
                else:
                    # 2D plots
                    scatter_x = x if plot_type in ['xy', 'xy_separate', 'xz', 'xz_separate'] else y
                    scatter_y = z if plot_type in ['xz', 'xz_separate', 'yx', 'yx_separate'] else y
                    
                    self.object_plots[plot_type][obj_id] = ax.scatter(
                        scatter_x, scatter_y, 
                        color=color, alpha=0.7, s=200
                    )
                    
                    # 2D plot text
                    self.annotations[plot_type][obj_id] = ax.text(
                        scatter_x, scatter_y, 
                        f'{obj_id}', 
                        color='white', ha='center', va='center', fontsize=8
                    )

                    if obj.type == 2:
                        icon = mpimg.imread('visualization_icons/car_2.png') 
                        image_box = OffsetImage(icon, zoom=0.001)
                    elif obj.type == 1:
                        icon = mpimg.imread('visualization_icons/bike.png')
                        image_box = OffsetImage(icon, zoom=0.03)
                    else:
                        icon = mpimg.imread('visualization_icons/person.png')
                        image_box = OffsetImage(icon, zoom=0.03)
                        
                    image_box = OffsetImage(icon, zoom=0.03)
                    self.icons[plot_type][obj_id] = ax.add_artist(AnnotationBbox(
                        image_box, 
                        (x + 0.2, z - 0.3),  # Position next to the point
                        frameon=False,  # No bounding box around the icon
                        zorder=3
                    ))
                
                # Trajectory plot
                traj_x = x if plot_type in ['3d', '3d_separate', 'xy', 'xy_separate', 'xz', 'xz_separate'] else y
                traj_y = z if plot_type in ['3d', '3d_separate', 'xz', 'xz_separate', 'yx', 'yx_separate'] else y
                
                self.object_trajectories[plot_type][obj_id] = ax.plot(
                    traj_x, traj_y, color=color, linewidth=2, alpha=0.5
                )
            
            # Plot on enabled plots
            if '3d' in self.axes:
                plot_object('3d', self.axes['3d'])
            if '3d_separate' in self.axes:
                plot_object('3d_separate', self.axes['3d_separate'])
            if 'xy' in self.axes:
                plot_object('xy', self.axes['xy'])
            if 'xy_separate' in self.axes:
                plot_object('xy_separate', self.axes['xy_separate'])
            if 'xz' in self.axes:
                plot_object('xz', self.axes['xz'])
            if 'xz_separate' in self.axes:
                plot_object('xz_separate', self.axes['xz_separate'])
            if 'yx' in self.axes:
                plot_object('yx', self.axes['yx'])
            if 'yx_separate' in self.axes:
                plot_object('yx_separate', self.axes['yx_separate'])
        
        # Update legends
        if '3d' in self.axes:
            self.axes['3d'].legend()

    def render(self, tracked_objects):
        """
        Update and render the visualization
        """
        self.update_visualization(tracked_objects)
        
        # Draw only enabled figures
        for fig_name, fig in self.figs.items():
            plt.figure(fig.number)
            plt.draw()
        
        plt.pause(0.01)  # Small pause to update the plot

    def close(self):
        """
        Close the visualization
        """
        plt.ioff()
        # Close only enabled figures
        for fig in self.figs.values():
            plt.close(fig)

    def change_display_configuration(self, new_display_plots):
        """
        Change which plots are displayed during runtime
        
        Parameters:
        -----------
        new_display_plots : dict
            Dictionary with plot display configuration
            Example: {'3d_separate': False, 'xy_separate': False}
        """
        # Update display configuration
        self.display_plots.update(new_display_plots)
        
        # Close figures that are no longer needed
        for plot_name, is_displayed in new_display_plots.items():
            if not is_displayed and plot_name in self.figs:
                plt.close(self.figs[plot_name])
                del self.figs[plot_name]
                
                # Remove corresponding axes
                if plot_name in self.axes:
                    del self.axes[plot_name]
