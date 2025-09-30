#!/usr/bin/env python3
"""
Real-time loss plotting utility for training monitoring.
Automatically saves plots to disk so progress can be viewed even if training is interrupted.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional
import json


class LossPlotter:
    """
    Real-time loss plotter that tracks training and evaluation losses.
    Automatically saves plots and loss data to disk.
    """
    
    def __init__(self, save_dir: str = ".", plot_filename: str = "loss_curve.png", 
                 data_filename: str = "loss_data.json", title: str = "Training Loss Curve"):
        """
        Initialize the loss plotter.
        
        Args:
            save_dir: Directory to save plots and data
            plot_filename: Name of the plot image file
            data_filename: Name of the JSON file to store loss data
            title: Title for the plot
        """
        self.save_dir = save_dir
        self.plot_path = os.path.join(save_dir, plot_filename)
        self.data_path = os.path.join(save_dir, data_filename)
        self.title = title
        
        # Loss tracking
        self.train_losses = []
        self.train_iterations = []
        self.eval_losses = []
        self.eval_iterations = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load existing data if available
        self._load_existing_data()
        
        # Set up matplotlib for non-interactive plotting
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.style.use('default')
    
    def _load_existing_data(self):
        """Load existing loss data from disk if available."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.train_losses = data.get('train_losses', [])
                    self.train_iterations = data.get('train_iterations', [])
                    self.eval_losses = data.get('eval_losses', [])
                    self.eval_iterations = data.get('eval_iterations', [])
                print(f"✅ Loaded existing loss data from {self.data_path}")
            except Exception as e:
                print(f"⚠️  Could not load existing loss data: {e}")
    
    def _save_data(self):
        """Save current loss data to disk."""
        data = {
            'train_losses': self.train_losses,
            'train_iterations': self.train_iterations,
            'eval_losses': self.eval_losses,
            'eval_iterations': self.eval_iterations
        }
        
        try:
            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save loss data: {e}")
    
    def add_train_loss(self, iteration: int, loss: float):
        """
        Add a training loss point.
        
        Args:
            iteration: Training iteration number
            loss: Training loss value
        """
        self.train_iterations.append(iteration)
        self.train_losses.append(loss)
        self._update_plot()
        self._save_data()
    
    def add_eval_loss(self, iteration: int, loss: float):
        """
        Add an evaluation loss point.
        
        Args:
            iteration: Training iteration number
            loss: Evaluation loss value
        """
        self.eval_iterations.append(iteration)
        self.eval_losses.append(loss)
        self._update_plot()
        self._save_data()
    
    def add_losses(self, iteration: int, train_loss: Optional[float] = None, 
                   eval_loss: Optional[float] = None):
        """
        Add both training and evaluation losses at once.
        
        Args:
            iteration: Training iteration number
            train_loss: Training loss value (optional)
            eval_loss: Evaluation loss value (optional)
        """
        if train_loss is not None:
            self.train_iterations.append(iteration)
            self.train_losses.append(train_loss)
        
        if eval_loss is not None:
            self.eval_iterations.append(iteration)
            self.eval_losses.append(eval_loss)
        
        self._update_plot()
        self._save_data()
    
    def _update_plot(self):
        """Update and save the loss plot."""
        try:
            # Clear the current plot
            self.ax.clear()
            
            # Plot training losses
            if self.train_losses:
                self.ax.plot(self.train_iterations, self.train_losses, 
                           'b-', label='Training Loss', linewidth=2, alpha=0.8)
                self.ax.scatter(self.train_iterations[-5:], self.train_losses[-5:], 
                              c='blue', s=20, alpha=0.6)
            
            # Plot evaluation losses
            if self.eval_losses:
                self.ax.plot(self.eval_iterations, self.eval_losses, 
                           'r-', label='Evaluation Loss', linewidth=2, alpha=0.8)
                self.ax.scatter(self.eval_iterations[-3:], self.eval_losses[-3:], 
                              c='red', s=30, alpha=0.8)
            
            # Formatting
            self.ax.set_xlabel('Iteration', fontsize=12)
            self.ax.set_ylabel('Loss', fontsize=12)
            self.ax.set_title(self.title, fontsize=14, fontweight='bold')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(fontsize=11)
            
            # Add current loss values as text
            if self.train_losses:
                current_train = self.train_losses[-1]
                self.ax.text(0.02, 0.98, f'Latest Train Loss: {current_train:.4f}', 
                           transform=self.ax.transAxes, fontsize=10, 
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='lightblue', alpha=0.8))
            
            if self.eval_losses:
                current_eval = self.eval_losses[-1]
                self.ax.text(0.02, 0.88, f'Latest Eval Loss: {current_eval:.4f}', 
                           transform=self.ax.transAxes, fontsize=10, 
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='lightcoral', alpha=0.8))
            
            # Set reasonable y-axis limits
            all_losses = self.train_losses + self.eval_losses
            if all_losses:
                min_loss = min(all_losses)
                max_loss = max(all_losses)
                margin = (max_loss - min_loss) * 0.1
                # self.ax.set_ylim(max(0, min_loss - margin), max_loss + margin)
            
            # Tight layout and save
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=100, bbox_inches='tight')
            plt.draw()
            plt.pause(0.01)  # Small pause to allow plot to update
            
        except Exception as e:
            print(f"⚠️  Could not update plot: {e}")
    
    def get_current_losses(self):
        """
        Get the most recent loss values.
        
        Returns:
            dict: Dictionary with latest train and eval losses
        """
        result = {}
        if self.train_losses:
            result['train_loss'] = self.train_losses[-1]
            result['train_iteration'] = self.train_iterations[-1]
        if self.eval_losses:
            result['eval_loss'] = self.eval_losses[-1]
            result['eval_iteration'] = self.eval_iterations[-1]
        return result
    
    def save_final_plot(self, filename: Optional[str] = None):
        """
        Save a final high-quality version of the plot.
        
        Args:
            filename: Optional custom filename for the final plot
        """
        if filename:
            final_path = os.path.join(self.save_dir, filename)
        else:
            final_path = os.path.join(self.save_dir, "final_" + os.path.basename(self.plot_path))
        
        try:
            plt.savefig(final_path, dpi=300, bbox_inches='tight')
            print(f"💾 Final high-quality plot saved to {final_path}")
        except Exception as e:
            print(f"⚠️  Could not save final plot: {e}")
    
    def close(self):
        """Close the plotter and clean up resources."""
        plt.close(self.fig)
        plt.ioff()  # Turn off interactive mode


# Convenience function for quick setup
def create_loss_plotter(save_dir: str = ".", model_name: str = "GPT") -> LossPlotter:
    """
    Create a loss plotter with default settings.
    
    Args:
        save_dir: Directory to save plots and data
        model_name: Name to include in plot title
    
    Returns:
        LossPlotter instance
    """
    return LossPlotter(
        save_dir=save_dir,
        plot_filename=f"{model_name.lower()}_loss_curve.png",
        data_filename=f"{model_name.lower()}_loss_data.json",
        title=f"{model_name} Training Loss Curve"
    )


# Example usage
if __name__ == "__main__":
    # Demo the loss plotter
    import time
    import random
    
    print("🎯 Demo: Real-time loss plotting")
    plotter = create_loss_plotter(save_dir="./demo_plots", model_name="Demo-GPT")
    
    # Simulate training with decreasing loss
    for i in range(0, 1000, 10):
        # Simulate training loss (decreasing with noise)
        train_loss = 4.0 * np.exp(-i / 500) + 0.1 * random.random()
        plotter.add_train_loss(i, train_loss)
        
        # Add eval loss every 50 iterations
        if i % 50 == 0 and i > 0:
            eval_loss = train_loss + 0.2 + 0.1 * random.random()
            plotter.add_eval_loss(i, eval_loss)
        
        time.sleep(0.1)  # Simulate training time
        
        if i % 100 == 0:
            current = plotter.get_current_losses()
            print(f"Iteration {i}: {current}")
    
    plotter.save_final_plot("demo_final_plot.png")
    plotter.close()
    print("✅ Demo completed!") 