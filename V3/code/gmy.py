import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from agentR import Agent
from sim import restart, next
import matplotlib.pyplot as plt
import csv

class ExperimentConfig:
    #HYPERPARAMETERS
    def __init__(self, 
                 experiment_name=f"experiment_{int(time.time())}",
                 save_dir="experiment_data",
                 hyperparameter_sets=None,
                 episodes_per_config=1000,  #number of games
                 max_time_steps=1000,
                 save_frequency=10):
        
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.episodes_per_config = episodes_per_config
        self.max_time_steps = max_time_steps
        self.save_frequency = save_frequency
        
        # Define different hyperparameter configurations to test
        if hyperparameter_sets is None:
            self.hyperparameter_sets = [
                {"gamma": 0.99, "epsilon": 0.9, "batch_size": 64, "eps_end": 0.01, "lr": 0.0003, "target_update": 100},
                {"gamma": 0.95, "epsilon": 0.8, "batch_size": 64, "eps_end": 0.001, "lr": 0.0003, "target_update": 200},
                {"gamma": 0.9, "epsilon": 0.8, "batch_size": 64, "eps_end": 0.0001, "lr": 0.0003, "target_update": 50},
                {"gamma": 0.99, "epsilon": 0.9, "batch_size": 128, "eps_end": 0.01, "lr": 0.0001, "target_update": 150},
                {"gamma": 0.9, "epsilon": 0.7, "batch_size": 32, "eps_end": 0.001, "lr": 0.001, "target_update": 75}
            ]
        else:
            self.hyperparameter_sets = hyperparameter_sets
            
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/{self.experiment_name}", exist_ok=True)
        os.makedirs(f"{self.save_dir}/{self.experiment_name}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/{self.experiment_name}/plots", exist_ok=True)
        os.makedirs(f"{self.save_dir}/{self.experiment_name}/trajectories", exist_ok=True)
        os.makedirs(f"{self.save_dir}/{self.experiment_name}/weights", exist_ok=True)
        os.makedirs(f"{self.save_dir}/{self.experiment_name}/successful_trajectories", exist_ok=True)
        
        # Save configuration
        self.save_config()
        
    def save_config(self):
        config_data = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameter_sets": self.hyperparameter_sets,
            "episodes_per_config": self.episodes_per_config,
            "max_time_steps": self.max_time_steps
        }
        
        with open(f"{self.save_dir}/{self.experiment_name}/config.json", 'w') as f:
            json.dump(config_data, f, indent=4)


def save_episode_data(config_id, episode_id, episode_data, config):
    """Save detailed data about an episode"""
    episode_dir = f"{config.save_dir}/{config.experiment_name}/trajectories/config_{config_id}"
    os.makedirs(episode_dir, exist_ok=True)
    
    # Save trajectory coordinates
    with open(f"{episode_dir}/episode_{episode_id}_trajectory.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for coord in episode_data["trajectory"]:
            writer.writerow(coord)
    
    # Save states, actions, rewards
    with open(f"{episode_dir}/episode_{episode_id}_data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'x', 'y', 'x_velocity', 'y_velocity', 'target_distance', 
                         'action', 'reward', 'cumulative_reward'])
        
        for i, (state, action, reward, cum_reward) in enumerate(zip(
                episode_data["states"], 
                episode_data["actions"], 
                episode_data["rewards"],
                episode_data["cumulative_rewards"])):
            
            writer.writerow([i, state[0], state[1], state[2], state[3], state[4], 
                             action, reward, cum_reward])


def save_successful_trajectory(config_id, episode_id, episode_data, config, success_count):
    """Save trajectory data when ship successfully reaches the goal"""
    # Create directory for successful trajectories
    success_dir = f"{config.save_dir}/{config.experiment_name}/successful_trajectories/config_{config_id}"
    os.makedirs(success_dir, exist_ok=True)
    
    # Save trajectory coordinates
    with open(f"{success_dir}/success_{success_count}_episode_{episode_id}_trajectory.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for coord in episode_data["trajectory"]:
            writer.writerow(coord)
    
    # Save complete episode data with all details
    with open(f"{success_dir}/success_{success_count}_episode_{episode_id}_complete.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'x', 'y', 'x_velocity', 'y_velocity', 'target_distance', 
                         'action', 'reward', 'cumulative_reward'])
        
        for i, (state, action, reward, cum_reward) in enumerate(zip(
                episode_data["states"], 
                episode_data["actions"], 
                episode_data["rewards"],
                episode_data["cumulative_rewards"])):
            
            writer.writerow([i, state[0], state[1], state[2], state[3], state[4], 
                             action, reward, cum_reward])
    
    # Save just the sequence of actions for easy reproduction
    with open(f"{success_dir}/success_{success_count}_episode_{episode_id}_actions.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'action'])
        for i, action in enumerate(episode_data["actions"]):
            writer.writerow([i, action])
    
    # Save initial state and other metadata
    if len(episode_data["states"]) > 0:
        initial_state = episode_data["states"][0]
        metadata = {
            "config_id": config_id,
            "episode_id": episode_id,
            "success_number": success_count,
            "total_steps": len(episode_data["actions"]),
            "total_reward": sum(episode_data["rewards"]),
            "initial_state": {
                "x": float(initial_state[0]),
                "y": float(initial_state[1]),
                "xv": float(initial_state[2]),
                "yv": float(initial_state[3]),
                "target_distance": float(initial_state[4])
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{success_dir}/success_{success_count}_episode_{episode_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)


def save_network_weights(agent, config, config_id, episode=None):
    """Save the network weights to CSV files for analysis"""
    weights_dir = f"{config.save_dir}/{config.experiment_name}/weights/config_{config_id}"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Create a filename suffix based on whether this is a specific episode or final weights
    suffix = f"_episode_{episode}" if episode is not None else "_final"
    
    # Save Q_eval network weights
    for name, param in agent.Q_eval.named_parameters():
        # Convert the parameter tensor to a numpy array
        weight_data = param.data.cpu().numpy()
        
        # Save to CSV - handle different dimensionality
        if len(weight_data.shape) == 2:  # 2D weight matrices
            pd.DataFrame(weight_data).to_csv(
                f"{weights_dir}/Q_eval_{name.replace('.', '_')}{suffix}.csv", 
                index=False
            )
        else:  # 1D bias vectors
            pd.DataFrame(weight_data.reshape(1, -1)).to_csv(
                f"{weights_dir}/Q_eval_{name.replace('.', '_')}{suffix}.csv", 
                index=False
            )
    
    # Save Q_target network weights
    for name, param in agent.Q_target.named_parameters():
        # Convert the parameter tensor to a numpy array
        weight_data = param.data.cpu().numpy()
        
        # Save to CSV - handle different dimensionality
        if len(weight_data.shape) == 2:  # 2D weight matrices
            pd.DataFrame(weight_data).to_csv(
                f"{weights_dir}/Q_target_{name.replace('.', '_')}{suffix}.csv", 
                index=False
            )
        else:  # 1D bias vectors
            pd.DataFrame(weight_data.reshape(1, -1)).to_csv(
                f"{weights_dir}/Q_target_{name.replace('.', '_')}{suffix}.csv", 
                index=False
            )


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def plot_learning_curve(x, scores, epsilons, filename, config_info=None, lines=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Episodes", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
    
    ax2.scatter(x, running_avg, color="C1", s=10)
    ax2.plot(x, running_avg, color="C1", alpha=0.5)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)
            
    # Add configuration information
    if config_info is not None:
        config_text = f"Gamma: {config_info['gamma']}, Epsilon: {config_info['epsilon']}\n" \
                     f"Batch Size: {config_info['batch_size']}, LR: {config_info['lr']}, " \
                     f"Target Update: {config_info.get('target_update', 'N/A')}\n" \
                     f"Successes: {config_info.get('goal_reached_count', 0)}/{config_info.get('total_episodes', 0)}"
        
        # Add training time if available
        if 'training_time' in config_info:
            config_text += f"\nTraining Time: {format_time(config_info['training_time'])}"
            
        plt.figtext(0.5, 0.01, config_text, ha="center", fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.title(f"Learning Curve - Config {config_info.get('config_id', '')}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_successful_trajectories(config, config_id, success_trajectories, planet_pos=(300, 300)):
    """Plot all successful trajectories for a configuration"""
    if not success_trajectories:
        return  # No successful trajectories to plot
    
    plt.figure(figsize=(10, 10))
    
    # Plot planet
    planet_circle = plt.Circle(planet_pos, 40, color='blue', alpha=0.5)  # Approximate planet size
    plt.gca().add_patch(planet_circle)
    
    # Plot each trajectory
    for idx, (episode_id, trajectory) in enumerate(success_trajectories):
        # Extract x and y coordinates
        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]
        
        # Plot trajectory with gradient color to show direction
        points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Use a colormap to show progression
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', alpha=0.7)
        lc.set_array(np.linspace(0, 1, len(x_coords)-1))
        plt.gca().add_collection(lc)
        
        # Mark start and end points
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=5, label=f"Start {idx+1}" if idx == 0 else "")
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=5, label=f"End {idx+1}" if idx == 0 else "")
    
    plt.title(f"Successful Trajectories for Configuration {config_id}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio
    
    # Set reasonable limits based on trajectory data
    all_x = [point[0] for _, traj in success_trajectories for point in traj]
    all_y = [point[1] for _, traj in success_trajectories for point in traj]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    plt.xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
    plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    plt.savefig(f"{config.save_dir}/{config.experiment_name}/successful_trajectories/config_{config_id}/trajectory_plot.png")
    plt.close()


def plot_comparative_results(experiment_config, results_data):
    """Plot comparative results across different configurations"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot average reward per episode
    for config_id, data in results_data.items():
        x = list(range(1, len(data['episode_rewards']) + 1))
        running_avg = np.empty(len(data['episode_rewards']))
        for t in range(len(data['episode_rewards'])):
            running_avg[t] = np.mean(data['episode_rewards'][max(0, t-20):(t+1)])
        
        # Include target update frequency and success count in label
        target_update = experiment_config.hyperparameter_sets[config_id].get('target_update', 'N/A')
        success_count = sum(data['successes'])
        total_episodes = len(data['successes'])
        
        # Add timing info if available
        time_info = ""
        if 'training_time' in data:
            time_info = f", Time={format_time(data['training_time'])}"
            
        label = f"Config {config_id} (TU={target_update}, Success={success_count}/{total_episodes}{time_info})"
        axes[0].plot(x, running_avg, label=label)
    
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Comparative Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot success rate (moving average)
    for config_id, data in results_data.items():
        x = list(range(1, len(data['successes']) + 1))
        success_rate = np.empty(len(data['successes']))
        for t in range(len(data['successes'])):
            success_rate[t] = np.mean(data['successes'][max(0, t-20):(t+1)])
        
        # Include target update frequency in label
        target_update = experiment_config.hyperparameter_sets[config_id].get('target_update', 'N/A')
        success_count = sum(data['successes'])
        total_episodes = len(data['successes'])
        
        # Add timing info if available
        time_info = ""
        if 'training_time' in data:
            time_info = f", Time={format_time(data['training_time'])}"
            
        label = f"Config {config_id} (TU={target_update}, Success={success_count}/{total_episodes}{time_info})"
        axes[1].plot(x, success_rate, label=label)
    
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Success Rate (20 episode window)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/plots/comparative_results.png")
    plt.close()


def plot_config_timing(experiment_config, results_data):
    """Create a bar chart showing training time for each configuration"""
    # Extract timing data
    config_ids = []
    training_times = []
    success_counts = []
    
    for config_id, data in results_data.items():
        if 'training_time' in data:
            config_ids.append(config_id)
            training_times.append(data['training_time'])
            success_counts.append(sum(data['successes']))
    
    if not config_ids:  # No timing data available
        return
    
    # Calculate success per hour for each config
    total_episodes = experiment_config.episodes_per_config
    success_rates = [count / total_episodes for count in success_counts]
    success_per_hour = [rate / (time/3600) if time > 0 else 0 for rate, time in zip(success_rates, training_times)]
    
    # Create bar chart for training times
    fig, ax = plt.subplots(figsize=(12, 6))
    norm = plt.Normalize(min(success_rates) if success_rates else 0, max(success_rates) if success_rates else 1)
    
    bars = ax.bar(config_ids, training_times)
    
    # Color bars by success rate
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(norm(success_rates[i])))
    
    # Add value labels on top of bars
    for i, v in enumerate(training_times):
        ax.text(i, v + 0.1*max(training_times), 
                f"{format_time(v)}\n{success_counts[i]} successes", 
                ha='center', fontsize=9)
    
    ax.set_title('Training Time by Configuration')
    ax.set_xlabel('Configuration ID')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_xticks(config_ids)
    ax.grid(axis='y', alpha=0.3)
    
    # Add a colorbar to show success rate
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Success Rate')
    
    plt.tight_layout()
    plt.savefig(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/plots/configuration_timing.png")
    plt.close()
    
    # Create a second chart for success per hour
    fig, ax = plt.subplots(figsize=(12, 6))
    norm = plt.Normalize(min(success_rates) if success_rates else 0, max(success_rates) if success_rates else 1)
    
    bars = ax.bar(config_ids, success_per_hour)
    
    # Color bars by success rate
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(norm(success_rates[i])))
    
    # Add value labels
    for i, v in enumerate(success_per_hour):
        ax.text(i, v + 0.1*max(success_per_hour), 
                f"{v:.2f}\nsuccess/hour", 
                ha='center', fontsize=9)
    
    ax.set_title('Success Rate per Hour by Configuration')
    ax.set_xlabel('Configuration ID')
    ax.set_ylabel('Successful Episodes per Hour')
    ax.set_xticks(config_ids)
    ax.grid(axis='y', alpha=0.3)
    
    # Add a colorbar to show success rate
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Success Rate')
    
    plt.tight_layout()
    plt.savefig(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/plots/success_per_hour.png")
    plt.close()


def save_summary_statistics(experiment_config, results_data):
    """Save summary statistics for each configuration"""
    summary_data = []
    
    for config_id, data in results_data.items():
        # Calculate success count and rate
        success_count = sum(data['successes'])
        total_episodes = len(data['successes'])
        success_rate = success_count / total_episodes if total_episodes > 0 else 0
        
        # Get training time if available
        training_time = data.get('training_time', 0)
        training_time_formatted = format_time(training_time)
        
        # Calculate efficiency metrics if time data is available
        episodes_per_hour = (total_episodes / training_time) * 3600 if training_time > 0 else 0
        success_per_hour = (success_count / training_time) * 3600 if training_time > 0 else 0
        
        summary = {
            'config_id': config_id,
            'avg_reward': np.mean(data['episode_rewards']),
            'max_reward': np.max(data['episode_rewards']),
            'success_count': success_count,
            'total_episodes': total_episodes,
            'success_rate': success_rate,
            'training_time_seconds': training_time,
            'training_time': training_time_formatted,
            'episodes_per_hour': episodes_per_hour,
            'success_per_hour': success_per_hour,
            'avg_steps_to_completion': np.mean([steps for steps, success in zip(data['episode_steps'], data['successes']) 
                                              if success and steps < experiment_config.max_time_steps]),
            'hyperparameters': experiment_config.hyperparameter_sets[config_id]
        }
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/summary_statistics.csv", index=False)
    
    with open(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/summary_report.txt", 'w') as f:
        f.write(f"Experiment: {experiment_config.experiment_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Summary of Results:\n")
        f.write("===================\n\n")
        
        for summary in summary_data:
            f.write(f"Configuration {summary['config_id']}:\n")
            f.write(f"  Hyperparameters: {summary['hyperparameters']}\n")
            f.write(f"  Average Reward: {summary['avg_reward']:.2f}\n")
            f.write(f"  Maximum Reward: {summary['max_reward']:.2f}\n")
            f.write(f"  Goal Reached: {summary['success_count']} / {summary['total_episodes']} episodes ({summary['success_rate']:.2%})\n")
            f.write(f"  Training Time: {summary['training_time']} ({summary['training_time_seconds']:.1f} seconds)\n")
            f.write(f"  Episodes Per Hour: {summary['episodes_per_hour']:.1f}\n")
            f.write(f"  Successful Episodes Per Hour: {summary['success_per_hour']:.1f}\n")
            
            if summary['success_count'] > 0:
                f.write(f"  Average Steps to Goal (successful episodes): {summary['avg_steps_to_completion']:.1f}\n\n")
            else:
                f.write(f"  Average Steps to Goal: N/A (no successful episodes)\n\n")


if __name__ == "__main__":
    # Record experiment start time
    experiment_start_time = time.time()
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        experiment_name=f"orbital_nav_target_network_{int(time.time())}",
        episodes_per_config=1000  # Increased from 100 to 1000
    )
    
    # Dictionary to store results for each configuration
    results_data = {}
    
    # Create a file to log configuration times
    with open(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/config_timing_log.txt", 'w') as timing_log:
        timing_log.write(f"{'='*50}\n\n")
    
    # Run experiments for each hyperparameter configuration
    for config_id, hyperparams in enumerate(experiment_config.hyperparameter_sets):
        print(f"\n{'='*50}")
        print(f"Starting configuration {config_id}: {hyperparams}")
        print(f"{'='*50}")
        
        # Record start time for this configuration
        config_start_time = time.time()
        
        # Create agent with current hyperparameters
        agent = Agent(
            gamma=hyperparams["gamma"], 
            epsilon=hyperparams["epsilon"],
            batch_size=hyperparams["batch_size"], 
            n_actions=5, 
            eps_end=hyperparams["eps_end"],
            input_dims=[5], 
            lr=hyperparams["lr"],
            target_update=hyperparams.get("target_update", 100),
            load_model=False,
            model_dir=f"{experiment_config.save_dir}/{experiment_config.experiment_name}/models"
        )
        
        # Initialize metrics for this configuration
        scores = []
        eps_history = []
        episode_steps = []
        successes = []
        goal_reached_count = 0
        successful_trajectories = []  # Store successful trajectory data for plotting
        
        # Create results directory for this configuration
        config_dir = f"{experiment_config.save_dir}/{experiment_config.experiment_name}/trajectories/config_{config_id}"
        os.makedirs(config_dir, exist_ok=True)
        
        # Training loop
        for episode in range(experiment_config.episodes_per_config):
            print(f"Config {config_id}, Episode {episode}/{experiment_config.episodes_per_config}")
            
            # Initialize episode data collection
            episode_data = {
                "trajectory": [],
                "states": [],
                "actions": [],
                "rewards": [],
                "cumulative_rewards": []
            }
            
            time_step = 0
            score = 0
            cumulative_reward = 0
            done = False
            success = False
            
            observation = restart(True, True)
            
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = next(action)
                
                score += reward
                cumulative_reward += reward
                
                episode_data["trajectory"].append((observation[0], observation[1]))
                episode_data["states"].append(observation)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["cumulative_rewards"].append(cumulative_reward)
                
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                
                observation = observation_
                
                if time_step >= experiment_config.max_time_steps:
                    print("TIME OUT")
                    done = True
                
                if done and reward > 5:  
                    success = True
                    goal_reached_count += 1
                    
                    save_successful_trajectory(config_id, episode, episode_data, experiment_config, goal_reached_count)
                    
                    successful_trajectories.append((episode, episode_data["trajectory"]))
                    
                    print(f"ZIEL ERREICHT! Notige Eps: {goal_reached_count}")
                
                time_step += 1
            
            if episode % experiment_config.save_frequency == 0 or episode == experiment_config.episodes_per_config - 1:
                save_episode_data(config_id, episode, episode_data,
                                  experiment_config)
            #check every 25 times
            if episode % 25 == 0:
                model_path = f"{experiment_config.save_dir}/{experiment_config.experiment_name}/models/config_{config_id}_episode_{episode}.pt"
                agent.save_model(model_path)
                
                save_network_weights(agent, experiment_config, config_id, episode)
            
            scores.append(score)
            eps_history.append(agent.epsilon)
            episode_steps.append(time_step)
            successes.append(1 if success else 0)
            
            # Calculate elapsed time for this configuration so far
            current_elapsed = time.time() - config_start_time
            eta = (current_elapsed / (episode + 1)) * (experiment_config.episodes_per_config - episode - 1)
            
            print(f'Episode {episode}, Score: {score:.2f}, Avg Score: {np.mean(scores[-20:]):.2f}, Epsilon: {agent.epsilon:.4f}, Steps: {time_step}, Success: {success}')
            print(f'Config {config_id} Time: {format_time(current_elapsed)}, ETA: {format_time(eta)}')
        
        # Calculate total time for this configuration
        config_end_time = time.time()
        config_elapsed_time = config_end_time - config_start_time
        
        # Log the configuration timing
        with open(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/config_timing_log.txt", 'a') as timing_log:
            timing_log.write(f"Configuration {config_id}:\n")
            timing_log.write(f"  Hyperparameters: {hyperparams}\n")
            timing_log.write(f"  Start Time: {datetime.fromtimestamp(config_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            timing_log.write(f"  End Time: {datetime.fromtimestamp(config_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            timing_log.write(f"  Total Time: {format_time(config_elapsed_time)} ({config_elapsed_time:.2f} seconds)\n")
            timing_log.write(f"  Success Rate: {goal_reached_count}/{experiment_config.episodes_per_config} ({goal_reached_count/experiment_config.episodes_per_config:.2%})\n")
            timing_log.write(f"  Episodes Per Hour: {(experiment_config.episodes_per_config/config_elapsed_time)*3600:.2f}\n")
            timing_log.write(f"  Successful Episodes Per Hour: {(goal_reached_count/config_elapsed_time)*3600:.2f}\n\n")
        
        # Print timing info to console
        print(f"\nConfiguration {config_id} completed in {format_time(config_elapsed_time)} ({config_elapsed_time:.2f} seconds)")
        print(f"Episodes Per Hour: {(experiment_config.episodes_per_config/config_elapsed_time)*3600:.2f}")
        print(f"Successful Episodes Per Hour: {(goal_reached_count/config_elapsed_time)*3600:.2f}")
        
        # Save the final model for this configuration
        final_model_path = f"{experiment_config.save_dir}/{experiment_config.experiment_name}/models/config_{config_id}_final.pt"
        agent.save_model(final_model_path)
        
        # Save final network weights
        save_network_weights(agent, experiment_config, config_id)
        
        # Create a combined plot of all successful trajectories
        if successful_trajectories:
            plot_successful_trajectories(experiment_config, config_id, successful_trajectories)
        
        # Plot learning curve for this configuration
        x = list(range(1, experiment_config.episodes_per_config + 1))
        plot_filename = f"{experiment_config.save_dir}/{experiment_config.experiment_name}/plots/config_{config_id}_learning_curve.png"
        plot_learning_curve(x, scores, eps_history, plot_filename, 
                           config_info={
                               "config_id": config_id, 
                               **hyperparams,
                               "goal_reached_count": goal_reached_count,
                               "total_episodes": experiment_config.episodes_per_config,
                               "training_time": config_elapsed_time
                           })
        
        # Print final success statistics
        print(f"\nConfig {config_id} Final Statistics:")
        print(f"Goal reached in {goal_reached_count} out of {experiment_config.episodes_per_config} episodes ({goal_reached_count/experiment_config.episodes_per_config:.2%})")
        print(f"Average reward: {np.mean(scores):.2f}")
        if goal_reached_count > 0:
            success_step_indices = [i for i, success in enumerate(successes) if success]
            avg_success_steps = np.mean([episode_steps[i] for i in success_step_indices])
            print(f"Average steps to reach goal (successful episodes): {avg_success_steps:.1f}")
            print(f"Successful trajectory data saved in {experiment_config.save_dir}/{experiment_config.experiment_name}/successful_trajectories/config_{config_id}/")
        
        # Store results for this configuration
        results_data[config_id] = {
            "episode_rewards": scores,
            "epsilons": eps_history,
            "episode_steps": episode_steps,
            "successes": successes,
            "goal_reached_count": goal_reached_count,
            "training_time": config_elapsed_time,
            "hyperparams": hyperparams
        }
        
        # Save raw results for this configuration
        results_df = pd.DataFrame({
            "episode": x,
            "reward": scores,
            "epsilon": eps_history,
            "steps": episode_steps,
            "success": successes
        })
        results_df.to_csv(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/config_{config_id}_results.csv", index=False)
    
    # Calculate total experiment time
    experiment_end_time = time.time()
    experiment_elapsed_time = experiment_end_time - experiment_start_time
    
    # Save overall experiment timing info
    with open(f"{experiment_config.save_dir}/{experiment_config.experiment_name}/config_timing_log.txt", 'a') as timing_log:
        timing_log.write(f"\n{'='*50}\n")
        timing_log.write(f"Overall Experiment Timing:\n")
        timing_log.write(f"  Start Time: {datetime.fromtimestamp(experiment_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        timing_log.write(f"  End Time: {datetime.fromtimestamp(experiment_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        timing_log.write(f"  Total Time: {format_time(experiment_elapsed_time)} ({experiment_elapsed_time:.2f} seconds)\n")
        
        # Calculate overall statistics
        total_episodes = experiment_config.episodes_per_config * len(experiment_config.hyperparameter_sets)
        total_successes = sum(data["goal_reached_count"] for data in results_data.values())
        
        timing_log.write(f"  Total Episodes: {total_episodes}\n")
        timing_log.write(f"  Total Successful Episodes: {total_successes}\n")
        timing_log.write(f"  Overall Success Rate: {total_successes/total_episodes:.2%}\n")
        timing_log.write(f"  Episodes Per Hour: {(total_episodes/experiment_elapsed_time)*3600:.2f}\n")
        timing_log.write(f"  Successful Episodes Per Hour: {(total_successes/experiment_elapsed_time)*3600:.2f}\n")
    
    # After all configurations are done, create comparative plots
    plot_comparative_results(experiment_config, results_data)
    
    # Create a plot showing timing data
    plot_config_timing(experiment_config, results_data)
    
    # Generate and save summary statistics
    save_summary_statistics(experiment_config, results_data)
    
    print(f"\n{'='*50}")
    print(f"Experiment complete in {format_time(experiment_elapsed_time)} ({experiment_elapsed_time:.2f} seconds)")
    print(f"Results saved to {experiment_config.save_dir}/{experiment_config.experiment_name}/")
    print(f"{'='*50}")