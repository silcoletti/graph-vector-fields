# =============================================================================
# GVF Final Experiment Code
#
# This script reproduces the main empirical validation results for the paper:
# "Graph Vector Fields: A New Framework for Personalised Risk Assessment"
#
# It performs the following steps:
# 1. Generates the "Ultimate Stress Test" synthetic dataset (V12).
# 2. Implements the final, context-aware GVF-MoE architecture and a generalist baseline.
# 3. Trains the GVF-MoE model on the first 80% of the temporal data.
# 4. Evaluates the trained model on the full dataset to extract gating weights.
# 5. Generates a plot (`gating_weights_analysis.png`) visualizing the Gating Network's
#    adaptive behavior during the simulated "lockdown" event.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. V12 Dataset: Ultimate Stress Test with Asymmetric Influence ---
def generate_ultimate_stress_dataset(num_agents=80, num_days=180):
    """
    Generates the V12 "Ultimate Stress Test" dataset.
    This dataset introduces:
    1. "Super-spreaders" with disproportionate influence.
    2. A "lockdown" event positioned to span across the train/test split
       to test the model's adaptability.
    """
    print("--- Generating V12 Ultimate Stress Test Dataset ---")
    num_timesteps = num_days * 4
    grid_size, proximity_radius = 30.0, 4.0

    # The lockdown event is timed to span the train/test split (at 80%)
    lockdown_start = int(num_timesteps * 0.75)
    lockdown_end = lockdown_start + 30 * 4

    feature_names = ['x', 'y', 'speed', 'hr', 'skin_temp', 'pm2_5', 'reported_stress']
    agent_states = np.zeros((num_agents, num_timesteps, len(feature_names)))

    base_immunity = np.random.uniform(0.2, 1.0, num_agents)

    contagious_indices = np.random.choice(num_agents, 10, replace=False)
    is_contagious = np.zeros(num_agents)
    is_contagious[contagious_indices] = 1
    is_super_spreader = np.zeros(num_agents)
    is_super_spreader[np.random.choice(contagious_indices, 2, replace=False)] = 1

    ground_truth_risk = np.zeros((num_agents, num_timesteps, 3))
    agent_positions = np.random.rand(num_agents, 2) * grid_size
    pollution_hotspot = np.random.rand(2) * grid_size
    dataset = []

    for t in range(num_timesteps):
        current_proximity_radius = proximity_radius * (0.1 if lockdown_start <= t < lockdown_end else 1.0)
        edge_index = (torch.cdist(torch.tensor(agent_positions), torch.tensor(agent_positions)) < current_proximity_radius).nonzero(as_tuple=False).t().contiguous()
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        for i in range(num_agents):
            move = np.random.randn(2) * base_immunity[i] * (0.5 if lockdown_start <= t < lockdown_end else 1.0)
            agent_positions[i] = np.clip(agent_positions[i] + move, 0, grid_size - 1)

            speed = np.linalg.norm(move)
            hr = 75 - (base_immunity[i] * 15) + (is_contagious[i] * 10) + (is_super_spreader[i] * 5) + np.random.normal(0, 1.5)
            skin_temp = 37.2 - (base_immunity[i] * 0.8) + (is_contagious[i] * 0.5) + np.random.normal(0, 0.2)

            dist_to_hotspot = np.linalg.norm(agent_positions[i] - pollution_hotspot)
            pm2_5 = 25 * np.exp(-dist_to_hotspot / 10) + np.random.rand() * 5

            reported_stress = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

            agent_states[i, t] = [agent_positions[i, 0], agent_positions[i, 1], speed, hr, skin_temp, pm2_5, reported_stress]

            neighbors = edge_index[1, edge_index[0] == i].numpy()

            env_risk = np.clip(pm2_5 / 30.0, 0, 1)

            phys_risk_internal = 1.0 - base_immunity[i]
            phys_risk_external = 0.0
            if len(neighbors) > 0:
                neighbor_influence = is_contagious[neighbors] + is_super_spreader[neighbors]
                phys_risk_external = np.mean(neighbor_influence)
            phys_risk = np.clip(phys_risk_internal * 0.4 + phys_risk_external * 0.6, 0, 1)

            behav_risk = np.clip(speed / 2.0, 0, 1)
            ground_truth_risk[i, t] = [env_risk, phys_risk, behav_risk]

        dataset.append(Data(x=torch.tensor(agent_states[:, t, :], dtype=torch.float32),
                          edge_index=edge_index,
                          y=torch.tensor(ground_truth_risk[:, t, :], dtype=torch.float32)))

    print("--- Dataset Generation Complete ---\n")
    return dataset, feature_names, lockdown_start, lockdown_end

# --- 2. Final Context-Aware GVF-MoE Architecture ---
class EvolveGCN_O_Expert(nn.Module):
    """
    A faithful implementation of the EvolveGCN-O expert.
    The GCN weights are treated as the hidden state of an RNN,
    allowing them to evolve over time.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rnn = nn.RNN(in_channels * in_channels, in_channels * in_channels)
        self.proj = nn.Linear(in_channels, out_channels)
        self.gcn_conv = GCNConv(in_channels, in_channels, bias=False)

    def forward(self, x, edge_index, W_t=None):
        if W_t is None:
            W_t = self.gcn_conv.lin.weight

        W_t_flat = W_t.flatten().unsqueeze(0).unsqueeze(0)
        _, W_t_flat_new = self.rnn(W_t_flat)
        W_t_new = W_t_flat_new.squeeze(0).reshape(self.in_channels, self.in_channels)

        x = F.linear(x, W_t_new)
        x = self.gcn_conv.propagate(edge_index, x=x, edge_weight=None)

        return self.proj(F.relu(x)), W_t_new

class StaticExpert(nn.Module):
    """A standard GAT-based expert for environmental and behavioral data."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gat1 = GATConv(input_dim, 16, heads=2)
        self.gat2 = GATConv(16 * 2, output_dim, heads=1)
    def forward(self, x, edge_index):
        return self.gat2(F.elu(self.gat1(x, edge_index)), edge_index)

class GatingNetwork(nn.Module):
    """An MLP that computes expert weights based on a context vector."""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, num_experts))
    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

class GVF_MoE_Final_Model(nn.Module):
    """
    The final, complete GVF-MoE model. It features a shared context encoder
    to inform all specialized experts.
    """
    def __init__(self, full_input_dim, final_output_dim, embedding_dim=8):
        super().__init__()
        self.num_experts = 3

        # Shared encoder to create a context vector
        self.shared_encoder = nn.Sequential(nn.Linear(full_input_dim, 16), nn.ReLU(), nn.Linear(16, embedding_dim))

        # Experts receive their specific features plus the shared context
        self.env_expert = StaticExpert(input_dim=3 + embedding_dim, output_dim=final_output_dim)
        self.phys_expert = EvolveGCN_O_Expert(in_channels=2 + embedding_dim, out_channels=final_output_dim)
        self.behav_expert = StaticExpert(input_dim=3 + embedding_dim, output_dim=final_output_dim)

        # The Gating Network uses the context to decide on weights
        self.gating = GatingNetwork(embedding_dim, self.num_experts)
        self.W_t = None
        self.last_gating_weights = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1. Create the shared context vector
        context_vector = self.shared_encoder(x)

        # 2. Split raw data and concatenate with context for each expert
        x_env_raw = x[:, [0, 1, 5]]
        x_phys_raw = x[:, [3, 4]]
        x_behav_raw = x[:, [0, 1, 2]]

        x_env_contextual = torch.cat([x_env_raw, context_vector], dim=1)
        x_phys_contextual = torch.cat([x_phys_raw, context_vector], dim=1)
        x_behav_contextual = torch.cat([x_behav_raw, context_vector], dim=1)

        # 3. Compute outputs from context-aware experts
        env_out = self.env_expert(x_env_contextual, edge_index)
        phys_out, new_W_t = self.phys_expert(x_phys_contextual, edge_index, self.W_t)
        self.W_t = new_W_t.detach()
        behav_out = self.behav_expert(x_behav_contextual, edge_index)

        # 4. The Gating Network weighs the final outputs using the context
        weights = self.gating(context_vector)
        self.last_gating_weights = weights

        final_risk_vector = (weights[:, 0].unsqueeze(1) * env_out +
                             weights[:, 1].unsqueeze(1) * phys_out +
                             weights[:, 2].unsqueeze(1) * behav_out)

        return final_risk_vector

    def reset_hidden_state(self):
        self.W_t = None

# --- 3. Training and Evaluation Functions ---
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    if isinstance(model, GVF_MoE_Final_Model):
        model.reset_hidden_state()

    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_and_get_weights(model, loader):
    model.eval()
    all_gating_weights = []
    if isinstance(model, GVF_MoE_Final_Model):
        model.reset_hidden_state()

    for data in loader:
        _ = model(data)
        weights = model.last_gating_weights
        all_gating_weights.append(weights.cpu().numpy())

    return np.concatenate(all_gating_weights, axis=0)

# --- 4. Analysis and Plotting Function ---
def analyze_and_plot_gating_weights(weights, num_agents, lockdown_start, lockdown_end):
    print("\n--- Analyzing and Plotting Gating Dynamics ---")

    num_timesteps = len(weights) // num_agents
    if num_timesteps == 0:
        print("Warning: No timesteps to analyze.")
        return

    weights_over_time = weights.reshape(num_timesteps, num_agents, -1)
    avg_weights_over_time = np.mean(weights_over_time, axis=1)

    # Robustly handle the time phases
    pre_lockdown_slice = avg_weights_over_time[:lockdown_start]
    lockdown_slice = avg_weights_over_time[lockdown_start:lockdown_end]
    post_lockdown_slice = avg_weights_over_time[lockdown_end:]

    pre_lockdown_weights = np.mean(pre_lockdown_slice, axis=0) if pre_lockdown_slice.size > 0 else np.zeros(3)
    lockdown_weights = np.mean(lockdown_slice, axis=0) if lockdown_slice.size > 0 else np.zeros(3)
    post_lockdown_weights = np.mean(post_lockdown_slice, axis=0) if post_lockdown_slice.size > 0 else np.zeros(3)

    labels = ['EnvExpert', 'PhysExpert', 'BehavExpert']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, pre_lockdown_weights, width, label='Pre-Lockdown', color='skyblue')
    rects2 = ax.bar(x, lockdown_weights, width, label='Lockdown', color='salmon')
    rects3 = ax.bar(x + width, post_lockdown_weights, width, label='Post-Lockdown', color='lightgreen')

    ax.set_ylabel('Average Weight')
    ax.set_title('Gating Network Expert Weights During Simulation Phases')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig('gating_weights_analysis.png')
    print("Plot saved to gating_weights_analysis.png")
    plt.show()


# --- 5. Main Execution Block ---
if __name__ == '__main__':
    # --- Hyperparameters ---
    LEARNING_RATE = 0.001
    EPOCHS = 120
    NUM_AGENTS = 80

    # --- Data Generation ---
    dataset, feature_names, lockdown_start, lockdown_end = generate_ultimate_stress_dataset(num_agents=NUM_AGENTS)

    # --- Preprocessing ---
    print("--- Preprocessing: Normalizing Features ---")
    all_features = np.concatenate([d.x.numpy() for d in dataset], axis=0)
    scaler = StandardScaler().fit(all_features)
    for d in dataset:
        d.x = torch.tensor(scaler.transform(d.x.numpy()), dtype=torch.float32)
    print("--- Preprocessing Complete ---\n")

    # --- Data Splitting and Loading ---
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]

    # Loader for training
    train_loader = DataLoader(train_dataset, batch_size=720, shuffle=False)
    # Loader for the final analysis on the entire dataset
    full_loader = DataLoader(dataset, batch_size=720, shuffle=False)

    # --- Model Initialization ---
    INPUT_DIM, OUTPUT_DIM = 7, 3
    model = GVF_MoE_Final_Model(INPUT_DIM, OUTPUT_DIM)

    # --- Training ---
    print(f"\n--- Training {model.__class__.__name__} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        if epoch % 20 == 0:
            print(f'Epoch {epoch:02d}, Loss: {loss:.6f}')

    # --- Evaluation and Visualization ---
    # Extract weights from the entire dataset after training
    gating_weights = evaluate_and_get_weights(model, full_loader)

    # Analyze and plot the weights using the original temporal indices
    analyze_and_plot_gating_weights(gating_weights, NUM_AGENTS, lockdown_start, lockdown_end)
