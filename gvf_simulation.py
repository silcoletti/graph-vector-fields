import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import functional as F

# --- 1. Advanced Synthetic Dataset with Dynamic Graphs (V3) ---
def generate_graph_dataset(num_agents=50, num_days=90):
    """
    Generates a complex synthetic dataset simulating agents in a dynamic environment.
    Includes a dynamic social graph and multiple orthogonal risk factors.
    """
    print("--- Generating V3 Dataset with Dynamic Graph ---")
    timesteps_per_day = 4
    num_timesteps = num_days * timesteps_per_day
    grid_size, proximity_radius = 20.0, 2.5
    
    # Agent State Vector: [x_coord, y_coord, speed, skin_temp, medication_flag]
    agent_states = np.zeros((num_agents, num_timesteps, 5))
    # Ground-Truth Risk Vector: [env_phys_risk_x, env_phys_risk_y, contagious_risk]
    ground_truth_risk = np.zeros((num_agents, num_timesteps, 3))
    
    agent_positions = np.random.rand(num_agents, 2) * grid_size
    contagious_status = np.zeros(num_agents)
    contagious_status[np.random.choice(num_agents, 2, replace=False)] = 1  # Patient zero
    
    dataset = []

    for t in range(num_timesteps):
        # Build the dynamic social graph for the current timestep
        positions_tensor = torch.tensor(agent_positions, dtype=torch.float32)
        dist_matrix = torch.cdist(positions_tensor, positions_tensor)
        edge_index = (dist_matrix < proximity_radius).nonzero(as_tuple=False).t().contiguous()
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remove self-loops

        # Propagate the contagious risk through the graph
        infected_nodes = np.where(contagious_status == 1)[0]
        for u, v in edge_index.t().numpy():
            if u in infected_nodes and np.random.rand() < 0.1:  # Contagion probability
                contagious_status[v] = 1
                
        # Update state and risk for each agent
        for i in range(num_agents):
            move = np.random.randn(2)
            agent_positions[i] = np.clip(agent_positions[i] + move, 0, grid_size - 1)
            speed, skin_temp = np.linalg.norm(move), 37.0 + np.random.randn() * 0.1
            med_flag = np.random.choice([0, 1], p=[0.9, 0.1])
            agent_states[i, t] = [agent_positions[i,0], agent_positions[i,1], speed, skin_temp, med_flag]
            
            # Calculate orthogonal risks
            env_risk_vec = np.array([10,10]) - agent_positions[i]
            env_risk = 0.5/(1+np.linalg.norm(env_risk_vec)) * env_risk_vec
            phys_risk = np.array([skin_temp - 37, -speed]) * 0.2
            cont_risk = contagious_status[i] * (1 - (med_flag * 0.5)) # Medication reduces risk
            ground_truth_risk[i, t] = [env_risk[0]+phys_risk[0], env_risk[1]+phys_risk[1], cont_risk]
            
        # Save the graph snapshot for this timestep
        dataset.append(Data(x=torch.tensor(agent_states[:,t,:], dtype=torch.float32),
                          edge_index=edge_index,
                          y=torch.tensor(ground_truth_risk[:,t,:], dtype=torch.float32)))
    print("--- Dataset Generation Complete ---\n")
    return dataset

# --- 2. GVF Model V4 with Data Specificity ---
class GraphBehavioralExpert(torch.nn.Module):
    """A GNN expert for processing spatio-behavioral data and graph structure."""
    def __init__(self, in_channels, out_channels):
        super(GraphBehavioralExpert, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
    def forward(self, x_graph, edge_index):
        x = F.relu(self.conv1(x_graph, edge_index))
        return self.conv2(x, edge_index)

class PhysioContextExpert(nn.Module):
    """An MLP expert for processing physiological and contextual data."""
    def __init__(self, input_dim, output_dim):
        super(PhysioContextExpert, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, output_dim))
    def forward(self, x_context):
        return self.net(x_context)

class GVF_MoE_V4_Model(nn.Module):
    """The final GVF model with specialized experts and a gating network."""
    def __init__(self, full_input_dim, graph_input_dim, context_input_dim, output_dim):
        super(GVF_MoE_V4_Model, self).__init__()
        self.graph_expert = GraphBehavioralExpert(graph_input_dim, output_dim)
        self.context_expert = PhysioContextExpert(context_input_dim, output_dim)
        self.gating = nn.Sequential(nn.Linear(full_input_dim, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # --- Data Specificity Implementation ---
        # Route specific data slices to corresponding experts
        x_graph_behavioral = x[:, :3]  # x_coord, y_coord, speed
        x_physio_context = x[:, 3:]    # skin_temp, medication_flag
        
        graph_out = self.graph_expert(x_graph_behavioral, edge_index)
        context_out = self.context_expert(x_physio_context)
        
        # Combine outputs using weights from the gating network (which sees the full state)
        expert_outputs = torch.stack([graph_out, context_out], dim=1)
        weights = F.softmax(self.gating(x), dim=1).unsqueeze(2)
        
        return (expert_outputs * weights).sum(dim=1)

class BaselineMLP(nn.Module):
    """A standard MLP baseline that is graph-agnostic."""
    def __init__(self, input_dim, output_dim):
        super(BaselineMLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))
    def forward(self, data):
        return self.layers(data.x)

# --- 3. Training & Evaluation Functions ---
def train(model, loader, balancing_coefficient=0.01):
    """Training loop including the load balancing loss for the MoE model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        main_loss = criterion(out, data.y)
        
        balancing_loss = 0
        if isinstance(model, GVF_MoE_V4_Model):
            # Calculate load balancing loss to prevent gating collapse
            gating_weights = F.softmax(model.gating(data.x), dim=1)
            avg_expert_importance = gating_weights.mean(dim=0)
            balancing_loss = torch.std(avg_expert_importance)**2 * model.gating[-1].out_features
        
        # Combine the main loss with the balancing loss
        loss = main_loss + (balancing_coefficient * balancing_loss)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test_and_inspect_gating(model, loader):
    """Evaluation loop that also inspects the gating weights for the MoE model."""
    model.eval()
    print("\n--- Inspecting Gating Network Weights ---")
    try:
        data_batch = next(iter(loader))
        if isinstance(model, GVF_MoE_V4_Model):
            weights = F.softmax(model.gating(data_batch.x), dim=1)
            print("Average expert weights (Graph, Context):", weights.mean(dim=0).numpy())
        else:
            print("Model is an MLP, no gating inspection possible.")
    except StopIteration:
        print("DataLoader is empty, cannot inspect.")
    
    all_preds, all_true, all_preds_cont, all_true_cont = [], [], [], []
    for data in loader:
        preds = model(data)
        all_preds.append(preds.numpy())
        all_true.append(data.y.numpy())
        all_preds_cont.append(preds[:, 2].numpy())
        all_true_cont.append(data.y[:, 2].numpy())
        
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_preds_cont = np.concatenate(all_preds_cont, axis=0)
    all_true_cont = np.concatenate(all_true_cont, axis=0)
    
    mse = np.mean((all_true - all_preds)**2)
    contagion_mse = np.mean((all_true_cont - all_preds_cont)**2)
    return {"Total MSE": mse, "Contagion Risk MSE": contagion_mse}

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # Generate the dataset
    dataset = generate_graph_dataset(num_agents=30, num_days=60)
    # Note: Use torch_geometric's DataLoader for graph data
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define model dimensions
    FULL_INPUT_DIM, GRAPH_INPUT_DIM, CONTEXT_INPUT_DIM, OUTPUT_DIM = 5, 3, 2, 3
    
    # Train and evaluate the GVF model
    gvf_model_v4 = GVF_MoE_V4_Model(FULL_INPUT_DIM, GRAPH_INPUT_DIM, CONTEXT_INPUT_DIM, OUTPUT_DIM)
    print("Training GVF Model V4 (with Load Balancing)...")
    for epoch in range(1, 41): # Increased epochs for the balancing loss to take effect
        loss = train(gvf_model_v4, loader)
        if epoch % 5 == 0: print(f'GVF Epoch {epoch:02d}, Loss: {loss:.6f}')
    
    # Train and evaluate the baseline model
    baseline_model = BaselineMLP(FULL_INPUT_DIM, OUTPUT_DIM)
    print("\nTraining Baseline MLP...")
    for epoch in range(1, 21):
        loss = train(baseline_model, loader)
        if epoch % 5 == 0: print(f'Baseline Epoch {epoch:02d}, Loss: {loss:.6f}')

    # Final Evaluation and Inspection
    print("\n--- Evaluating GVF Model ---")
    gvf_results = test_and_inspect_gating(gvf_model_v4, loader)
    
    print("\n--- Evaluating Baseline MLP ---")
    baseline_results = test_and_inspect_gating(baseline_model, loader)
    
    # Print the final comparison table
    print("\n--- Final Evaluation: Specialization vs. Generalization ---")
    print(f"{'Metric':<25} | {'GVF MoE V4 (Specialized)':<25} | {'Baseline MLP':<20}")
    print("-" * 78)
    print(f"{'Total MSE':<25} | {gvf_results['Total MSE']:<25.6f} | {baseline_results['Total MSE']:<20.6f}")
    print(f"{'Contagion Risk MSE':<25} | {gvf_results['Contagion Risk MSE']:<25.6f} | {baseline_results['Contagion Risk MSE']:<20.6f}")