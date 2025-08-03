# graph-vector-fields
Official implementation of "Graph Vector Fields: A New Framework for Personalised Risk Assessment"
This repository contains the official source code and experiments for the paper "Graph Vector Fields: A New Framework for Personalised Risk Assessment". We introduce **Graph Vector Fields (GVFs)**, a novel framework that moves beyond traditional scalar scores to represent health risk as a dynamic, interpretable vector field on a graph.

The code includes:

1) A synthetic data generator that creates a complex, dynamic environment with multiple interacting risk factors, including a social graph with contagious risk propagation.

2) The full PyTorch implementation of our specialized Mixture-of-Experts (MoE) model, featuring a Graph Neural Network (GNN) expert and a load balancing loss for stable training.

3) Scripts to reproduce the empirical validation, including training and evaluation against the baseline model.

This work provides a robust proof-of-concept for the GVF framework, demonstrating its superior ability to model complex, heterogeneous, and relational health data compared to standard approaches.
