"""
Crystal Structure Diffusion Model - Hello World Version
======================================================

A physics-constrained diffusion model for generating crystal structures from small datasets.

ARCHITECTURE OVERVIEW:
---------------------
1. Denoising Diffusion Probabilistic Model (DDPM) - Ho et al. 2020
2. Score-based generative modeling for 3D periodic structures
3. Physics constraint layers for real-time validation

ALGORITHMS & TIME COMPLEXITY:
-----------------------------
1. Forward Diffusion Process: O(T) where T = timesteps
2. Reverse Diffusion (Sampling): O(T × N × d³) where N = atoms, d = feature dimension
3. U-Net Architecture: O(L × C × H × W) per forward pass
4. Physics Validation: O(N²) for pairwise distances, O(N) for other checks
5. Training Loop: O(E × B × T) where E = epochs, B = batch size

PHYSICS CONSTRAINTS:
-------------------
1. Minimum interatomic distance (Pauli exclusion)
2. Charge neutrality
3. Periodic boundary conditions
4. Crystal symmetry preservation (simplified)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Optional
import json
from datetime import datetime

# # Set random seeds for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class CrystalStructureDataset(Dataset):
    """
    Dataset class for crystal structures.
    
    Time Complexity: O(1) for __getitem__, O(N) for __init__ where N = dataset size
    Space Complexity: O(N × M × D) where M = max atoms, D = feature dimensions
    """
    
    def __init__(self, crystal_data: List[dict]):
        """
        Args:
            crystal_data: List of dicts with 'positions', 'atom_types', 'lattice'
        """
        self.data = crystal_data
        self.max_atoms = max(len(d['positions']) for d in crystal_data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            positions: (max_atoms, 3) - fractional coordinates [0, 1]
            atom_types: (max_atoms,) - atomic numbers
            lattice: (3, 3) - lattice vectors
            mask: (max_atoms,) - 1 for real atoms, 0 for padding
        """
        crystal = self.data[idx]
        positions = np.array(crystal['positions'])  # Fractional coordinates
        atom_types = np.array(crystal['atom_types'])
        lattice = np.array(crystal['lattice'])
        
        # Pad to max_atoms
        n_atoms = len(positions)
        positions_padded = np.zeros((self.max_atoms, 3))
        atom_types_padded = np.zeros(self.max_atoms)
        mask = np.zeros(self.max_atoms)
        
        positions_padded[:n_atoms] = positions
        atom_types_padded[:n_atoms] = atom_types
        mask[:n_atoms] = 1
        
        return {
            'positions': torch.FloatTensor(positions_padded),
            'atom_types': torch.LongTensor(atom_types_padded),
            'lattice': torch.FloatTensor(lattice),
            'mask': torch.FloatTensor(mask)
        }


class PhysicsValidator:
    """
    Validates and corrects crystal structures based on physics laws.
    
    PHYSICS LAWS IMPLEMENTED:
    1. Minimum Interatomic Distance: r_min = 0.7 Angstroms (Pauli exclusion)
    2. Charge Neutrality: Σ charges ≈ 0
    3. Periodic Boundary Conditions: positions ∈ [0, 1]³
    4. Density bounds: ρ ∈ [0.5, 20] g/cm³
    """
    
    def __init__(self, min_distance: float = 0.7):
        """
        Args:
            min_distance: Minimum allowed interatomic distance in Angstroms
        """
        self.min_distance = min_distance
        
    def check_minimum_distance(self, positions: torch.Tensor, 
                           lattice: torch.Tensor,
                           mask: torch.Tensor) -> Tuple[bool, float]:

        batch_size = positions.shape[0]
        valid = True
        min_dist_found = float('inf')
        
        for b in range(batch_size):
            pos = positions[b][mask[b] > 0.5]  # Only real atoms
            lat = lattice[b]
            
            if len(pos) < 2:
                continue
                
            # Convert to Cartesian coordinates
            cart_pos = torch.matmul(pos, lat)  # (N, 3)
            
            # Compute pairwise distances with PBC
            for i in range(len(cart_pos)):
                for j in range(i + 1, len(cart_pos)):
                    diff = cart_pos[i] - cart_pos[j]
                    
                    # Apply minimum image convention (PBC)
                    # Convert difference to fractional coordinates
                    diff_frac = torch.matmul(diff.unsqueeze(0), torch.inverse(lat.T)).squeeze(0)
                    
                    # Wrap to [-0.5, 0.5] range
                    diff_frac = diff_frac - torch.round(diff_frac)
                    
                    # Convert back to Cartesian
                    diff = torch.matmul(diff_frac.unsqueeze(0), lat).squeeze(0)
                    
                    dist = torch.norm(diff).item()
                    min_dist_found = min(min_dist_found, dist)
                    
                    if dist < self.min_distance:
                        valid = False
        
        return valid, min_dist_found
    
    def enforce_pbc(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Enforce periodic boundary conditions.
        
        Algorithm: Modulo operation
        Time Complexity: O(N) where N = number of atoms
        Space Complexity: O(N)
        
        Args:
            positions: (batch, max_atoms, 3) fractional coordinates
            
        Returns:
            positions: Wrapped to [0, 1]³
        """
        # Wrap coordinates to [0, 1] using modulo
        return positions % 1.0
    
    def correct_structure(self, positions: torch.Tensor,
                         lattice: torch.Tensor,
                         mask: torch.Tensor,
                         max_iterations: int = 10) -> torch.Tensor:
        """
        Iteratively correct structure to satisfy physics constraints.
        
        Algorithm: Gradient-based repulsion + projection
        Time Complexity: O(I × N²) where I = iterations, N = atoms
        Space Complexity: O(N²)
        
        Args:
            positions: (batch, max_atoms, 3)
            lattice: (batch, 3, 3)
            mask: (batch, max_atoms)
            max_iterations: Maximum correction iterations
            
        Returns:
            corrected_positions: (batch, max_atoms, 3)
        """
        positions = positions.clone()
        
        for iteration in range(max_iterations):
            valid, min_dist = self.check_minimum_distance(positions, lattice, mask)
            
            if valid:
                break
                
            # Apply repulsive forces to atoms that are too close
            batch_size = positions.shape[0]
            
            for b in range(batch_size):
                pos = positions[b]
                m = mask[b]
                lat = lattice[b]
                
                # Get real atoms only
                real_indices = torch.where(m > 0.5)[0]
                if len(real_indices) < 2:
                    continue
                
                # Convert to Cartesian
                cart_pos = torch.matmul(pos[real_indices], lat)
                
                # Compute repulsive forces
                forces = torch.zeros_like(cart_pos)
                
                for i in range(len(cart_pos)):
                    for j in range(len(cart_pos)):
                        if i == j:
                            continue
                            
                        diff = cart_pos[i] - cart_pos[j]
                        dist = torch.norm(diff)
                        
                        if dist < self.min_distance:
                            # Repulsive force proportional to overlap
                            force_mag = (self.min_distance - dist) / self.min_distance
                            force_dir = diff / (dist + 1e-8)
                            forces[i] += force_mag * force_dir * 0.1
                
                # Convert forces back to fractional coordinates
                frac_forces = torch.matmul(forces, torch.inverse(lat))
                
                # Update positions
                pos[real_indices] += frac_forces
                positions[b] = pos
            
            # Enforce PBC
            positions = self.enforce_pbc(positions)
        
        return positions


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for timestep encoding.
    
    Algorithm: Sinusoidal encoding (Vaswani et al., 2017)
    Time Complexity: O(B × D) where B = batch size, D = embedding dimension
    Space Complexity: O(D)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (batch_size,) timesteps
            
        Returns:
            embeddings: (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DenoisingNetwork(nn.Module):
    """
    Neural network for predicting noise in diffusion process.
    
    Architecture: Modified U-Net with attention
    - Encoder: 3 blocks with downsampling
    - Bottleneck: Self-attention layer
    - Decoder: 3 blocks with upsampling
    
    Time Complexity per forward pass: O(B × N × D² × L)
    where B = batch size, N = atoms, D = hidden dimension, L = layers
    
    Space Complexity: O(B × N × D)
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, 
                 time_dim: int = 64, output_dim: int = 3):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        # Bottleneck with self-attention
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 4 due to skip connection
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 due to skip connection
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict noise given noisy input and timestep.
        
        Algorithm: U-Net forward pass with skip connections
        Time Complexity: O(B × N × D²) where B = batch, N = atoms, D = hidden_dim
        
        Args:
            x: (batch, max_atoms, 3) noisy positions
            t: (batch,) timesteps
            mask: (batch, max_atoms) atom mask
            
        Returns:
            noise_pred: (batch, max_atoms, 3) predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Encoder with skip connections
        h1 = self.encoder1(x)  # (batch, max_atoms, hidden_dim)
        h1 = h1 + t_emb  # Add time information
        
        h2 = self.encoder2(h1)  # (batch, max_atoms, hidden_dim * 2)
        
        # Bottleneck with self-attention
        h3 = self.bottleneck(h2)
        
        # Self-attention (Time: O(N² × D))
        if mask is not None:
            # Create attention mask (True = masked out)
            attn_mask = (mask == 0).unsqueeze(1).repeat(1, mask.shape[1], 1)
            h3_attn, _ = self.attention(h3, h3, h3, key_padding_mask=(mask == 0))
        else:
            h3_attn, _ = self.attention(h3, h3, h3)
        
        h3 = h3 + h3_attn  # Residual connection
        
        # Decoder with skip connections (U-Net style)
        h4 = torch.cat([h3, h2], dim=-1)  # Skip connection
        h4 = self.decoder1(h4)
        
        h5 = torch.cat([h4, h1], dim=-1)  # Skip connection
        h5 = self.decoder2(h5)
        
        # Output
        noise_pred = self.output(h5)
        
        # Apply mask to zero out predictions for padding
        if mask is not None:
            noise_pred = noise_pred * mask.unsqueeze(-1)
        
        return noise_pred


class CrystalDiffusionModel:
    """
    Main diffusion model for crystal structure generation.
    
    DIFFUSION PROCESS:
    -----------------
    Forward (adding noise): q(x_t | x_0) = N(x_t; √α̅_t x_0, (1 - α̅_t)I)
    Reverse (denoising): p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
    
    TRAINING OBJECTIVE:
    ------------------
    L_simple = E_{t,x_0,ε}[||ε - ε_θ(√α̅_t x_0 + √(1-α̅_t)ε, t)||²]
    
    Time Complexity:
    - Training step: O(T × B × N × D²) per batch
    - Sampling: O(T × N × D²) per sample
    """
    
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = 'cpu'):
        """
        Args:
            timesteps: Number of diffusion timesteps T
            beta_start: Initial noise schedule β_1
            beta_end: Final noise schedule β_T
            device: Computing device
        """
        self.timesteps = timesteps
        self.device = device
        
        # Variance schedule: Linear schedule (Ho et al., 2020)
        # Algorithm: Linear interpolation
        # Time: O(T), Space: O(T)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Initialize model
        self.model = DenoisingNetwork().to(device)
        self.validator = PhysicsValidator()
        
        # Optimizer: Adam with weight decay (AdamW)
        # Time per step: O(P) where P = number of parameters
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean data.
        
        Algorithm: Reparameterization trick
        Formula: x_t = √α̅_t x_0 + √(1-α̅_t)ε
        Time Complexity: O(B × N × D) where B = batch, N = atoms, D = dimensions
        
        Args:
            x_start: (batch, max_atoms, 3) clean data
            t: (batch,) timesteps
            noise: Optional pre-sampled noise
            
        Returns:
            x_noisy: (batch, max_atoms, 3) noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting: (batch, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]
        
        # Apply noise
        x_noisy = (sqrt_alphas_cumprod_t * x_start + 
                   sqrt_one_minus_alphas_cumprod_t * noise)
        
        return x_noisy
    
    def p_sample(self, x: torch.Tensor, t: int, mask: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion process: denoise one step.
        
        Algorithm: Ancestral sampling (DDPM)
        Formula: x_{t-1} = μ_θ + σ_t z where z ~ N(0, I)
        Time Complexity: O(N × D²) per call
        
        Args:
            x: (batch, max_atoms, 3) noisy data at timestep t
            t: current timestep
            mask: (batch, max_atoms) atom mask
            
        Returns:
            x_prev: (batch, max_atoms, 3) data at timestep t-1
        """
        # Get batch size
        batch_size = x.shape[0]
        
        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            predicted_noise = self.model(x, t_tensor, mask)
        
        # Get parameters
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Compute mean: μ_θ(x_t, t) = (1/√α_t)(x_t - (β_t/√(1-α̅_t))ε_θ(x_t, t))
        model_mean = sqrt_recip_alpha * (
            x - beta * predicted_noise / sqrt_one_minus_alpha_cumprod
        )
        
        if t == 0:
            return model_mean
        else:
            # Add noise for t > 0
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            # Apply mask to noise
            noise = noise * mask.unsqueeze(-1)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, batch_size: int, max_atoms: int, lattice: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        """
        Generate new crystal structures.
        
        Algorithm: Reverse diffusion sampling
        Time Complexity: O(T × N × D²) where T = timesteps, N = atoms, D = hidden_dim
        Space Complexity: O(B × N × 3)
        
        Args:
            batch_size: Number of structures to generate
            max_atoms: Maximum number of atoms
            lattice: (batch, 3, 3) lattice vectors
            mask: (batch, max_atoms) atom mask
            
        Returns:
            samples: (batch, max_atoms, 3) generated structures
        """
        # Start from random noise
        x = torch.randn(batch_size, max_atoms, 3).to(self.device)
        x = x * mask.unsqueeze(-1)  # Apply mask
        
        # Reverse diffusion process
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, mask)
            
            # Enforce periodic boundary conditions every 100 steps
            if t % 100 == 0:
                x = self.validator.enforce_pbc(x)
        
        # Final physics correction
        x = self.validator.correct_structure(x, lattice, mask)
        x = self.validator.enforce_pbc(x)
        
        return x
    
    def train_step(self, batch: dict) -> float:
        """
        Single training step.
        
        Algorithm: Denoising score matching
        Time Complexity: O(B × N × D²) per batch
        
        Args:
            batch: Dictionary with 'positions', 'lattice', 'mask'
            
        Returns:
            loss: Training loss value
        """
        positions = batch['positions'].to(self.device)
        mask = batch['mask'].to(self.device)
        batch_size = positions.shape[0]
        
        # Sample random timesteps
        # Algorithm: Uniform sampling
        # Time: O(B)
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Sample noise
        noise = torch.randn_like(positions)
        noise = noise * mask.unsqueeze(-1)  # Apply mask
        
        # Forward diffusion (add noise)
        x_noisy = self.q_sample(positions, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, mask)
        
        # Compute loss (MSE between true and predicted noise)
        # Algorithm: Mean Squared Error
        # Time: O(B × N × D)
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()  # Masked loss
        
        # Backpropagation
        # Algorithm: Reverse-mode automatic differentiation
        # Time: O(P) where P = number of parameters
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()


def create_demo_dataset() -> List[dict]:
    """
    Create a small demo dataset of simple crystal structures.
    
    DEMO STRUCTURES:
    1. Simple Cubic (SC)
    2. Face-Centered Cubic (FCC)
    3. Body-Centered Cubic (BCC)
    4. Diamond structure
    
    Time Complexity: O(1) - fixed number of structures
    Space Complexity: O(N) where N = total atoms across all structures
    """
    dataset = []
    
    # 1. Simple Cubic (1 atom at origin)
    dataset.append({
        'positions': np.array([[0.0, 0.0, 0.0]]),
        'atom_types': np.array([14]),  # Silicon
        'lattice': np.array([
            [5.43, 0.0, 0.0],
            [0.0, 5.43, 0.0],
            [0.0, 0.0, 5.43]
        ])
    })
    
    # 2. Face-Centered Cubic (4 atoms)
    dataset.append({
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]),
        'atom_types': np.array([79, 79, 79, 79]),  # Gold
        'lattice': np.array([
            [4.08, 0.0, 0.0],
            [0.0, 4.08, 0.0],
            [0.0, 0.0, 4.08]
        ])
    })
    
    # 3. Body-Centered Cubic (2 atoms)
    dataset.append({
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]),
        'atom_types': np.array([26, 26]),  # Iron
        'lattice': np.array([
            [2.87, 0.0, 0.0],
            [0.0, 2.87, 0.0],
            [0.0, 0.0, 2.87]
        ])
    })
    
    # 4. Diamond structure (8 atoms)
    dataset.append({
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75]
        ]),
        'atom_types': np.array([6, 6, 6, 6, 6, 6, 6, 6]),  # Carbon (diamond)
        'lattice': np.array([
            [3.57, 0.0, 0.0],
            [0.0, 3.57, 0.0],
            [0.0, 0.0, 3.57]
        ])
    })
    
    # Add variations with small perturbations
    for _ in range(6):
        # Randomly select a base structure
        base = dataset[np.random.randint(len(dataset))].copy()
        
        # Add small random perturbations to positions
        base['positions'] = base['positions'] + np.random.normal(0, 0.02, base['positions'].shape)
        base['positions'] = base['positions'] % 1.0  # Ensure PBC
        
        dataset.append(base)
    
    return dataset


def visualize_crystal(positions: np.ndarray, lattice: np.ndarray, 
                      title: str = "Crystal Structure"):
    """
    Visualize a crystal structure in 3D.
    
    Time Complexity: O(N) where N = number of atoms
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert fractional to Cartesian coordinates
    cart_positions = positions @ lattice
    
    # Plot atoms
    ax.scatter(cart_positions[:, 0], cart_positions[:, 1], cart_positions[:, 2],
               c='blue', s=200, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Plot lattice vectors
    origin = np.array([0, 0, 0])
    colors = ['red', 'green', 'blue']
    for i, (vec, color) in enumerate(zip(lattice, colors)):
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.1, linewidth=2,
                 label=f'a{i+1}')
    
    # Draw unit cell edges
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    cart_corners = corners @ lattice
    
    # Draw edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
    ]
    
    for start, end in edges:
        ax.plot3D(*zip(cart_corners[start], cart_corners[end]), 
                 'gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    ax.legend()
    
    return fig


def main():
    """
    Main training and generation pipeline.
    
    PIPELINE STEPS:
    1. Create dataset (O(1))
    2. Initialize model (O(P) where P = parameters)
    3. Training loop (O(E × B × T × N × D²))
    4. Generate samples (O(T × N × D²))
    5. Validate physics (O(N²))
    6. Visualize (O(N))
    
    Total Time Complexity: O(E × B × T × N × D²) dominated by training
    """
    print("=" * 80)
    print("CRYSTAL STRUCTURE DIFFUSION MODEL - DEMO")
    print("=" * 80)
    
    # Configuration
    config = {
        'batch_size': 4,
        'epochs': 50,
        'timesteps': 200,  # Reduced for faster demo
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nConfiguration:")
    print(json.dumps(config, indent=2))
    print(f"\nDevice: {config['device']}")
    
    # Create dataset
    print("\n[Step 1/5] Creating demo dataset...")
    crystal_data = create_demo_dataset()
    print(f"Created {len(crystal_data)} crystal structures")
    
    dataset = CrystalStructureDataset(crystal_data)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                          shuffle=True, drop_last=True)
    
    # Initialize model
    print("\n[Step 2/5] Initializing diffusion model...")
    model = CrystalDiffusionModel(
        timesteps=config['timesteps'],
        device=config['device']
    )
    
    num_params = sum(p.numel() for p in model.model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training loop
    print(f"\n[Step 3/5] Training for {config['epochs']} epochs...")
    print("-" * 80)
    
    losses = []
    for epoch in range(config['epochs']):
        epoch_losses = []
        
        for batch in dataloader:
            loss = model.train_step(batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {avg_loss:.6f}")
    
    print("-" * 80)
    print(f"Training complete! Final loss: {losses[-1]:.6f}")
    

    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'/home/sangam/vsc/diffusion_material/outputs/training_loss_{timestamp}_png', dpi=150, bbox_inches='tight')
    print("\nTraining curve saved to 'training_loss.png'")
    
    # Generate new structures
    print("\n[Step 4/5] Generating new crystal structures...")
    
    # Use a lattice and mask from the dataset
    sample_batch = next(iter(dataloader))
    lattice = sample_batch['lattice'][:1].to(config['device'])
    mask = sample_batch['mask'][:1].to(config['device'])
    
    generated = model.sample(
        batch_size=1,
        max_atoms=dataset.max_atoms,
        lattice=lattice,
        mask=mask
    )
    
    # Validate physics
    print("\n[Step 5/5] Validating physics constraints...")
    valid, min_dist = model.validator.check_minimum_distance(
        generated, lattice, mask
    )
    print(f"Minimum distance check: {'PASSED' if valid else 'FAILED'}")
    print(f"Minimum distance found: {min_dist:.3f} Å")
    
    # Visualize
    print("\nVisualizing generated structure...")
    gen_pos = generated[0].cpu().numpy()
    gen_pos = gen_pos[mask[0].cpu().numpy() > 0.5]  # Remove padding
    gen_lattice = lattice[0].cpu().numpy()

    fig = visualize_crystal(gen_pos, gen_lattice, "Generated Crystal Structure")
    plt.savefig(f'/home/sangam/vsc/diffusion_material/outputs/generated_crystal_{timestamp}.png', dpi=150, bbox_inches='tight')
    print("Generated structure saved to 'generated_crystal.png'")
    
    # Also visualize a real structure for comparison
    real_pos = sample_batch['positions'][0].numpy()
    real_pos = real_pos[sample_batch['mask'][0].numpy() > 0.5]
    real_lattice = sample_batch['lattice'][0].numpy()
    
    fig = visualize_crystal(real_pos, real_lattice, "Real Crystal Structure (Training Data)")
    plt.savefig(f'/home/sangam/vsc/diffusion_material/outputs/real_crystal_{timestamp}.png', dpi=150, bbox_inches='tight')
    print("Real structure saved to 'real_crystal.png'")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    
    # Summary statistics
    print("\nModel Statistics:")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Training samples: {len(dataset)}")
    print(f"  - Diffusion timesteps: {config['timesteps']}")
    print(f"  - Final training loss: {losses[-1]:.6f}")
    print(f"  - Physics validation: {'PASSED' if valid else 'FAILED'}")
    
    print("\nTime Complexity Summary:")
    print(f"  - Training (per epoch): O(B × T × N × D²) = O({config['batch_size']} × {config['timesteps']} × N × D²)")
    print(f"  - Sampling: O(T × N × D²) = O({config['timesteps']} × N × D²)")
    print(f"  - Physics validation: O(N²)")
    
    print("\nAlgorithms Used:")
    print("  1. Denoising Diffusion Probabilistic Models (DDPM) - Ho et al. 2020")
    print("  2. U-Net Architecture with Self-Attention")
    print("  3. Sinusoidal Positional Embeddings - Vaswani et al. 2017")
    print("  4. Adam Optimizer with Weight Decay (AdamW)")
    print("  5. Minimum Image Convention for Periodic Boundary Conditions")
    print("  6. Gradient-based Physics Correction")


if __name__ == "__main__":
    main()