# Crystal Diffusion Model

---
This notebook thanks about an **Diffusion model that generates brand-new crystal structures** (like atoms arranged in a solid material). It does this using a technique called a **Diffusion Model**, the same core idea behind AI image generators like Stable Diffusion, but applied to materials science.

**The big idea in plain English:**
1. Take real crystal structures (like Aluminum, Iron, Diamond)
2. Gradually scramble them with random noise until they're unrecognizable
3. Teach a neural network to **un-scramble** noise back into a valid crystal
4. Use that skill in reverse: start from pure random noise and generate a **new** crystal structure

---

## Pipeline Overview

```
REAL DATA → NOISE → NOISY STRUCTURE → NEURAL NET TRAINS TO DENOISE
                                              ↓
NEW CRYSTAL ← PHYSICS CHECK ← GENERATED STRUCTURE ← REVERSE FROM PURE NOISE
```

---

## Cell by Cell Documentation

---

### Cell 1: Imports & Setup

**What it does:** Loads all required Python libraries and detects hardware.

**Key variables:**

| Variable | Value | Meaning |
|---|---|---|
| `device` | `'cuda'` or `'cpu'` | Where calculations run. CUDA = GPU (fast), CPU = fallback (slow) |
| `timestamp` | xxyyzz | Unique ID used for naming saved files |
| `plt.rcParams['figure.dpi']` | 150 | Resolution of all plots |

**Libraries used:**
- `torch`: The deep learning framework everything is built on
- `numpy`: Numerical arrays and math
- `matplotlib`: All the visualizations
- `torch.nn`: Neural network building blocks (layers, activations)
- `torch.nn.functional`: Functions like MSE loss
- `torch.utils.data.Dataset / DataLoader`: Tools for feeding batches of data to the model

---

### Cell 2: Dataset Creation: `create_demo_dataset()`

**What it does:** Creates 21 synthetic crystal structures to train on.

**Why we need this:** The model needs examples of "valid" crystal structures to learn from. We use three famous crystal geometries as starting points.

**The three base structures:**

| Crystal | Material | Atoms/Cell | Lattice Constant |
|---|---|---|---|
| FCC (Face-Centered Cubic) | Aluminum (Al, Z=13) | 4 | a = 4.05 Å |
| BCC (Body-Centered Cubic) | Iron (Fe, Z=26) | 2 | a = 2.87 Å |
| Diamond | Carbon (C, Z=6) | 8 | a = 3.57 Å |

**What is "fractional coordinate"?**  
Instead of absolute positions in Angstroms, atoms are described as fractions of the unit cell. So `[0.25, 0.25, 0.25]` means the atom is at 25% along each axis of the cell. All positions are in `[0, 1]`.

**Augmentation (making variants):** For each base structure, 6 more variants are created:
- **Thermal perturbation** (i < 3): Positions shifted by small random amounts (`noise_scale = 0.02 + i × 0.01`). Simulates atoms vibrating at temperature.
- **Strain** (i ≥ 3): The whole lattice is stretched/compressed by ±3% (`strain = 1.0 ± 0.03`). Simulates mechanical stress.

After adding noise, positions are wrapped back using `% 1.0` (periodic boundary conditions — if an atom goes past 1.0, it wraps back to 0.0, like Pac-Man).

**Final result:** 3 base + 18 variants = **21 structures total**.

---

### Cell 3: `CrystalStructureDataset` (PyTorch Dataset class)

**What it does:** Converts the raw list of crystal dictionaries into a format the neural network can consume in batches.

**The padding problem:** FCC has 4 atoms, BCC has 2, Diamond has 8. You can't stack these into a batch tensor because they're different sizes. Solution: **pad everything to `max_atoms = 8`**.

#### `__init__(self, crystal_data: List[dict])`
- **`crystal_data`** — The list of 21 crystal dictionaries from `create_demo_dataset()`
- Scans all structures to find `max_atoms = 8` (the largest structure — Diamond)

#### `__getitem__(self, idx)`
Fetches one structure and returns padded tensors:

| Tensor | Shape | dtype | Meaning |
|---|---|---|---|
| `positions` | `(8, 3)` | float32 | Fractional coordinates of atoms. Rows beyond n_atoms are all zeros (padding) |
| `atom_types` | `(8,)` | int64 | Atomic number (C=6, Al=13, Fe=26). Zero for padding |
| `lattice` | `(3, 3)` | float32 | The 3 lattice vectors. Each row is a vector in Angstroms |
| `mask` | `(8,)` | float32 | 1 for real atoms, 0 for padding |

**The mask is critical.** Without it, the model would try to learn about padding zeros as if they were real atoms, corrupting the training.

**DataLoader:** `DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)`
- `batch_size=4` — Process 4 structures at once
- `shuffle=True` — Random order each epoch (prevents the model memorizing order)
- `drop_last=True` — Discard the last batch if it has fewer than 4 structures (avoids inconsistent batch sizes)

This adds one more dimension: all tensors become `(batch_size, ...)`, so positions become `(4, 8, 3)`.

---

### Cell 4: `PhysicsValidator`

**What it does:** Checks whether a generated crystal structure is physically valid — specifically, that no two atoms are too close together.

**Why this matters:** In real physics, atoms cannot overlap. Electrons repel each other (Pauli Exclusion Principle). If two atoms are closer than ~0.7 Å, the structure is physically impossible.

#### `__init__(self, min_distance: float = 0.7)`
- **`min_distance`** — The minimum allowed distance between any two atoms (in Ångströms). Set to `0.7 Å` by default.

#### `check_minimum_distance(positions, lattice, mask)` → `(bool, float)`

**Parameters:**
- **`positions`**: `(batch, max_atoms, 3)` — Fractional coordinates
- **`lattice`**: `(batch, 3, 3)` — Lattice vectors
- **`mask`**: `(batch, max_atoms)` — Which slots are real atoms

**Returns:**
- `valid` — `True` if ALL distances ≥ min_distance
- `min_dist_found` — The closest pair found (in Å)

**How it works step-by-step:**
1. For each structure in the batch, extract only real atoms (where `mask > 0.5`)
2. Convert positions from fractional to Cartesian: `cart = pos @ lattice`
3. For every pair of atoms (i, j), compute the **Minimum Image Convention** distance:
   - Step 1: Difference vector in Cartesian: `Δr = r_i - r_j`
   - Step 2: Convert to fractional: `Δf = Δr · L⁻¹`
   - Step 3: Wrap to `[-0.5, 0.5]`: `Δf' = Δf - round(Δf)` — this finds the shortest path across periodic boundaries
   - Step 4: Back to Cartesian: `Δr' = Δf' · L`
   - Step 5: `dist = |Δr'|`
4. If any `dist < min_distance`, mark as invalid

**Complexity:** `O(N²)` per structure — every atom is checked against every other atom.

#### `enforce_pbc(positions)` → `torch.Tensor`
- Wraps all fractional coordinates to `[0, 1]` using `positions % 1.0`
- Applied after each reverse diffusion step to keep atoms inside the unit cell

---

### Cell 5 — `create_diffusion_schedule(T, beta_start, beta_end)` → `dict`

**What it does:** Pre-computes all the noise scaling factors for every timestep from t=0 to t=T-1.

**Parameters:**
- **`T`** (`timesteps`) — Total number of steps. Set to `200`. More steps = smoother noise addition but slower training.
- **`beta_start`** — Noise at the very first step. Set to `0.0001` (tiny noise).
- **`beta_end`** — Noise at the very last step. Set to `0.02` (substantial noise).

**What is β (beta)?** At each timestep `t`, a fraction β_t of the signal is replaced with noise. β_start is very small (barely any noise) and β_end is larger (a lot of noise). By t=200, ~87% of the signal has been destroyed.

**The linear schedule:** `betas = linspace(0.0001, 0.02, 200)` — β increases evenly from start to end.

**Derived quantities (all stored in the returned `dict`):**

| Key | Formula | Meaning |
|---|---|---|
| `betas` | `linspace(β_start, β_end, T)` | Noise variance at each step |
| `alphas` | `1 - betas` | Signal retained at each step |
| `alphas_cumprod` (ᾱ_t) | `cumprod(alphas)` | Cumulative signal from t=0 to t |
| `sqrt_alphas_cumprod` | `√ᾱ_t` | How much of the original signal remains |
| `sqrt_one_minus_alphas_cumprod` | `√(1-ᾱ_t)` | How much noise is mixed in |
| `posterior_variance` | `betas · (1-ᾱ_{t-1}) / (1-ᾱ_t)` | Noise added during reverse step |

**Concrete values at key timesteps:**

| t | Signal (√ᾱ_t) | Noise (√(1-ᾱ_t)) | Interpretation |
|---|---|---|---|
| 1 | 0.9999 | 0.0100 | Almost original structure |
| 51 | 0.9357 | 0.3527 | Slightly noisy |
| 101 | 0.7723 | 0.6353 | 50/50 signal and noise |
| 151 | 0.5617 | 0.8273 | Mostly noise |
| 200 | 0.3636 | 0.9316 | Almost pure noise |

---

### Cell 6: `forward_diffusion_sample(x_0, t, schedule, noise=None)` → `(x_t, noise)`

**What it does:** Given a clean crystal structure `x_0`, adds the correct amount of noise for timestep `t`.

**This is the "training time corruption."** We never actually need to run 200 sequential noise-adding steps, thanks to the closed-form formula, we can jump directly to any timestep.

**Parameters:**
- **`x_0`**: `(batch, max_atoms, 3)` — Clean fractional coordinates
- **`t`**: `(batch,)` — Which timestep to jump to for each sample in the batch
- **`schedule`**: The dictionary from `create_diffusion_schedule()`
- **`noise`**: Optional pre-generated noise. If `None`, fresh noise is sampled: `ε ~ N(0, I)`

**The formula:**
```
x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε
```
- **`√ᾱ_t`** — Scales down the signal
- **`√(1-ᾱ_t)`** — Scales up the noise
- As t → T, signal → 0 and noise → 1: the structure is destroyed

**Returns:**
- `x_t` — The noisy structure at timestep t
- `noise` — The exact noise that was added (the model will later try to predict this)

**Shape trick:** `schedule['sqrt_alphas_cumprod'][t][:, None, None]` — The `[:, None, None]` reshapes from `(batch,)` to `(batch, 1, 1)` so it can broadcast against `(batch, max_atoms, 3)`.

---

### Cell 7: Neural Network Architecture

#### `SinusoidalPositionEmbeddings(dim)`

**What it does:** Converts a timestep number (e.g., `t=142`) into a 128-dimensional vector that the network can understand.

**Why not just feed in the number directly?** A raw integer like `142` carries little geometric meaning. Sinusoidal embeddings encode the timestep into a rich pattern (like musical harmonics) that the network can work with.

**Formula:**
```
PE(t, 2i)   = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

**Parameter:**
- **`dim`** — Output dimension (`128`). Higher = more expressive encoding.

**Forward:**
- **Input**: `time` — `(batch,)` integer timesteps
- **Output**: `embeddings` — `(batch, 128)` float vectors

---

#### `CrystalDenoisingNetwork(hidden_dim, time_embed_dim, num_layers)`

**What it does:** The core neural network. Takes a noisy crystal structure and a timestep, and predicts what noise was added.

**Parameters:**
- **`hidden_dim`** (`128`): Width of internal layers. More = more capacity, slower training. Options: 64, 128, 256, 512.
- **`time_embed_dim`** (`128`): Width of the time embedding. Usually matches `hidden_dim`.
- **`num_layers`** (`4`): Number of residual processing blocks. More = deeper reasoning but slower. Options: 2, 4, 6, 8.

**Architecture (4 stages):**

```
Input: x_t (batch, 8, 3)   +   t (batch,)

Stage 1 — Time Embedding:
  t → SinusoidalEmbedding(128) → Linear(128→128) → GELU → Linear(128→128)
  t_emb: (batch, 128)  →  reshape to (batch, 1, 128)

Stage 2 — Encoder:
  x_t: (batch, 8, 3) → Linear(3→128) → GELU → Linear(128→128)
  h: (batch, 8, 128)
  h = h + t_emb     ← inject time information into atom features

Stage 3 — Residual Blocks × 4:
  for each block:
    h = h + [Linear(128→128) → GELU → Linear(128→128)](h)
  (The "+ h" is the "residual connection" — helps gradients flow during training)

Stage 4 — Decoder:
  h: (batch, 8, 128) → Linear(128→128) → GELU → Linear(128→3)
  noise_pred: (batch, 8, 3)
  noise_pred = noise_pred * mask[:, :, None]   ← zero out padding

Output: predicted noise (batch, 8, 3)
```

**Total parameters:** 199,043 (~200K) (I asked AI for this, I did not calculate this myself)

**What is GELU?** Gaussian Error Linear Unit, an activation function (non-linearity) that decides how much of a signal to "pass through". Similar to ReLU but smoother.

**What is a residual connection?** `h = h + layer(h)` means the input is always added back. This prevents the network from "forgetting" information as it goes deeper, and makes training much more stable.

**`forward(x, t, mask)` → `noise_pred`:**
- **`x`**: `(batch, 8, 3)`: Noisy positions
- **`t`**: `(batch,)`: Current timestep
- **`mask`**: `(batch, 8)`: Atom/padding mask
- **Returns**: `(batch, 8, 3)`: Predicted noise

---

### Cell 8: `CrystalDiffusionModel`

**What it does:** The "master class" that wraps the schedule, network, validator, optimizer, and all training/generation logic in one place.

#### `__init__(self, timesteps=200, device='cpu')`
Creates and connects all components:
- **`timesteps`** : Total diffusion steps. Passed to `create_diffusion_schedule()`
- **`device`** : `'cuda'` or `'cpu'`. All tensors moved here.

Internal attributes:
- `self.schedule`: The dict of noise arrays from `create_diffusion_schedule(200)`
- `self.model`: `CrystalDenoisingNetwork(128, 128, 4)`
- `self.validator`: `PhysicsValidator(0.7)`
- `self.optimizer`: `AdamW(lr=1e-4, weight_decay=1e-4)`

**AdamW optimizer:** Adam with weight decay. Adaptively adjusts the learning rate for each parameter. `weight_decay=1e-4` prevents the model from relying too heavily on any single weight (regularization).

---

#### `compute_loss(self, batch)` → `torch.Tensor`

The mathematical heart of training. Computes how bad the model's noise prediction is.

**Algorithm:**
```
1. x_0 = batch['positions']            ← clean crystal structure
2. t ~ Uniform(0, 200)                 ← random timestep for each sample
3. ε ~ N(0, I)                         ← random noise
4. x_t = forward_diffusion_sample(x_0, t, schedule, ε)
5. ε_pred = self.model(x_t, t, mask)   ← neural net predicts noise
6. loss = MSE(ε_pred * mask, ε * mask) ← only count real atoms
```

**MSE Loss:** `mean((ε_pred - ε)²)` over all real atom positions. A loss of 0.0 would mean perfect prediction. We got from ~0.57 to ~0.15 (73.8% improvement).

---

#### `train_step(self, batch)` → `float`

Performs one complete gradient update.

```
1. Set model to train mode
2. loss = compute_loss(batch)
3. optimizer.zero_grad()          ← clear old gradients
4. loss.backward()                ← compute ∂loss/∂(each weight)
5. clip_grad_norm(max=1.0)        ← prevent exploding gradients
6. optimizer.step()               ← update all weights
7. return loss.item()
```

**Gradient clipping:** If gradients become very large, they're rescaled to have maximum norm 1.0. This prevents catastrophic weight updates ("exploding gradients").

---

#### `sample(self, batch_size, max_atoms, lattice, mask)` → `torch.Tensor`

The generation algorithm. Runs the **reverse diffusion process** to create a new crystal from noise.

**Algorithm (DDPM Sampling):**
```
Start: x_T ~ N(0, I)    ← pure random noise

for t = T-1 down to 0:
    ε_pred = self.model(x_t, t, mask)
    
    β_t    = schedule.betas[t]
    ᾱ_t    = schedule.alphas_cumprod[t]
    α_t    = 1 - β_t
    
    coeff  = β_t / √(1 - ᾱ_t)
    μ_θ    = (1/√α_t) · (x_t - coeff · ε_pred)   ← denoised mean
    
    if t > 0:
        z       ~ N(0, I)
        σ_t     = √(posterior_variance[t])
        x_{t-1} = μ_θ + σ_t · z                  ← stochastic step
    else:
        x_0 = μ_θ                                 ← final clean structure

x_0 = enforce_pbc(x_0)    → wrap to [0,1]³
```

**Why add noise during the reverse process?** Adding small amounts of noise at each reverse step (except the last) prevents the model from collapsing to a single output. It maintains diversity in the generated structures.

**Parameters:**
- **`batch_size`** — How many structures to generate at once
- **`max_atoms`** — Size of the position tensor (set to 8, matching training data)
- **`lattice`** — The unit cell to use. Borrowed from a training structure.
- **`mask`** — Which atom slots are "active". Borrowed from a training structure.

---

### Cell 9: Training Loop

**What it does:** Runs the full training process.

**Configuration:**
```python
config = {
    'epochs': 50,         # Full passes through the dataset
    'batch_size': 4,      # Structures per gradient update
    'log_interval': 10    # Print progress every 10 epochs
}
```

**How training works:**
- 21 structures ÷ batch_size 4 = **5 batches per epoch** (1 batch is dropped since 21 is not divisible by 4)
- 50 epochs × 5 batches = **250 total gradient updates**
- Each update: sample random timesteps for each structure, add corresponding noise, predict it, measure error, update weights

**Training results from this run:**
- Initial loss: 0.5688
- Final loss: 0.1492
- Best loss: 0.0999
- Improvement: 73.8%

**What does "loss decreasing" mean?** The model is getting better at predicting which noise was added to a structure. A lower loss = better denoising = better generation.

---

### Cell 10: Reverse Diffusion Visualization

Calls `model.sample()` with a fresh random seed and visualizes the denoising process at several timesteps (t=199, 150, 100, 50, 0). Shows how the structure "emerges" from noise.

---

### Cell 11: Physics Validation of Generated Structure

Calls `validator.check_minimum_distance()` on the generated structure. Reports whether it passes the 0.7 Å threshold. In the example run, the structure **FAILED** with min_dist = 0.387 Å — atoms were too close. This is expected with only 50 training epochs and a small dataset; more training improves this.

---

### Cell 12: Structure Comparison

Computes and compares metrics between the training data and the generated structure:

| Metric | Formula | Meaning |
|---|---|---|
| `density` | `mass_amu × 1.66054e-27 kg / (volume_Å³ × 1e-30 m³) / 1000` | Grams per cubic centimeter |
| `volume` | `|det(L)|` where L is the 3×3 lattice matrix | Ångströms³ |
| `RDF` | Radial Distribution Function | Histogram of all pairwise distances — shows neighbor shells |
| `coordination_numbers` | Atoms within cutoff radius | Average neighbors per atom |

**Visualizations produced:**
- 3D scatter plot: real crystal vs generated crystal
- RDF comparison: real vs generated (should show similar peaks if model learned well)
- Coordinate histograms: distribution of x, y, z values

---

### Cells 13–15: Batch Generation & Analysis

Generates multiple structures (the user can set how many) and collects statistics across all of them.

**For each generated structure:**
1. Run `model.sample()` 
2. Run `validator.check_minimum_distance()` → `valid` flag
3. Compute density, volume, coordination numbers, min distance

**Batch plots (12 subplots):**
1. Density distribution histogram
2. Volume distribution histogram  
3. Coordination number distribution
4. Min distance distribution with 0.7 Å threshold line
5. Physics validation success rate (pie chart: PASSED vs FAILED %)
6. 4 example 3D structure plots
7. Atoms vs density scatter plot
8. Volume vs density scatter (colored by valid/invalid)

---

### Cell 16: Summary Report

Prints a complete summary of the run, including all configuration, architecture, training results, and generation results. Also lists all output files saved to the `outputs/` folder.

---

## Complete Parameter Reference

| Parameter | Where Set | Default | Effect of Increasing | Effect of Decreasing |
|---|---|---|---|---|
| `T` (timesteps) | `CrystalDiffusionModel.__init__` | 200 | More gradual diffusion, higher quality, slower | Faster but coarser |
| `hidden_dim` | `CrystalDenoisingNetwork.__init__` | 128 | More expressive, more params, slower | Faster but less capable |
| `num_layers` | `CrystalDenoisingNetwork.__init__` | 4 | Deeper reasoning, more params | Faster but shallower |
| `epochs` | `config` dict | 50 | Better learning, more time | Faster but worse quality |
| `batch_size` | `config` dict | 4 | Smoother gradients, more memory | More updates but noisier |
| `lr` | `AdamW` optimizer | 1e-4 | Faster training but may overshoot | Stable but very slow |
| `weight_decay` | `AdamW` optimizer | 1e-4 | More regularization | Less regularization |
| `min_distance` | `PhysicsValidator.__init__` | 0.7 Å | Stricter validation | More permissive |
| `beta_start` | `create_diffusion_schedule` | 0.0001 | More initial noise | Less initial noise |
| `beta_end` | `create_diffusion_schedule` | 0.02 | More final noise | Less final noise |
| `max_atoms` | Inferred from data | 8 | More atoms possible | Memory savings |

---

## Data Flow Summary

```
crystal_data (list of 21 dicts)
    ↓ CrystalStructureDataset
dataset (padded to max_atoms=8)
    ↓ DataLoader (batch_size=4)
batches: {positions(4,8,3), atom_types(4,8), lattice(4,3,3), mask(4,8)}
    ↓ create_diffusion_schedule(T=200)
schedule: {betas, alphas_cumprod, sqrt_*}  (length 200 arrays)
    ↓ forward_diffusion_sample(x_0, t, schedule)
x_t (4,8,3): noisy positions    +    ε (4,8,3): true noise
    ↓ CrystalDenoisingNetwork.forward(x_t, t, mask)
ε_pred (4,8,3): predicted noise
    ↓ MSE Loss
scalar loss value
    ↓ AdamW optimizer
updated network weights
    ...after 250 updates...
trained model
    ↓ model.sample(...)
x_0_generated (1,8,3): fractional coordinates of new crystal
    ↓ PhysicsValidator.check_minimum_distance(...)
valid: bool    min_dist: float
    ↓ Analysis
density, volume, RDF, coordination numbers, plots
```

(Just refer to the html file I made, should be in this repo)

---

## Common Errors and Their Meaning

| Error | Cause | Fix |
|---|---|---|
| Physics validation FAILED | Model not trained enough, atoms overlap | Train more epochs (50 → 100+) |
| Loss not decreasing | Learning rate too high or too low | Try lr=1e-3 (higher) or lr=1e-5 (lower) |
| Structures look random | Too few epochs, model hasn't learned | Train 200+ epochs, add more data |
| Out of memory | batch_size or hidden_dim too large | Reduce batch_size to 2, hidden_dim to 64 |
| Loss = NaN | Exploding gradients | Already handled by clip_grad_norm(1.0); if persists, reduce lr |

---

## Some Key Terms

**Fractional coordinates:** Atom positions expressed as fractions of the unit cell dimensions (values in [0, 1]).

**Lattice vectors:** Three vectors (rows of the 3×3 matrix L) that define the shape and size of the unit cell.

**Forward diffusion:** The process of progressively adding Gaussian noise to a structure until it's pure noise. Used during training.

**Reverse diffusion:** The process of starting from pure noise and iteratively denoising until a valid structure emerges. Used during generation.

**DDPM:** Denoising Diffusion Probabilistic Model — the mathematical framework this model is based on (Ho et al., 2020).

**Periodic Boundary Conditions (PBC):** The crystal repeats infinitely in all directions. Atoms near an edge of the unit cell are neighbors with atoms on the opposite edge.

**Minimum Image Convention:** When computing distances under PBC, always use the shortest path (possibly through a periodic boundary) rather than the direct path within the unit cell.

**MSE Loss:** Mean Squared Error — `mean((prediction - truth)²)`. The lower, the better.

**Residual connection:** `output = input + f(input)`. Helps very deep networks train by giving gradients a "shortcut" back to early layers.

**AdamW:** A gradient descent optimizer that adapts the learning rate for each weight and adds weight decay to prevent overfitting.

**Gradient clipping:** Scaling down gradients if their total magnitude exceeds a threshold (1.0 here). Prevents catastrophic weight updates.

**GELU activation:** `x · Φ(x)` where Φ is the Gaussian CDF. Smoother alternative to ReLU — "gates" the signal based on its magnitude.

**Sinusoidal embeddings:** Converting a scalar timestep `t` into a high-dimensional vector using sin/cos functions at many frequencies. Allows the network to distinguish fine differences between timesteps.
