# Guided Diffusion for Altermagnet Crystal Generation

## A Complete Beginner's Guide Every Step, Every Parameter, Every Theory

# PART 1: WHAT IS AN ALTERMAGNET?

## 1.1 Starting From Scratch --- Magnetism in Materials

Every atom contains electrons, and every electron behaves like a tiny
bar magnet. This property is called **spin** --- it is a quantum
mechanical property of electrons that comes in exactly two flavours:

**Spin up** (↑): the electron's magnetic field points "up" **Spin down**
(↓): the electron's magnetic field points "down"

In most everyday materials, electrons pair up in opposite spins and
cancel each other out, so the material has no net magnetism. But in some
materials, the electron spins are not cancelled --- they align in
patterns that give the material a collective magnetic personality.

There are three main families of magnetic materials, distinguished by
how the spins are arranged:

### Family 1: Ferromagnets (e.g. iron, nickel, cobalt)

All the atomic spins point in the **same direction**:

    ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
    ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
    ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑

Net magnetisation: **Large and nonzero** You can pick up a paper clip
with it Problem for devices: the stray magnetic field causes crosstalk
between components

### Family 2: Antiferromagnets (e.g. chromium, MnO)

Spins alternate up/down in a regular pattern:

    ↑ ↓ ↑ ↓ ↑ ↓ ↑ ↓
    ↓ ↑ ↓ ↑ ↓ ↑ ↓ ↑
    ↑ ↓ ↑ ↓ ↑ ↓ ↑ ↓

Net magnetisation: **Zero** (they cancel perfectly) No stray field ---
great for dense devices Problem: very difficult to read or write because
there's no net signal

### Family 3: Altermagnets (e.g. RuO₂, CrSb, MnTe) --- DISCOVERED \~2022

Spins also alternate, but with a **crucial twist**: the spinup atoms and
spindown atoms have **different spatial orientations** in the crystal.
They are NOT related by simple inversion or translation --- they are
related by a **rotation** of the crystal:

    ↑  ↓  ↑  ↓      ← spin alternates...
     \ | / \ | /
      \|/   \|/     ← ...but the local atomic environment is ROTATED not inverted
      /|\   /|\

Net magnetisation: **Zero** (like an antiferromagnet) But electronic
bands are **spinsplit** (like a ferromagnet) **Best of both worlds**

## 1.2 The Rotation Symmetry Criterion --- The Defining Rule

This is the central rule you must understand:

> In an altermagnet, the spinup sublattice and the spindown sublattice
> are **chemically identical** but **physically rotated** with respect
> to each other. The rotation is a symmetry of the crystal, but NOT
> inversion and NOT translation.

**What does this mean concretely?**

Take RuO₂ (ruthenium dioxide), the most famous altermagnet: There are 2
Ru (ruthenium) atoms per unit cell Atom 1 (Ru at corner): surrounded by
O atoms in one rotational arrangement → assigned spin ↑ Atom 2 (Ru at
body centre): surrounded by O atoms rotated by **90 degrees** → assigned
spin ↓

The two Ru sites are **chemically identical** (both Ru, both bonded to 6
O atoms at same distances), but their local environments are **rotated
90° relative to each other**. This 90° rotation is called a **C₄
symmetry operation** (rotation by 360/4 = 90 degrees).

This rotation is what gives altermagnets their unusual property: the
spin splitting in momentum space has the same rotational symmetry as the
realspace crystal rotation.

## 1.3 Why Does This Matter? --- The Physics Payoff

The rotation symmetry breaking produces a **momentumdependent spin
splitting** in the electronic band structure. This means:

Electrons moving in one direction through the crystal have spin ↑
Electrons moving in the perpendicular direction have spin ↓

This is called **kspace spin splitting** (k = crystal momentum
direction).

**Practical consequences:** 1. **Anomalous Hall Effect** --- produces a
transverse voltage without an external magnetic field 2. **Giant
Magnetoresistance** --- large resistance change in a magnetic field 3.
**Spinpolarised currents** --- can be used to write/read magnetic memory
4. **All of this at zero net magnetisation** --- no stray fields, no
crosstalk in circuits

**Known altermagnets and why they're exciting:**

| Material \| Type \| Transition Temp. \| Why Important \|

\|\|\|\|\| \| RuO₂ \| Rutile \| \>300 K (room temp!) \| First confirmed,
large spin splitting \| \| CrSb \| NiAs \| 705 K \| Highest known Tc,
very stable \| \| MnTe \| NiAs \| 307 K \| Thinfilm compatible \| \|
MnF₂ \| Rutile \| 67 K \| Model system, wellstudied \| \| La₂CuO₄ \|
Perovskite \| 325 K \| Connected to highTc superconductors \|

## 1.4 Crystal Structure Terminology (Explained Simply)

Before we encode crystals as numbers, let's understand the vocabulary:

### Unit Cell

The smallest repeating unit of a crystal. Like a single tile in an
infinitely repeating floor pattern. We only need to describe one unit
cell --- the rest of the crystal is built by copying it in three
directions.

### Lattice Vectors

Three vectors **a₁, a₂, a₃** that describe the directions and lengths of
the unit cell edges. Packed into a **3×3 matrix L**:

    L = [a₁]   =  [ a₁ₓ  a₁ᵧ  a₁_z ]
        [a₂]      [ a₂ₓ  a₂ᵧ  a₂_z ]
        [a₃]      [ a₃ₓ  a₃ᵧ  a₃_z ]

For a simple cube: L = diag(a, a, a) where a is the side length in
Angstroms (1 Å = 10⁻¹⁰ metres).

### Fractional Coordinates

The position of each atom expressed as fractions of the lattice vectors.
An atom at fractional position (0.5, 0.5, 0.5) sits exactly at the body
centre of the unit cell. Values always between 0 and 1. The actual
position in real space (Cartesian coordinates) is:

    r_cartesian = L^T × r_fractional

### Space Group

A number (1--230) that classifies the full symmetry of the crystal ---
which rotations, reflections, and translations leave the crystal
unchanged. Altermagnets live in specific space groups that have the
right rotation symmetry.

### Sublattice

A subset of atomic sites that are all chemically equivalent (same
element, same bonding environment). In a magnetic crystal, the **spinup
sublattice** contains all spinup sites, and the **spindown sublattice**
contains all spindown sites.

# PART 2: WHAT IS A DIFFUSION MODEL?

## 2.1 The Core Idea --- Learning to Undo Chaos

A diffusion model is a type of neural network that learns to generate
new data (in our case, crystal structures) by learning to **reverse a
noising process**.

Here is the complete intuition in one sentence:

> We teach the model to "unblur" random noise into realistic crystal
> structures --- by training it on millions of examples of
> partiallynoised crystals.

### The Two Processes

**Forward Process** (we define this, no learning needed): 1. Take a real
crystal structure x₀ 2. Add a tiny bit of Gaussian noise → x₁ 3. Add a
bit more noise → x₂ 4. Keep going for T=1000 steps 5. At step T, you
have pure random noise xT that looks nothing like a crystal

**Reverse Process** (this is what we LEARN): 1. Start from pure noise xT
2. Apply the trained network to predict "which direction to remove
noise" → x\_{T1} 3. Repeat until you reach x₀ 4. x₀ is a brand new
crystal structure that was never in the training set!

### Why Does This Work?

The key insight is that if you learn to reverse the noising process, you
have implicitly learned the **probability distribution** of all possible
crystals. Generating a new sample means sampling from this learned
distribution.

## 2.2 The Mathematics of Forward Diffusion

### Notation first

x₀ = clean crystal (what we want to generate) xₜ = crystal after t steps
of noising ε (epsilon) = random Gaussian noise sample βₜ (beta_t) =
noise schedule --- how much noise to add at step t αₜ = 1 βₜ (how much
signal to keep at step t) ᾱₜ (alpha_bar_t) = cumulative product of all
αₛ from s=1 to t

### The Forward Diffusion Formula

At each step t, we mix the previous crystal with noise:

    q(xₜ | x_{t1}) = N(xₜ ; √(1−βₜ) · x_{t1},  βₜ · I)

Reading this: "xₜ is drawn from a Gaussian distribution centred at a
slightly shrunk version of x\_{t1}, with variance βₜ"

### The Magic Shortcut (Direct Sampling at Any Timestep)

Because the forward process is Gaussian, we can skip directly to any
step t without running all t steps:

    q(xₜ | x₀) = N(xₜ ; √ᾱₜ · x₀,  (1 − ᾱₜ) · I)

Or equivalently, we can write it as a simple formula:

    xₜ = √ᾱₜ · x₀  +  √(1 − ᾱₜ) · ε
            ↑                ↑
        signal part      noise part
        (shrinks to 0     (grows from 0
         as t increases)   to 1 as t increases)

When t=0: xₜ = x₀ (pure signal, no noise) When t=T: xₜ ≈ ε (pure noise,
signal is gone)

### The Noise Schedule βₜ

βₜ controls how quickly we destroy the signal. We use the **cosine
schedule**:

    ᾱₜ = cos²( (t/T + 0.008) / (1 + 0.008) × π/2 )

This is better than a linear schedule because it: Decays slowly at the
start (preserving more crystal structure information in early steps)
Decays faster in the middle (avoids saturation at the end) Leads to
better quality samples

## 2.3 The Reverse Process --- Denoising

### What the Score Network Learns

The network εθ(xₜ, t) --- called a "score network" or "noise predictor"
--- learns to predict the noise **ε** that was added to produce xₜ from
x₀.

Training objective:

    Minimize E_{x₀, ε, t} [ || ε  −  εθ(xₜ, t) ||² ]

In plain English: given a noisy crystal xₜ and the timestep t, predict
what the noise was. If you can predict the noise, you can subtract it
and get a cleaner crystal.

### Generating a New Sample (Reverse Diffusion)

Start from pure noise: xT \~ N(0, I)

For each step t from T down to 1: 1. Predict noise: ε̂ = εθ(xₜ, t) 2.
Compute the denoised estimate: `μ = (1/√αₜ) × [xₜ − (βₜ/√(1−ᾱₜ)) × ε̂]`
3. Add a small amount of fresh noise (for stochasticity, so each run
gives a different result): `x_{t1} = μ + √βₜ × z     where z ~ N(0, I)`

At t=1 (final step): x₀ = μ (no added noise at the last step)

## 2.4 DiffCSP --- Extending Diffusion to Crystals

The paper "DiffCSP" (Crystal Structure Prediction by Joint Equivariant
Diffusion, Jiao et al. NeurIPS 2023) extends vanilla diffusion to handle
the special properties of crystal structures:

### Challenge 1: Periodic Boundary Conditions

Fractional coordinates must stay in \[0, 1). If an atom drifts to 1.2,
it wraps back to 0.2. **Solution:** Use wrapped Gaussian (von Mises)
distributions on the torus \[0,1)³

### Challenge 2: Lattice Matrix Must Be Physical

The lattice matrix L must be positive definite (physical crystal volumes
must be positive). **Solution:** Diffuse in the SPD (Symmetric Positive
Definite) manifold; in practice we use the Cholesky parameterisation

### Challenge 3: Equivariance

If you rotate the entire crystal, the noise prediction should rotate
consistently. **Solution:** Use SE(3)equivariant graph neural networks
(in our simplified demo, we approximate this with an MLP)

### Challenge 4: Variable Number of Atoms

Crystals can have different numbers of atoms per unit cell.
**Solution:** Condition on composition (fix which elements are present);
pad to fixed size N_max

## 2.5 Guided Diffusion --- Steering Towards Altermagnets

### The Problem with Unconditional Generation

A diffusion model trained on all crystals will generate "some crystal"
--- but not necessarily an altermagnet. We want to generate specifically
altermagnetic crystals.

### Classifier Guidance

Dhariwal & Nichol (NeurIPS 2021) showed that you can **steer** a
diffusion model's sampling process using the gradient of a classifier:

    ∇ log p_guided(x) = ∇ log p(x)  +  λ · ∇ log p(y=target | x)
                        ─────────────   ──────────────────────────
                        score network      guidance gradient
                        (any crystal)      (towards target class)

Where λ (lambda) is the **guidance strength** --- a hyperparameter you
choose: λ = 0: purely unconditional, generates any crystal λ = 1: soft
nudge towards altermagnets λ = 3: strong bias, most samples will be
altermagnetic λ \> 10: too strong, samples lose quality
(overconstrained)

### Our Altermagnetic Guidance Function g(x)

We define a differentiable function g(x) that scores how "altermagnetic"
a crystal configuration is:

    Step 1: Extract fractional coordinates and spin labels from x
    Step 2: Compute centroid of spinup sublattice: c_up = mean(positions where spin=+1)
    Step 3: Compute centroid of spindown sublattice: c_down = mean(positions where spin=1)
    Step 4: Compute angle θ = angle between c_up and c_down vectors in the abplane
    Step 5: Score = cos²(θ − θ_target)

Where θ_target depends on crystal symmetry: **θ_target = 90°** for
tetragonal/rutile systems (C₄ rotation symmetry, space group 136)
**θ_target = 60°** for hexagonal/NiAs systems (C₃ rotation symmetry,
space group 194)

**Why cosine squared?** When θ = θ_target: cos²(0) = 1.0 → perfect
altermagnet score When θ = 0° (both sublattices aligned): cos²(90°) =
0.0 → not altermagnetic The score is always in \[0, 1\], smooth, and
differentiable

### Computing the Guidance Gradient

We compute the gradient of g(x) numerically using **finite
differences**:

    ∂g/∂xᵢ ≈ [g(x + δ·eᵢ) − g(x − δ·eᵢ)] / (2δ)

Where eᵢ is a unit vector in the ith dimension, and δ is a small step
(10⁻³).

This gives us a vector pointing in the direction of steepest increase in
the altermagnetic score.

# PART 3: ENCODING CRYSTALS AS NUMBERS

## 3.1 The Complete Feature Vector --- x ∈ ℝ⁸¹

We encode each crystal as a flat vector of 81 floatingpoint numbers:

    x = [ L_flat | X_flat | Z_flat | S ]
           9 dims   18 dims   48 dims  6 dims

### Lattice Matrix L → 9 numbers

The 3×3 lattice matrix flattened rowbyrow and divided by 10
(normalisation):

    L = [[a, 0, 0],    → [a/10, 0, 0, 0, b/10, 0, 0, 0, c/10]
         [0, b, 0],       for a simple orthorhombic crystal
         [0, 0, c]]

### Fractional Coordinates X → 18 numbers

6 atomic sites × 3 coordinates per site (padded to N_max=6 atoms):

    X = [[x₁, y₁, z₁],   → [x₁, y₁, z₁, x₂, y₂, z₂, ..., x₆, y₆, z₆]
         [x₂, y₂, z₂],       all values between 0 and 1
         ...                  (fractional coordinates)
         [x₆, y₆, z₆]]

### Element Embeddings Z → 48 numbers

6 atomic sites × 8 element features per site:

    For each element, compute 8 features:
    z = [ atomic_number/100,
          electronegativity/4,
          covalent_radius/2,
          is_transition_metal (0 or 1),
          is_halide (0 or 1),
          is_chalcogenide (0 or 1),
          period/7,
          group/18 ]

All divided by their maximum values so they fall in \[0, 1\].

Why these features? Each one captures a different aspect of atomic
bonding behaviour: Atomic number: overall size/complexity
Electronegativity: how strongly an atom attracts electrons (controls
ionic vs covalent bonding) Covalent radius: physical size of the atom
Transition metal flag: these elements are the ones that carry magnetic
moments Halide/chalcogenide: common anion types in magnetic oxides and
fluorides Period/Group: position in periodic table, correlated with
orbital occupation

### Spin Labels S → 6 numbers

One spin value per site: +1 (spin up), 1 (spin down), or 0
(nonmagnetic):

    S = [+1, 1, 0, 0, 0, 0]   → first two sites magnetic, rest nonmagnetic

# PART 4: THE SCORE NETWORK ARCHITECTURE

## 4.1 Sinusoidal Time Embedding

The network needs to know **which timestep t we are at** during the
noising/denoising process, because at t=100 (lightly noised), the
denoising job is very different from t=900 (heavily noised).

We encode t using sinusoidal functions at many frequencies, exactly like
positional encoding in Transformers:

    γ(t) = [sin(t · f₀), cos(t · f₀),
             sin(t · f₁), cos(t · f₁),
             ...
             sin(t · f₆₃), cos(t · f₆₃)]   ∈ ℝ¹²⁸

    where fₖ = 1/10000^(k/64)   gives frequencies from 1 down to 1/10000

Why sinusoids? Every value of t gets a **unique** vector **Nearby
timesteps** get **similar** embeddings (smooth interpolation) Works for
any t value, including ones not seen during training

## 4.2 Residual Connections

A residual connection ("skip connection") means we add the input of a
layer to its output:

    h_out = h_in + f(h_in)    instead of    h_out = f(h_in)

This matters because: Gradients can flow directly backwards through the
addition (no vanishing gradient) The network only needs to learn a
**residual correction** (easier than learning from scratch) Deeper
networks become trainable

## 4.3 SiLU Activation

SiLU (Sigmoidweighted Linear Unit) is the activation function we use:

    SiLU(x) = x × σ(x) = x / (1 + e^{x})

It is smoother than ReLU, has nonzero gradients everywhere, and
empirically works better for this type of continuous data generation.

## 4.4 LayerNorm

LayerNorm normalises each activation vector to have mean 0 and standard
deviation 1:

    LayerNorm(x) = (x − mean(x)) / (std(x) + ε)

This prevents activations from exploding or vanishing during training,
and makes training much more stable.

# PART 5: THE ADAM OPTIMISER

## 5.1 Why Not Plain Gradient Descent?

Vanilla gradient descent updates weights as:

    θ ← θ − lr × gradient

Problems: it uses the same learning rate for all parameters, sensitive
to choice of lr.

## 5.2 Adam (Adaptive Moment Estimation)

Adam maintains a running estimate of the mean (first moment) and
variance (second moment) of the gradients:

    m ← β₁ × m + (1−β₁) × gradient         first moment (momentum)
    v ← β₂ × v + (1−β₂) × gradient²        second moment (RMSprop)

    m_hat = m / (1 − β₁ᵗ)                  bias correction
    v_hat = v / (1 − β₂ᵗ)

    θ ← θ − lr × m_hat / (√v_hat + ε)

**Typical hyperparameter values:** lr (learning rate) = 0.001 to 0.005
--- how large each step is β₁ = 0.9 --- momentum for first moment (90%
memory of past gradients) β₂ = 0.999 --- momentum for second moment
(99.9% memory) ε = 10⁻⁸ --- tiny constant to prevent division by zero

**Why Adam is better:** Large gradients → smaller effective step (via
v_hat) Small gradients → larger effective step (better exploration)
Adapts perparameter, not one size fits all

# PART 6: READING THE NOTEBOOK

## 6.1 What Each Section Does

1.  **Imports & Setup** --- loads numpy, matplotlib, sets dark theme
2.  **Dataset Generation** --- creates 192 synthetic altermagnets
    (rutile + NiAs types) with random variation
3.  **Dataset Statistics** --- histograms of lattice parameters, PCA
    scatter, pie chart by space group
4.  **Crystal Structure Visualisation** --- draws unit cells with spin
    arrows
5.  **Symmetry Analysis** --- measures sublattice rotation angle θ and
    order parameter η
6.  **Feature Engineering** --- encodes crystals as 81dimensional
    vectors
7.  **Forward Diffusion** --- shows how noise accumulates over T=1000
    steps
8.  **Score Network** --- defines and tests the MLP noise predictor
9.  **Training** --- 400 steps of Adam training, shows loss convergence
10. **Guidance Signal** --- demonstrates the altermagnetic scoring
    function
11. **Guided Sampling** --- generates new crystals at λ=0, 1, 3
12. **Results Analysis** --- compares guidance levels on score, angle
    accuracy, validity

## 6.2 Key Parameters to Know

| Parameter \| Value \| Meaning \|

\|\|\|\| \| T \| 1000 \| Number of diffusion timesteps \| \| β_min \|
0.0001 \| Smallest noise increment \| \| β_max \| 0.02 \| Largest noise
increment \| \| N_max \| 6 \| Maximum atoms per unit cell (padding
target) \| \| x_dim \| 81 \| Crystal feature vector dimension \| \|
t_dim \| 128 \| Time embedding dimension \| \| hidden \| 256 \| MLP
hidden layer width \| \| lr \| 0.005 \| Adam learning rate \| \| batch
\| 16 \| Crystals per training step \| \| λ (lambda) \| 0, 1, 3 \|
Guidance strengths tested \| \| θ_target \| 90° \| Target rotation angle
for rutile altermagnets \| \| δ \| 0.001 \| Finite difference step for
guidance gradient \|

## 6.3 How to Interpret the Plots

**Dataset Statistics:** Look for clear separation between Rutile (SG
136) and NiAs (SG 194) in lattice parameter space. The c parameter is
much larger for NiAs (\~6--7 Å) than Rutile (\~3 Å), reflecting the
hexagonal layered structure.

**Crystal Structure:** Red/pink atoms and arrows = spin up sublattice.
Blue = spin down. Notice how the two magnetic sublattices are spatially
rotated, not just swapped. The angle between them should be near 90°
(Rutile) or 60° (NiAs).

**Forward Diffusion:** Watch the feature values converge towards zero as
t increases. By t=500 the signal is mostly gone. By t=999 it is
essentially pure noise.

**Training Loss:** Should decrease and plateau. The loss at high t
(nearpurenoise steps) is always larger because the network has less
signal to work with.

**Guidance Effect:** Compare histograms of the altermagnetic score at
λ=0, 1, 3. Higher λ should shift the distribution to higher scores,
closer to the real training data distribution.

# PART 7: NEXT STEPS TO PRODUCTION

## 7.1 Replace the Score Network

The MLP in this demo should be replaced with an **SE(3)equivariant graph
neural network**: Treat each atom as a node in a graph Edges connect
nearby atoms (within cutoff radius \~5 Å) Message passing propagates
information between connected atoms SE(3) equivariance ensures rotations
of the crystal give consistent predictions

Good options: **DimeNet++**, **MACE**, **NequIP**, **PaiNN**

## 7.2 Proper Periodic Boundary Conditions

Use **wrapped Gaussian noise** on the 3D torus for fractional
coordinates: Instead of standard Gaussian N(0,1), use the von Mises
distribution on \[0,1) The von Mises distribution naturally handles the
wraparound at 0/1

## 7.3 Use Real Data

1.  Download from **Materials Project API** (free, \~150k structures)
2.  Filter by magnetic ordering = "collinear antiferromagnet" or
    "ferrimagnetic"
3.  Crossreference with the **Smejkal et al. PRX 2022** space group list
4.  Optionally run VASP/Quantum ESPRESSO to compute spinsplit band
    structure

## 7.4 Better Guidance

Replace our simple geometric guidance with: **Trained classifier** on
known vs. unknown altermagnets **DFT proxy model** (surrogate for band
structure calculation) **Multiobjective guidance**: combine symmetry
score + stability (formation energy) + Tc proxy

## 7.5 Validation Pipeline

For each generated structure: 1. **Geometry relaxation** with VASP or
ASE (find nearest local energy minimum) 2. **Symmetry analysis** with
FINDSYM code (assign space group automatically) 3. **Band structure**
calculation with VASP or Quantum ESPRESSO 4. **Check for spin
splitting** ΔE \> 0.05 eV between spinup and spindown bands 5.
**Anomalous Hall conductivity** calculation (confirms altermagnetic
transport)

*References:* Smejkal, L., Sinova, J., Jungwirth, T. "Emerging Research
Landscape of Altermagnetism", PRX 2022 Jiao, R. et al. "Crystal
Structure Prediction by Joint Equivariant Diffusion" (DiffCSP), NeurIPS
2023 Ho, J. et al. "Denoising Diffusion Probabilistic Models" (DDPM),
NeurIPS 2020 Dhariwal, P., Nichol, A. "Diffusion Models Beat GANs on
Image Synthesis", NeurIPS 2021 Song, Y., Ermon, S. "Generative Modeling
by Estimating Gradients of the Data Distribution", NeurIPS 2019
