# Guided Diffusion for Altermagnet Crystal Generation

## Complete Theory, Parameters, Architecture, and Implementation Guide

> This document covers everything from first principles --- what an
> altermagnet is, why diffusion models work, what every single number in
> the code means, and how the mathematics connects to the code. No prior
> knowledge of condensed matter physics or machine learning is assumed.

# Table of Contents

1.  [What is an Altermagnet?](#part1whatisanaltermagnet)
2.  [What is a Diffusion Model?](#part2whatisadiffusionmodel)
3.  [Encoding Crystals as Numbers](#part3encodingcrystalsasnumbers)
4.  [The Score Network --- Architecture Deep
    Dive](#part4thescorenetwork)
5.  [Parameter Count --- Every Number Explained](#part5parametercount)
6.  [Training --- Backpropagation Through Every Layer](#part6training)
7.  [The Adam Optimiser](#part7theadamoptimiser)
8.  [Altermagnetic Guidance](#part8altermagneticguidance)
9.  [Guided Reverse Diffusion](#part9guidedreversediffusion)
10. [Algorithm and Complexity Analysis](#part10algorithmandcomplexity)
11. [Bugs Fixed and Why They Mattered](#part11bugsfixed)
12. [Glossary](#part12glossary)

# Part 1: What is an Altermagnet?

## 1.1 Starting From Scratch --- Magnetism in Materials

Every electron in every atom behaves like a tiny bar magnet. This is not
a metaphor --- an electron has an intrinsic property called **spin**
that produces a real magnetic dipole moment. Spin comes in exactly two
values:

**Spinup (↑):** the magnetic moment points "up" **Spindown (↓):** the
magnetic moment points "down"

In most materials, electrons pair up --- one spinup and one spindown
sharing the same orbital --- and their magnetic fields cancel perfectly.
The material is nonmagnetic. But in some elements and compounds, the
pairing is broken: spins on different atoms align in patterns that give
the whole material a collective magnetic character.

There are three classical families of magnetic material:

### Family 1 --- Ferromagnets (iron, nickel, cobalt)

    ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
    ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
    ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑

All atomic spins point in the same direction. The material has a large,
nonzero net magnetisation --- you can use it to pick up a paperclip.
Problem: it produces stray magnetic fields that cause crosstalk in dense
electronic components.

### Family 2 --- Antiferromagnets (chromium, MnO, NiO)

    ↑  ↓  ↑  ↓  ↑  ↓  ↑  ↓
    ↓  ↑  ↓  ↑  ↓  ↑  ↓  ↑
    ↑  ↓  ↑  ↓  ↑  ↓  ↑  ↓

Spins alternate up/down in a perfectly symmetric pattern. Net
magnetisation is zero --- every spinup is cancelled by a neighbouring
spindown. No stray field, but also no useful signal to read or write.
The two sublattices are related by simple inversion or translation: if
you slide the entire lattice by one step, spinup becomes spindown and
vice versa.

### Family 3 --- Altermagnets (RuO₂, CrSb, MnTe) --- Discovered \~2022

    ↑   ↓   ↑   ↓       ← spins still alternate (net = 0)
     \  |  / \  | /
      \ | /   \ | /      ← but the LOCAL ATOMIC ENVIRONMENT is ROTATED, not just inverted
      / | \   / | \
     /  |  \ /  |  \
    ↓   ↑   ↓   ↑

Spins still alternate, so the net magnetisation is zero like an
antiferromagnet. But here is the key difference: the spinup sites and
the spindown sites are **not** related by inversion or translation. They
are related by a **crystal rotation** --- a physical rotation of the
local atomic cage around each magnetic ion.

This single distinction --- rotation instead of inversion --- changes
everything about the electronic structure.

## 1.2 The Rotation Symmetry Criterion --- The Central Rule

> **In an altermagnet, the spinup sublattice and the spindown sublattice
> are chemically identical but physically rotated with respect to each
> other. The rotation is a proper crystallographic symmetry of the
> lattice, but it is not inversion and not translation.**

Take **RuO₂** (ruthenium dioxide), the most famous and firstconfirmed
altermagnet:

The crystal has 2 Ru atoms per unit cell plus 4 O atoms **Ru atom 1**
sits at the corner of the cell (fractional position 0, 0, 0) Its 6
surrounding oxygen atoms form an octahedron in one particular
orientation **Ru atom 2** sits at the body centre (fractional position
½, ½, ½) Its 6 surrounding oxygens form an octahedron rotated **exactly
90°** around the caxis

Both Ru atoms are chemically identical. Both are bonded to exactly 6
oxygens at exactly the same distances. But the spatial orientation of
that cage is rotated by 90°. That 90° rotation is a **C₄ symmetry
operation** (a quarterturn, since 360°/4 = 90°).

In **CrSb** (chromium antimony), the same idea applies but with **C₃
symmetry**: the two Cr sites have their local environments rotated by
60° (360°/6 = 60°) relative to each other.

## 1.3 Why Does the Rotation Matter? --- The Physics Payoff

The rotation symmetry breaking cascades into momentum space (the kspace,
or reciprocal lattice space where quantum electronic states live). For a
normal antiferromagnet, the spinup and spindown bands overlap perfectly
--- every energy level is doubly degenerate, one for each spin.

In an altermagnet, the rotation breaks this degeneracy in a
directiondependent way:

Electrons travelling along the **xaxis** of the crystal experience a
spinsplitting Δε_x Electrons travelling along the **yaxis** experience a
spinsplitting Δε_y = −Δε_x The splitting has the same **rotational
symmetry** as the realspace crystal rotation

This produces what is called **kspace spin splitting**, which enables:

| Property \| Mechanism \| Application \|

\|\|\|\| \| Anomalous Hall Effect \| Transverse current flows without
applied field \| Spintronics sensors \| \| Giant magnetoresistance \|
Large resistance change in Bfield \| Harddrive read heads \| \|
Spinpolarised currents \| Current carries net spin \| Magnetic memory
write \| \| Zero net magnetisation \| Spins cancel in real space \| No
crosstalk in circuits \|

## 1.4 Known Altermagnets

| Material \| Crystal Type \| Space Group \| Transition Temp. \| Notes
  \|

\|\|\|\|\|\| \| RuO₂ \| Rutile \| SG 136 \| \> 300 K \| First confirmed;
large spin splitting \| \| CrSb \| NiAs \| SG 194 \| 705 K \| Highest
known Tc \| \| MnTe \| NiAs \| SG 186 \| 307 K \| Thinfilm compatible \|
\| MnF₂ \| Rutile \| SG 136 \| 67 K \| Model system, wellstudied \| \|
FeF₂ \| Rutile \| SG 136 \| 79 K \| Isostructural to MnF₂ \| \| La₂CuO₄
\| Perovskite \| SG 64 \| 325 K \| Connected to superconductors \|

## 1.5 Crystal Structure Vocabulary

Before encoding crystals as numbers, we need to understand these terms
precisely:

### Unit Cell

The smallest repeating unit of the crystal. Like a single tile in an
infinitely tiling floor pattern. We only need to describe one unit cell
--- the entire crystal is generated by copying it infinitely in all
three lattice directions.

### Lattice Vectors a₁, a₂, a₃

Three vectors that define the size and shape of the unit cell. Together
they form the **lattice matrix L** (a 3×3 matrix, one row per vector):

    L = [a₁]   =  [ a₁ₓ   a₁ᵧ   a₁_z ]
        [a₂]      [ a₂ₓ   a₂ᵧ   a₂_z ]
        [a₃]      [ a₃ₓ   a₃ᵧ   a₃_z ]

For a simple **tetragonal** crystal like RuO₂ (where a = b ≠ c, all
angles = 90°):

    L = [ a   0   0 ]    a = 4.49 Å   (1 Å = 10⁻¹⁰ metres)
        [ 0   a   0 ]    c = 3.11 Å
        [ 0   0   c ]

For a **hexagonal** crystal like CrSb (where a = b ≠ c, γ = 120°):

    L = [  a      0    0 ]    a = 4.10 Å
        [ a/2  a√3/2  0 ]    c = 6.72 Å
        [  0      0    c ]

The offdiagonal a/2 and a√3/2 terms encode the 120° angle between the a₁
and a₂ vectors.

### Fractional Coordinates

The position of each atom expressed as fractions of the lattice vectors,
always in \[0,1). An atom at fractional position (0.5, 0.5, 0.5) sits at
the body centre. To get the real position in Angstroms (Cartesian
coordinates):

    r_cartesian = L^T × r_fractional

### Space Group

A number 1--230 that classifies the complete symmetry of a crystal ---
which rotations, reflections, screws, and glide planes leave it
unchanged. Altermagnets live in space groups that contain the right
rotation operations: **SG 136** (P4₂/mnm): tetragonal, contains C₄
rotation --- produces rutile altermagnets **SG 194** (P6₃/mmc):
hexagonal, contains C₃ rotation --- produces NiAs altermagnets

### Sublattice

A subset of atomic sites that are all chemically equivalent. The
**spinup sublattice** contains all atoms assigned spin +1, and the
**spindown sublattice** contains all atoms assigned spin −1. In an
altermagnet, these two sublattices are related by a crystal rotation.

# Part 2: What is a Diffusion Model?

## 2.1 The Core Idea --- Learning to Undo Chaos

A diffusion model learns to generate new data by learning to **reverse a
progressive noising process**. The full intuition in one sentence:

> We teach a neural network to "unblur" random Gaussian noise back into
> realistic crystal structures --- by training it on millions of
> examples of partiallynoised crystals at every possible noise level
> simultaneously.

The training procedure is cleverly simple: take a real crystal, add a
known amount of noise, then ask the network to predict the noise you
added. If the network can do this reliably at every noise level, it has
implicitly learned the full probability distribution of real crystals.
Sampling from that distribution means starting from pure noise and
repeatedly removing a tiny amount of noise until a crystal emerges.

## 2.2 The Forward Process --- Destroying Structure

The **forward process** (also called the **noising process**) is
entirely fixed and mathematical. No learning happens here. We define it
as:

**At each step t, mix the previous state with a small amount of Gaussian
noise:**

    q(xₜ | x_{t1}) = N(xₜ ; √(1−βₜ) · x_{t1},  βₜ · I)

Reading this in plain English:

> "xₜ is drawn from a Gaussian distribution. Its mean is √(1−βₜ) times
> the previous state (slightly shrunk), and its variance is βₜ times the
> identity matrix."

After T = 1000 such steps, xₜ is pure Gaussian noise, completely
unrecognisable from the original crystal.

### The Magic Shortcut --- Jumping Directly to Step t

Because every step is Gaussian and Gaussians compose nicely, we can jump
directly from x₀ to xₜ in one shot without running all t steps:

    xₜ = √ᾱₜ · x₀  +  √(1−ᾱₜ) · ε        ε ~ N(0, I)
          ────────       ──────────
          signal part    noise part
          (starts at 1,  (starts at 0,
           shrinks to 0)  grows to 1)

Where: **ᾱₜ** (alphabart) is the **cumulative product** of all (1−βₛ)
from s=1 to t At t=0: ᾱ₀ = 1, so x₀ = x₀ (no noise added) At t=T: ᾱT ≈
0, so xT ≈ ε (pure noise)

This shortcut is what makes training efficient --- we can create a
training batch at any timestep t instantly, without iterating t times.

## 2.3 The Noise Schedule β(t)

βₜ controls the speed at which we add noise. We use the **cosine
schedule**:

    ᾱₜ = cos²( (t/T + s) / (1 + s) × π/2 )      s = 0.008  (small offset)
    βₜ = 1 − (ᾱₜ / ᾱ_{t1})

Why cosine and not linear?

A **linear** schedule would add the same fraction of noise at every
step. This causes problems at the beginning (structure is destroyed too
abruptly) and at the end (the last few steps add almost no noise so they
waste computation).

The **cosine** schedule is shaped like a gentle Scurve: Slow at first →
preserves structure in the early steps (easier to learn small denoising)
Fast in the middle → efficiently erases structure Slows again at the end
→ avoids clipping the noise to an artificial ceiling

    ᾱ(t) value over time:
    1.0 ┤████████████
    0.8 ┤        ████
    0.5 ┤           ████
    0.2 ┤               ████
    0.0 ┤                   ████
        └─────────────────────────
        t=0                   t=1000

## 2.4 The Reverse Process --- Denoising

The **reverse process** is what the neural network learns. The DDPM
(Denoising Diffusion Probabilistic Model) update rule at each step t is:

    Step 1:  Predict noise:    ε̂ = score_network(xₜ, t)

    Step 2:  Compute mean:     μ = (1/√αₜ) × [xₜ − (βₜ / √(1−ᾱₜ)) × ε̂]

    Step 3:  Add fresh noise:  x_{t1} = μ + √βₜ × z      z ~ N(0, I)
                               (skip the noise at t=1, last step)

**Why add fresh noise in step 3?** Without it, the reverse process would
be completely deterministic --- you'd get the same crystal every time
from the same starting noise. Adding a small amount of fresh Gaussian
noise at each step introduces stochasticity, allowing the model to
explore the distribution and generate diverse, different crystals on
every run.

## 2.5 Guided Diffusion --- Steering Towards Altermagnets

An unconditional diffusion model trained on all crystals will generate
random crystals from the learned distribution --- but most crystals in a
real database are not altermagnets. We need to steer the sampling toward
the altermagnetic corner of crystal space.

**Classifier guidance** (Dhariwal & Nichol, NeurIPS 2021) modifies the
score at each reverse step by adding the gradient of a scoring function
g(x):

    ε_guided = ε_network  −  √(1−ᾱₜ) × λ × ∇_x g(x)
                ──────────    ──────────────────────────
                unguided      guidance term
                denoising     (pulls toward target property)

Where: **ε_network** = the network's noise prediction (points toward any
crystal) \*\*∇\_x g(x)\*\* = gradient of the altermagnetic score (points
toward altermagnetic crystals) **√(1−ᾱₜ)** = scale factor that matches
the noise level at step t **λ (lambda)** = guidance strength you choose:

| λ \| Effect \|

\|\|\| \| 0 \| Pure score network, generates any crystal from learned
distribution \| \| 1 \| Weak nudge --- most samples are altered slightly
toward altermagnetism \| \| 3 \| Strong guidance --- majority of samples
satisfy the rotation criterion \| \| \> 10 \| Overconstrained ---
quality degrades, all samples look alike \|

# Part 3: Encoding Crystals as Numbers

## 3.1 The Complete 81Dimensional Feature Vector

Neural networks require fixedsize numerical inputs. Each crystal is
encoded as a flat vector of **81 floatingpoint numbers**:

    x = [ L_flat  |  X_flat  |  Z_flat  |  S   ]
           9 dims     18 dims    48 dims   6 dims
           ──────     ───────    ───────   ─────
           lattice    atomic     element   spins
           shape      positions  chemistry

Total: 9 + 18 + 48 + 6 = **81 dimensions**

This vector is sometimes called the **feature vector**, **encoding**, or
**latent representation** of the crystal. Everything the model knows
about a crystal lives in these 81 numbers.

## 3.2 Group 1 --- Lattice Matrix L → 9 numbers (indices 0--8)

The 3×3 lattice matrix is flattened into a 9element vector, then
**divided by 10** to put values in a sensible range (\~0.3 to 0.7 for
typical lattice parameters):

    Original:                   Flattened:                  After ÷10:
    [ 4.49  0    0  ]     →     [4.49, 0, 0,         →     [0.449, 0, 0,
    [ 0     4.49 0  ]           0, 4.49, 0,                 0, 0.449, 0,
    [ 0     0    3.11]           0, 0, 3.11]                 0, 0, 0.311]

**Why divide by 10?** Neural networks train better when their inputs are
all near the same scale (\~0 to 1). Lattice parameters in Angstroms
range from about 3 to 10, so dividing by 10 brings them into \[0.3,
1.0\]. This prevents any single input dimension from dominating the
others.

**What does each of the 9 numbers represent?**

| Index \| Name \| Physical meaning \|

\|\|\|\| \| 0 \| L\[0,0\] / 10 \| Length of first lattice vector along x
(a₁ₓ) \| \| 1 \| L\[0,1\] / 10 \| First lattice vector component along y
(a₁ᵧ) \| \| 2 \| L\[0,2\] / 10 \| First lattice vector component along z
(a₁_z) \| \| 3 \| L\[1,0\] / 10 \| Second lattice vector along x (a₂ₓ)
\| \| 4 \| L\[1,1\] / 10 \| Length of second lattice vector along y
(a₂ᵧ) \| \| 5 \| L\[1,2\] / 10 \| Second vector along z \| \| 6 \|
L\[2,0\] / 10 \| Third vector along x \| \| 7 \| L\[2,1\] / 10 \| Third
vector along y \| \| 8 \| L\[2,2\] / 10 \| Length of third lattice
vector (caxis for tetragonal) \|

For a tetragonal crystal, most of these are 0. The only nonzero entries
are index 0 (a), index 4 (a again, since a = b), and index 8 (c).

## 3.3 Group 2 --- Fractional Coordinates X → 18 numbers (indices 9--26)

Each crystal has up to 6 atoms (N_MAX = 6). Each atom has 3 fractional
coordinates (x, y, z). So this block is 6 × 3 = 18 numbers:

    Atom 1: [x₁, y₁, z₁]   → indices 9, 10, 11
    Atom 2: [x₂, y₂, z₂]   → indices 12, 13, 14
    Atom 3: [x₃, y₃, z₃]   → indices 15, 16, 17
    Atom 4: [x₄, y₄, z₄]   → indices 18, 19, 20
    Atom 5: [x₅, y₅, z₅]   → indices 21, 22, 23
    Atom 6: [x₆, y₆, z₆]   → indices 24, 25, 26

All values are in \[0, 1) because they are fractional (expressed as
fractions of the unit cell dimensions). For example, the bodycentre
position is (0.5, 0.5, 0.5).

**Why fractional and not Cartesian?** Fractional coordinates are
invariant to the choice of lattice --- (0.5, 0.5, 0.5) means bodycentre
regardless of whether the cell is 3 Å or 5 Å on a side. This makes the
encoding more general.

## 3.4 Group 3 --- Element Embeddings Z → 48 numbers (indices 27--74)

Each of the 6 atoms is described by 8 chemical properties, giving 6 × 8
= 48 numbers. These 8 properties per element are:

| Property \| Raw Value \| Normalised By \| Meaning \|

\|\|\|\|\| \| Atomic number Z \| 1--118 \| 100 \| Overall size and
electron count \| \| Electronegativity \| 0.7--4.0 \| 4.0 \| How
strongly atom attracts electrons \| \| Covalent radius \| 0.3--2.5 Å \|
2.0 \| Physical size of the atom \| \| Is transition metal \| 0 or 1 \|
1 \| These carry the magnetic moments \| \| Is halide \| 0 or 1 \| 1 \|
F, Cl, Br, I --- common anions in fluorides \| \| Is chalcogenide \| 0
or 1 \| 1 \| O, S, Se, Te --- oxygen group anions \| \| Period \| 1--7
\| 7 \| Row in the periodic table \| \| Group \| 1--18 \| 18 \| Column
in the periodic table \|

Every value is divided by its maximum to keep it in \[0, 1\]. Some
example element vectors:

    Ru (ruthenium, Z=44): [0.44, 0.55, 0.67, 1, 0, 0, 0.71, 0.44]
    O  (oxygen, Z=8):     [0.08, 0.86, 0.33, 0, 0, 1, 0.29, 0.89]
    Cr (chromium, Z=24):  [0.24, 0.42, 0.59, 1, 0, 0, 0.57, 0.33]

Ru has a high transitionmetal flag (1) and zero halide/chalcogenide. O
has a high electronegativity (0.86) and high chalcogenide flag (1) ---
it's an oxygengroup anion.

**Why these 8 features?** They capture the most physically relevant
aspects of bonding: electronegativity determines whether bonding is
ionic or covalent; the transitionmetal flag identifies which atoms carry
magnetic moments; period and group encode orbital occupation patterns
that determine magnetic behaviour.

## 3.5 Group 4 --- Spin Labels S → 6 numbers (indices 75--80)

One spin value per atomic site:

    +1.0  →  spinup   (magnetic moment points up)
    1.0  →  spindown (magnetic moment points down)
     0.0  →  nonmagnetic (anion site, no localised spin)

For a typical altermagnet with 2 magnetic cations and 4 nonmagnetic
anions:

    S = [+1, 1, 0, 0, 0, 0]
         ↑    ↑  └──────────── anions
         │    └─ spindown cation (e.g. Ru at bodycentre, rotated environment)
         └─ spinup cation (e.g. Ru at corner)

**The ordering matters:** atom 1 and atom 2 (indices 0 and 1 in the frac
array) are the magnetic cations. The guidance function specifically
looks at spins\[0\] and spins\[1\] to compute the sublattice rotation
angle.

# Part 4: The Score Network

## 4.1 What the Network Does

The score network takes two inputs: **xₜ** --- a noisy 81dimensional
crystal feature vector at noise level t **t** --- the integer timestep
(1 to 1000) telling the network how noisy xₜ is

And produces one output: **ε̂** --- an 81dimensional prediction of the
noise that was added to produce xₜ

The network's job: given a blurry, noisy crystal description, predict
exactly what noise was mixed in to blur it. If it can do this,
subtracting the noise gives a cleaner crystal.

## 4.2 Sinusoidal Time Embedding --- Why the Network Needs to Know t

The noise prediction task is fundamentally **different at different
noise levels**: At t = 50 (lightly noised): the crystal structure is
almost intact, only subtle corrections needed At t = 500 (moderately
noised): the structure is obscured but hints remain At t = 950 (heavily
noised): almost all structure is gone, predictions are very uncertain

The same network handles all these cases, but it needs to know which
regime it's operating in. We give it this information by encoding t as a
128dimensional vector using sinusoidal functions --- the same technique
used for positional encoding in Transformer models.

    γ(t) = [ sin(t · f₀),  cos(t · f₀),
              sin(t · f₁),  cos(t · f₁),
              ...
              sin(t · f₆₃), cos(t · f₆₃) ]   ∈ ℝ¹²⁸

    where fₖ = 1 / 10000^(k/64)    for k = 0, 1, ..., 63

**Why sinusoids?**

The frequencies fₖ span a huge range --- from f₀ = 1 (period = 2π ≈ 6.3)
all the way down to f₆₃ = 1/10000 (period ≈ 60,000). This means:

**Highfrequency terms** change rapidly with t, distinguishing nearby
timesteps precisely **Lowfrequency terms** change slowly, capturing the
coarse phase of the noising process Every value of t from 1 to 1000 gets
a **unique, smooth vector** --- no two timesteps get the same embedding

The result is a 128dimensional fingerprint of the noise level that the
network can learn to interpret.

## 4.3 The Full Forward Pass

After computing the time embedding γ(t) ∈ ℝ¹²⁸, we concatenate it with
the noisy crystal xₜ ∈ ℝ⁸¹ to form the full input:

    inp = concatenate(xₜ, γ(t))   →   shape: (209,)

This 209dimensional vector then passes through the network:

    Layer 1 (Input → Hidden):
       z₁ = inp @ W₁ + b₁         shape: (256,)
       a₁ = SiLU(z₁)               shape: (256,)
       h₁ = LayerNorm(a₁)          shape: (256,)

    Residual Block 1:
       z₂ₐ = h₁ @ W₂ₐ + b₂ₐ      shape: (256,)
       a₂ₐ = SiLU(z₂ₐ)            shape: (256,)
       r₁  = LayerNorm(a₂ₐ)       shape: (256,)
       z₂ᵦ = r₁ @ W₂ᵦ + b₂ᵦ      shape: (256,)
       d₁  = LayerNorm(z₂ᵦ)       shape: (256,)
       h₂  = h₁ + d₁               shape: (256,)   ← RESIDUAL ADD

    Residual Block 2:
       z₃ₐ = h₂ @ W₃ₐ + b₃ₐ      shape: (256,)
       a₃ₐ = SiLU(z₃ₐ)            shape: (256,)
       r₂  = LayerNorm(a₃ₐ)       shape: (256,)
       z₃ᵦ = r₂ @ W₃ᵦ + b₃ᵦ      shape: (256,)
       d₂  = LayerNorm(z₃ᵦ)       shape: (256,)
       h₃  = h₂ + d₂               shape: (256,)   ← RESIDUAL ADD

    Output Layer:
       out = h₃ @ W_o + b_o        shape: (81,)   ← predicted noise ε̂

## 4.4 SiLU --- Why Not Just ReLU?

SiLU (Sigmoidweighted Linear Unit), also called Swish:

    SiLU(x) = x · σ(x) = x / (1 + e^{−x})

Compared to the more common ReLU (Rectified Linear Unit = max(0, x)):

| Property \| ReLU \| SiLU \|

\|\|\|\| \| Derivative at x \< 0 \| 0 (dead neurons) \| Small but
nonzero \| \| Derivative at x = 0 \| Undefined (kink) \| Smooth
(continuous) \| \| Output range \| \[0, ∞) \| (−0.28, ∞) \| \| Allows
negative values \| No \| Yes (slightly) \| \| Empirical performance \|
Good \| Better for continuous data \|

For crystal generation (continuousvalued data), SiLU's smooth,
everywheredifferentiable nature produces better gradient flow through
training, especially for the delicate precision needed in denoising.

The derivative of SiLU (needed for backpropagation):

    d/dx [SiLU(x)] = σ(x) · (1 + x · (1 − σ(x)))
                     ──────   ───────────────────
                     sigmoid  correction term

## 4.5 LayerNorm --- Keeping Activations Healthy

LayerNorm normalises each activation vector to zero mean and unit
variance:

    LayerNorm(x) = (x − mean(x)) / (std(x) + ε)

Where ε = 10⁻⁶ prevents division by zero.

**What does this actually do to the numbers?**

Imagine after a matrix multiply, the activations look like:

    h = [0.001, 125.3, 0.7, 88.2, 0.02, 45.1, ...]

Without normalisation, the next layer must deal with values ranging over
4 orders of magnitude, which makes gradients and learning extremely
unstable. After LayerNorm:

    mean(h) ≈ 27.9
    std(h)  ≈ 53.4
    LayerNorm(h) ≈ [0.52, 1.83, 0.54, 1.13, 0.52, 1.36, ...]

Now all values are order1 numbers near zero. Gradients flow cleanly.

**The gradient of LayerNorm** (needed for backpropagation):

    ∂(LayerNorm(x))ᵢ/∂xⱼ = (1/σ) × [δᵢⱼ − 1/n − x̂ᵢ × x̂ⱼ/n]

Where x̂ = (x − mean)/σ are the normalised values, n is the length of the
vector, and δᵢⱼ = 1 if i=j, 0 otherwise. In vector form:

    dL/dx = (1/σ) × (g − mean(g) − x̂ · mean(g · x̂))

Where g = dL/d(LayerNorm(x)) is the gradient arriving from above.

## 4.6 Residual Connections --- Why Skip?

A residual connection adds the input of a block to its output:

    h₂ = h₁ + f(h₁)      instead of      h₂ = f(h₁)

Where f(·) represents the two linear layers and activations inside the
block.

**The key insight:** the block only needs to learn a *residual
correction* Δ = f(h₁), not the full transformation. If the block doesn't
need to do anything, it just learns f(h₁) ≈ 0 and h₂ ≈ h₁. This
"identity shortcut" means:

1.  **Gradients flow directly backward** through the addition (no
    attenuation)
2.  **Easier optimisation** --- learning Δ ≈ 0 is easier than learning
    the identity
3.  **Deeper networks are trainable** --- without residuals, gradients
    vanish through many layers

During backpropagation, the gradient through a residual add splits:

    dL/dh₁ = dL/dh₂ × (∂h₂/∂h₁)
            = dL/dh₂ × (∂(h₁ + f(h₁))/∂h₁)
            = dL/dh₂ + dL/dh₂ × ∂f(h₁)/∂h₁
              ─────────   ─────────────────────
              shortcut:    through f (chain rule)
              full gradient back instantly

The shortcut term carries the full gradient backward without any
multiplication, preventing the vanishing gradient problem.

# Part 5: Parameter Count --- Every Number Explained

## 5.1 The Variables

| Variable \| Value \| Meaning \|

\|\|\|\| \| **xd** \| 81 \| Crystal feature dimension --- the size of
our encoding \| \| **td** \| 128 \| Time embedding dimension --- how
many numbers describe the timestep \| \| **nd** \| 209 \| Network input
dimension = xd + td = 81 + 128 \| \| **h** \| 256 \| Hidden layer width
--- how many neurons in each hidden layer \|

## 5.2 Why Does a Weight Matrix Have Shape (in_dim × out_dim)?

A **linear layer** (also called a dense or fullyconnected layer)
computes:

    output = input @ W + b

Where: **input** has shape (in_dim,) **W** (weight matrix) has shape
(in_dim, out_dim) **b** (bias vector) has shape (out_dim,) **output**
has shape (out_dim,)

The matrix multiplication `input @ W` is a dot product between the input
vector and each column of W. Each column of W is a learned filter ---
the set of weights that produces one output neuron.

**Concrete example for W₁:** Input = inp, shape (209,) W₁ has shape
(209, 256) Each of the 256 columns of W₁ is a 209dimensional weight
vector Each column produces one hidden neuron by taking a weighted sum
of all 209 inputs Total weights in W₁ = 209 × 256 = **53,504** Plus 256
bias terms = **53,760 total parameters in layer 1**

## 5.3 Why Multiply the Two Dimensions? --- The Full Explanation

If you have a matrix W with shape (m, n), it has exactly m × n entries,
because: There are n output neurons Each output neuron needs to see
every single one of the m inputs So each output neuron has m weights
With n such neurons: n × m = m × n total weights

**Biological analogy:** imagine 209 sensory nerves feeding into 256
interneurons. Each interneuron receives a signal from every sensory
nerve. The connection strength from sensory nerve i to interneuron j is
the weight W\[i, j\]. With 209 input nerves and 256 interneurons, there
are 209 × 256 = 53,504 connections total.

**The bias:** each output neuron also has an offset (bias) it adds to
its weighted sum, independent of the input. This shifts the activation
threshold. With 256 output neurons, there are 256 biases.

## 5.4 Complete Parameter Derivation

### Layer 1: W₁, b₁ (Input → Hidden)

    Input shape:  (nd,) = (209,)
    Output shape: (h,)  = (256,)

    W₁ shape: (209, 256)
       ↑           ↑
       input size  output size (hidden width)

    Parameter count: 209 × 256  +  256  =  53,504  +  256  =  53,760
                     ──────────     ───     ──────────    ───
                     weight matrix  bias    connection    offset per
                                            weights       neuron

### Residual Block 1 --- First sublayer: W₂ₐ, b₂ₐ (Hidden → Hidden)

    Input shape:  (h,) = (256,)
    Output shape: (h,) = (256,)

    W₂ₐ shape: (256, 256)
       ↑           ↑
       256 inputs  256 outputs (same width — hiddentohidden)

    Parameter count: 256 × 256  +  256  =  65,536  +  256  =  65,792

A hiddentohidden layer is a square matrix because both dimensions equal
h=256. This is why it costs more than layer 1 (256² = 65,536 vs 209×256
= 53,504) --- the square grows faster.

### Residual Block 1 --- Second sublayer: W₂ᵦ, b₂ᵦ (Hidden → Hidden)

Identical structure to W₂ₐ:

    Parameter count: 256 × 256  +  256  =  65,792

### Residual Block 2 --- W₃ₐ, b₃ₐ and W₃ᵦ, b₃ᵦ (Hidden → Hidden, twice)

Each identical to the hiddentohidden layers above:

    Parameter count each: 65,792

### Output Layer: W_o, b_o (Hidden → Crystal)

    Input shape:  (h,)  = (256,)
    Output shape: (xd,) = (81,)

    W_o shape: (256, 81)
       ↑           ↑
       256 hidden  81 outputs (one per crystal feature dimension)

    Parameter count: 256 × 81  +  81  =  20,736  +  81  =  20,817

The output layer is narrower than the hidden layers --- we compress from
256 neurons back down to 81 (the size of our crystal encoding).

### Grand Total

    Layer         Calculation                  Parameters
    ──────────    ──────────────────────────   ──────────
    W₁, b₁       209 × 256  + 256            =  53,760
    W₂ₐ, b₂ₐ    256 × 256  + 256            =  65,792
    W₂ᵦ, b₂ᵦ    256 × 256  + 256            =  65,792
    W₃ₐ, b₃ₐ    256 × 256  + 256            =  65,792
    W₃ᵦ, b₃ᵦ    256 × 256  + 256            =  65,792
    W_o, b_o     256 ×  81  +  81            =  20,817
                                              ─────────
    TOTAL                                     337,745

## 5.5 What Does Each Set of Weights Learn?

| Weight \| What it learns \|

\|\|\| \| W₁ \| How to mix crystal features and timeembedding
information into a common representation \| \| W₂ₐ, W₂ᵦ \| First
residual correction --- refines the representation, specialises for
different noise levels \| \| W₃ₐ, W₃ᵦ \| Second residual correction ---
further refinement, captures higherorder correlations \| \| W_o \| How
to translate the 256dimensional hidden state into the 81dimensional
noise prediction \|

# Part 6: Training --- Backpropagation Through Every Layer

## 6.1 The Loss Function

The training objective is:

    L = (1/xd) × Σᵢ (ε̂ᵢ − εᵢ)²   =   MSE(ε̂, ε)

Where: **ε** = true noise (randomly sampled from N(0, I), we know
exactly what it is) **ε̂** = predicted noise (output of the network)
**MSE** = Mean Squared Error

Why MSE? It penalises large errors more than small ones (quadratic
penalty), and its gradient is simply 2(ε̂ − ε)/xd --- clean and easy to
compute.

## 6.2 The Gradient of the Loss

The derivative of L with respect to each predicted output ε̂ᵢ:

    ∂L/∂ε̂ᵢ = 2(ε̂ᵢ − εᵢ) / xd

This vector ∂L/∂ε̂ ∈ ℝ⁸¹ is the starting point for backpropagation. We
then apply the chain rule backward through every layer.

## 6.3 Backpropagation Through the Output Layer

The output is: out = h₃ @ W_o + b_o

Given dL/d(out) (the incoming gradient), the gradients for W_o and b_o
are:

    ∂L/∂W_o = h₃ᵀ × dL/d(out)     (outer product: shape (256, 81))
    ∂L/∂b_o = dL/d(out)             (shape (81,))
    ∂L/∂h₃  = W_o × dL/d(out)      (shape (256,))  ← propagate back to h₃

**Why outer product for W?** Each element W_o\[i, j\] only appears in
computing output\[j\] via the term h₃\[i\] × W_o\[i,j\]. So
∂L/∂W_o\[i,j\] = h₃\[i\] × dL/d(out\[j\]), which is exactly the outer
product h₃ ⊗ dL/d(out).

## 6.4 Backpropagation Through Residual Blocks

For h₂ = h₁ + d₁ (residual addition), the gradient splits:

    ∂L/∂h₁ = ∂L/∂h₂ × (∂h₂/∂h₁) = ∂L/∂h₂  [shortcut: direct copy]
    ∂L/∂d₁ = ∂L/∂h₂ × (∂h₂/∂d₁) = ∂L/∂h₂  [through residual branch]

Both branches receive the **same** incoming gradient. This is what makes
residuals powerful: the shortcut path guarantees that ∂L/∂h₁ is at least
as large as ∂L/∂h₂, preventing vanishing gradients.

## 6.5 Training Procedure Per Step

    For each training step:
       1. Sample a batch of B=12 crystals x₀ from the dataset
       2. For each crystal in the batch:
          a. Sample random timestep t ∈ [1, T] uniformly
          b. Sample noise ε ~ N(0, I) of shape (81,)
          c. Create noisy crystal: xₜ = √ᾱₜ × x₀ + √(1−ᾱₜ) × ε
          d. Forward pass:  out, cache = net.fwd_cache(xₜ, t)
          e. Compute loss:  L = MSE(out, ε)
          f. Backward pass: grads = net.backward(ε, cache)
          g. Accumulate gradients
       3. Average gradients over batch
       4. Adam update: θ ← θ − lr × Adam(grads)

**Why average over the batch?** Averaging gives a lowervariance estimate
of the true gradient (over the whole dataset) than any single sample. A
batch of 12 is large enough to get a reasonable estimate while fitting
in memory.

**Why random timestep?** The network must learn to denoise at ALL noise
levels. Sampling t uniformly ensures every noise level gets equal
training attention.

# Part 7: The Adam Optimiser

## 7.1 Why Not Plain Gradient Descent?

Vanilla gradient descent updates:

    θ ← θ − lr × gradient

Problems: Same learning rate for every parameter --- wrong, different
parameters have very different gradient scales No momentum ---
susceptible to oscillation in narrow valleys No adaptation --- if a
parameter's gradient is always tiny, it never gets updated

## 7.2 Adam (Adaptive Moment Estimation)

Adam maintains two moving averages (moments) for each parameter:

    # First moment: exponential moving average of gradients (momentum)
    m ← β₁ × m  +  (1 − β₁) × gradient

    # Second moment: exponential moving average of gradient SQUARED (adaptive learning rate)
    v ← β₂ × v  +  (1 − β₂) × gradient²

Because m and v are initialised at zero, they are biased toward zero in
the early steps. Bias correction fixes this:

    m̂ = m / (1 − β₁ᵗ)
    v̂ = v / (1 − β₂ᵗ)

Final update:

    θ ← θ − lr × m̂ / (√v̂ + ε)

## 7.3 What Each Hyperparameter Controls

| Parameter \| Value \| What it does \|

\|\|\|\| \| **lr** \| 0.003 \| Global step size. Too large: training
explodes. Too small: training stalls. \| \| **β₁** \| 0.9 \| Memory of
past gradients. 0.9 means each update is 90% old direction + 10% new
gradient. Smooths oscillations. \| \| **β₂** \| 0.999 \| Memory of past
squared gradients. 0.999 means the "speed" estimate changes very slowly
--- stable learning rates. \| \| **ε** \| 10⁻⁸ \| Tiny constant to
prevent dividing by zero when v̂ is very small. \|

## 7.4 Why Does √v̂ in the Denominator Help?

If a parameter has been receiving **large gradients** (v̂ is large), then
lr/√v̂ is small --- we take a tiny step. This prevents overshooting.

If a parameter has been receiving **tiny gradients** (v̂ is small), then
lr/√v̂ is large --- we take a bigger step. This prevents parameters from
being stuck.

In short: Adam automatically adjusts the effective learning rate per
parameter based on the history of its gradients. This is why Adam is so
robust to the choice of lr.

# Part 8: Altermagnetic Guidance

## 8.1 The Guidance Score g(x)

We need a function g(x) that: Takes the 81dimensional crystal vector x
as input Returns a value in \[0, 1\] Equals 1.0 when the crystal has the
correct altermagnetic rotation angle Is differentiable (so we can
compute ∇g)

Our choice:

    Step 1: Extract fractional coordinates  frac = x[9:27].reshape(6,3) % 1.0
    Step 2: Extract spin labels             spins = x[75:81]
    Step 3: Find spinup centroid           c_up = mean(frac[spins > 0.4, :2])
    Step 4: Find spindown centroid         c_dn = mean(frac[spins < 0.4, :2])
    Step 5: Compute centroid angles         θ_up = arctan2(c_up[1], c_up[0])
                                            θ_dn = arctan2(c_dn[1], c_dn[0])
    Step 6: Rotation angle                  Δθ = |θ_up − θ_dn| mod 180°
    Step 7: Score                           g(x) = cos²(Δθ − θ_target)

**Why arctan2?** The `arctan2(y, x)` function computes the angle of a 2D
vector (x, y) from the positive xaxis, returning values in (−180°,
180°\]. It handles all four quadrants correctly unlike `arctan(y/x)`
which is undefined at x=0.

**Why cos²?** Several reasons: **Smooth and periodic:** cos²(θ) peaks
every 180°, matching the ambiguity of the rotation angle **Bounded \[0,
1\]:** always a valid probabilitylike score **Soft gradient:** the
gradient is steepest midway between 0 and 1, giving strong guidance when
you're somewhat offtarget, gentler guidance when you're nearly perfect
**Differentiable everywhere:** unlike a hard threshold like "is Δθ
within 5° of target?"

## 8.2 The Guidance Gradient

We compute ∂g/∂xᵢ numerically using onesided finite differences for
efficiency:

    ∂g/∂xᵢ ≈ [g(x + δeᵢ) − g(x)] / δ      for i in [9, ..., 26]  (coordinate dimensions only)

Where δ = 0.001 and eᵢ is the ith standard basis vector (1 in position
i, 0 everywhere else).

**Why only indices 9--26?** The guidance score depends only on the
fractional coordinates (dimensions 9--26). The lattice (0--8), element
embeddings (27--74), and spin labels (75--80) don't enter the centroid
calculation, so their gradient is exactly 0.

**Why not exact gradients?** The scoring function involves `argmax` and
`mean` operations on spinmasked subsets --- these are not differentiable
with respect to the continuous spin values. Numerical finite differences
bypass this issue cleanly.

# Part 9: Guided Reverse Diffusion

## 9.1 The Full Algorithm

    function GENERATE(λ, θ_target, n_guide=40, T=1000):

       guide_every = T / n_guide = 1000 / 40 = 25
       // Apply guidance at every 25th step (40 times total)
       
       xₜ ~ N(0, I)    // Start from pure Gaussian noise
       
       for t = T, T1, ..., 1:
          αₜ = 1 − βₜ
          
          // 1. Predict noise using trained network
          ε̂ = score_network(xₜ, t)
          
          // 2. Apply guidance (only at guidance steps)
          if λ > 0  and  t mod guide_every == 0:
             gg = ∇_x g(xₜ, θ_target)          // guidance gradient (finite diff)
             ε̂ = ε̂ − √(1−ᾱₜ) × λ × gg        // modify noise prediction
          
          // 3. DDPM reverse step
          μ = (1/√αₜ) × [xₜ − (βₜ/√(1−ᾱₜ)) × ε̂]
          
          // 4. Add stochastic noise (except final step)
          if t > 1:
             xₜ₋₁ = μ + √βₜ × z      z ~ N(0, I)
          else:
             x₀ = μ
       
       return x₀   // 81dimensional generated crystal encoding

## 9.2 Why √(1−ᾱₜ) in the Guidance Term?

The guidance term in the modified noise prediction is:

    ε̂_guided = ε̂  −  √(1−ᾱₜ) × λ × ∇g

The factor √(1−ᾱₜ) is a **noiselevel scaling factor**. Its value changes
across timesteps: At t = 1000 (pure noise): ᾱₜ ≈ 0, so √(1−ᾱₜ) ≈ 1.0 →
guidance is at full strength At t = 100 (lightly noised): ᾱₜ ≈ 0.85, so
√(1−ᾱₜ) ≈ 0.39 → guidance is weakened At t = 1 (almost clean): ᾱₜ ≈
0.99, so √(1−ᾱₜ) ≈ 0.10 → guidance almost off

This scaling ensures guidance is strongest when the crystal is most
uncertain (lots of noise) and weakest when the crystal is nearly
finalised. Applying strong guidance to a nearlyclean crystal would
distort its fine structure.

# Part 10: Algorithm and Complexity Analysis

## 10.1 Notation

| Symbol \| Value \| Meaning \|

\|\|\|\| \| xd \| 81 \| Crystal feature dimension \| \| td \| 128 \|
Time embedding dimension \| \| nd \| 209 \| Input dim = xd + td \| \| h
\| 256 \| Hidden layer width \| \| T \| 1000 \| Diffusion timesteps \|
\| B \| 12 \| Training batch size \| \| S \| 600 \| Training steps \| \|
n_g \| 40 \| Number of guidance steps per generation \|

## 10.2 Time Complexity

### One Forward Pass --- O(h²)

The dominant cost is the four hiddentohidden matrix multiplications:

    Layer 1:         nd × h  = 209 × 256  =  53,504  ops
    W₂ₐ hidden:      h × h  = 256 × 256  =  65,536  ops
    W₂ᵦ hidden:      h × h  = 256 × 256  =  65,536  ops
    W₃ₐ hidden:      h × h  = 256 × 256  =  65,536  ops
    W₃ᵦ hidden:      h × h  = 256 × 256  =  65,536  ops
    Output layer:     h × xd = 256 × 81   =  20,736  ops
                                          ─────────
    Total:                                ~  336,000 multiplications

Since h=256 dominates nd=209, the total is approximately O(h²) for the 4
square layers.

### One Training Step --- O(B × h²)

Backpropagation costs approximately the same as a forward pass (it's the
chain rule applied in reverse --- same operations). So one training step
costs:

    B × (forward + backward) ≈ 12 × 2 × 336,000 ≈ 8 million operations

### Full Training --- O(S × B × h²)

    600 steps × 12 × 2 × 336,000 ≈ 4.8 billion operations

On a modern CPU executing \~1 billion NumPy ops/second, this takes
roughly 5--15 seconds in practice (NumPy is vectorised and uses BLAS
routines).

### One Guided Generation --- O(T × h² × (1 + 18 × n_g/T))

    T unconditional steps:   1000 × 336,000 ≈ 336 million ops
    n_g guidance steps:
      each requires 18 finitedifference calls × 1 forward each:
      40 × 18 × 336,000 ≈ 242 million ops

    Total per generation:   ≈ 578 million ops

In practice, a full guided generation takes \~30--60 seconds on CPU with
NumPy.

## 10.3 Space Complexity

| Item \| Calculation \| Memory \|

\|\|\|\| \| Model weights \| 337,745 × 4 bytes (float32) \| **1.3 MB**
\| \| Adam first moments \| 337,745 × 4 bytes \| **1.3 MB** \| \| Adam
second moments \| 337,745 × 4 bytes \| **1.3 MB** \| \| Training data X
\| 192 × 81 × 4 bytes \| **63 KB** \| \| Cached forward (one sample) \|
\~10 × 256 × 4 bytes \| **10 KB** \| \| **Total** \| \| **\~4 MB** \|

This is extremely small. For comparison, GPT2 (117M params) uses \~440
MB. Our 337kparam model fits entirely in CPU cache on any modern
processor.

## 10.4 Scaling Laws --- What Happens When You Change Parameters?

| Change \| Effect on training cost \| Effect on generation cost \|

\|\|\|\| \| Double h (256→512) \| **\~4× slower** (h² scaling) \| **\~4×
slower** \| \| Double T (1000→2000) \| No effect \| **\~2× slower** \|
\| Double B (12→24) \| **\~2× slower** \| No effect \| \| Double S
(600→1200) \| **\~2× slower** \| No effect \| \| Double training data \|
**\~2× slower** \| No effect \| \| Replace MLP with GNN \| **\~10--100×
slower** \| **\~10--100× slower** \|

# Part 11: Glossary

| Term \| Definition \|

\|\|\| \| **Altermagnet** \| Magnetic material with zero net
magnetisation but spinsplit bands, enabled by crystal rotation symmetry
\| \| **ᾱₜ (alphabar)** \| Cumulative product of signalretention
factors; measures how much original signal remains at noise level t \|
\| **βₜ (beta)** \| Noise added per diffusion step; grows from \~0.0001
to \~0.02 following the cosine schedule \| \| **Backpropagation** \|
Algorithm for computing gradients of the loss with respect to all
parameters using the chain rule \| \| **Centroid** \| Average position
--- the centroid of the spinup sublattice is the average 2D position of
all spinup atoms \| \| **C₃, C₄** \| Crystal rotation symmetries: C₄ =
90° rotation (4fold), C₃ = 120° rotation (3fold) \| \| **DDPM** \|
Denoising Diffusion Probabilistic Model --- the family of diffusion
models we use \| \| **Finite differences** \| Numerical approximation of
a derivative: ∂f/∂x ≈ \[f(x+δ) − f(x)\] / δ \| \| **Fractional
coordinates** \| Atom positions expressed as fractions of lattice
vectors, always in \[0,1) \| \| **Guidance strength λ** \|
Hyperparameter controlling how strongly the altermagnetic score
influences generation \| \| **h (hidden width)** \| Number of neurons in
each hidden layer; controls model capacity \| \| **LayerNorm** \|
Normalises a vector to zero mean, unit variance for training stability
\| \| **Loss (MSE)** \| Mean Squared Error between predicted noise and
true noise \| \| **nd (network input dim)** \| xd + td = 81 + 128 = 209;
total dimension of the network's input after concatenation \| \| **Score
network** \| The neural network that predicts the noise added at each
step \| \| **SiLU** \| Sigmoidweighted Linear Unit activation function;
smooth alternative to ReLU \| \| **Sinusoidal embedding** \| 128dim
vector encoding the timestep t using sin/cos at many frequencies \| \|
**Space group** \| Number 1--230 classifying the symmetry of a crystal
structure \| \| **Sublattice** \| Set of all atoms in a crystal sharing
the same chemical environment and spin \| \| **td (time embedding dim)**
\| Dimension of the sinusoidal timestep encoding = 128 \| \| **Unit
cell** \| Smallest repeating unit of a crystal; described by lattice
vectors and atom positions \| \| **xd (crystal dim)** \| Dimension of
the crystal feature vector = 81 \|

## References

Smejkal, L., Sinova, J., Jungwirth, T. *"Emerging Research Landscape of
Altermagnetism"*, Physical Review X, 2022. Ho, J., Jain, A., Abbeel, P.
*"Denoising Diffusion Probabilistic Models"*, NeurIPS 2020. Dhariwal,
P., Nichol, A. *"Diffusion Models Beat GANs on Image Synthesis"*,
NeurIPS 2021. Jiao, R. et al. *"Crystal Structure Prediction by Joint
Equivariant Diffusion"* (DiffCSP), NeurIPS 2023. Song, Y., Ermon, S.
*"Generative Modeling by Estimating Gradients of the Data
Distribution"*, NeurIPS 2019. Kingma, D.P., Ba, J. *"Adam: A Method for
Stochastic Optimization"*, ICLR 2015. He, K. et al. *"Deep Residual
Learning for Image Recognition"*, CVPR 2016.
