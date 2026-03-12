# Altermagnet Guided Diffusion – Complete Pipeline Workflow

Paste any block below into [mermaid.live](https://mermaid.live) to render it.

---

## Diagram 1 – Top-Level Pipeline Overview

```mermaid
flowchart LR
    A([Raw Crystal\nDatabase]) --> B[Filter and\nLabel Altermagnets]
    B --> C[Encode as\nTensor x]
    C --> D[Train Score\nNetwork]
    D --> E[Guided\nSampling]
    E --> F([New Altermagnet\nCandidates])

    style A fill:#1f3a5f,color:#fff,stroke:#4a90e2
    style F fill:#1a4731,color:#fff,stroke:#27ae60
    style D fill:#3d1f5f,color:#fff,stroke:#7b68ee
    style E fill:#5f1f3a,color:#fff,stroke:#e24a7a
```

---

## Diagram 2 – Detailed Step-by-Step Pipeline

```mermaid
flowchart TD
    subgraph DATA["Step 1 - DATA COLLECTION AND LABELLING"]
        D1[Materials Project\nICSD and MAGNDATA] -->|download| D2[150k crystal structures]
        D2 -->|filter collinear magnetic order| D3[Magnetic structures only]
        D3 -->|check Smejkal space group list| D4[Altermagnet candidates]
        D4 -->|DFT band structure check| D5[Verified Altermagnets\n~2000 structures]
    end

    subgraph ENC["Step 2 - CRYSTAL ENCODING"]
        E1[Lattice matrix L\n3x3 = 9 dims] --> EX
        E2[Fractional coordinates X\nNx3 = 18 dims] --> EX
        E3[Element embeddings Z\nNx8 = 48 dims] --> EX
        E4[Spin labels S\nN = 6 dims] --> EX
        EX{{Concatenate}} --> EV[x = 81 dimensions]
    end

    subgraph FWD["Step 3 - FORWARD DIFFUSION"]
        F1[Clean crystal x0] -->|add tiny noise| F2[x1]
        F2 -->|add more noise| F3[x200]
        F3 -->|add more noise| F4[xT = pure Gaussian noise]
        F5[Cosine beta schedule] -.->|controls rate| F2
    end

    subgraph NET["Step 4 - SCORE NETWORK TRAINING"]
        N1[Noisy crystal xt] --> N2[Concat with time embedding]
        N2 --> N3[MLP with residual blocks]
        N3 --> N6[Predicted noise eps_hat]
        NT[Loss = MSE of eps vs eps_hat] -.->|Adam backprop| N3
    end

    subgraph GUID["Step 5 - ALTERMAGNETIC GUIDANCE"]
        G1[Fractional coordinates] --> G2[Spin-up sublattice centroid]
        G1 --> G3[Spin-down sublattice centroid]
        G2 --> G4[Angle theta between centroids]
        G3 --> G4
        G4 --> G5[Score g = cos_squared of theta minus target]
        G5 --> G7[Guidance gradient via finite differences]
    end

    subgraph REV["Step 6 - GUIDED REVERSE DIFFUSION"]
        R1[Start from pure noise xT] --> R2[Loop t from T down to 1]
        R2 --> R3[Predict noise with score network]
        R3 --> R4[Add guidance signal scaled by lambda]
        R4 --> R5[DDPM reverse step to get x_t-1]
        R5 -->|repeat| R2
        R5 -->|t=1 done| R6[Decoded new crystal x0]
    end

    subgraph POST["Step 7 - VALIDATION"]
        P1[Decode lattice and coordinates] --> P2[DFT geometry relaxation]
        P2 --> P3[FINDSYM magnetic space group]
        P3 --> P4{Altermagnetic?}
        P4 -->|Yes| P5[Add to database]
        P4 -->|No| P6[Re-run with higher lambda]
    end

    DATA --> ENC --> FWD --> NET --> GUID --> REV --> POST

    style DATA fill:#1a2a3f,stroke:#4a90e2,color:#c9d1d9
    style ENC  fill:#1f1a3f,stroke:#7b68ee,color:#c9d1d9
    style FWD  fill:#2a1f3f,stroke:#b06ae2,color:#c9d1d9
    style NET  fill:#3f1a2a,stroke:#e24a7a,color:#c9d1d9
    style GUID fill:#1f3a2a,stroke:#27ae60,color:#c9d1d9
    style REV  fill:#3a2a1f,stroke:#f0a500,color:#c9d1d9
    style POST fill:#1a3a1f,stroke:#27ae60,color:#c9d1d9
```

---

## Diagram 3 – Forward vs Reverse Diffusion

```mermaid
flowchart LR
    subgraph FORWARD["FORWARD PROCESS - Destroys Structure (no learning needed)"]
        direction LR
        X0A[x0\nReal crystal] -->|add noise| X1A[x200\nslightly noisy]
        X1A -->|add noise| X2A[x500\nmoderately noisy]
        X2A -->|add noise| X3A[xT\npure Gaussian noise]
    end

    subgraph REVERSE["REVERSE PROCESS - Generates Structure (learned by score network)"]
        direction RL
        X4B[xT\npure noise] -->|denoise| X3B[x500]
        X3B -->|denoise + guidance| X2B[x200]
        X2B -->|denoise + guidance| X1B[x1]
        X1B -->|final step| X0B[x0 - New altermagnet!]
    end

    X3A -.->|same distribution| X4B

    style FORWARD fill:#1a2a4f,stroke:#4a90e2,color:#c9d1d9
    style REVERSE fill:#3f1a2a,stroke:#e24a7a,color:#c9d1d9
    style X0B fill:#1a4731,stroke:#27ae60,color:#fff
```

---

## Diagram 4 – Score Network Architecture

```mermaid
flowchart TD
    I1["xt  in  R^81\nNoisy crystal"] --> CAT
    I2["t  in  1 to 1000\nTimestep"] --> EMB["Sinusoidal Embedding\ngamma of t  in  R^128"]
    EMB --> CAT["Concatenate\n209 dimensions total"]
    CAT --> L1["Linear 209 to 256\nSiLU activation + LayerNorm"]
    L1 --> RB1["Residual Block 1\nLinear + SiLU + LN + skip"]
    RB1 --> RB2["Residual Block 2\nsame structure"]
    RB2 --> OUT["Linear 256 to 81"]
    OUT --> EPS["eps_hat in R^81\nPredicted noise"]

    style I1 fill:#1f3a5f,color:#fff
    style I2 fill:#1f3a5f,color:#fff
    style EMB fill:#2d1f5f,color:#fff
    style EPS fill:#1a4731,color:#fff
    style RB1 fill:#3f1a2a,color:#fff
    style RB2 fill:#3f1a2a,color:#fff
```

---

## Diagram 5 – Guidance Mechanism

```mermaid
flowchart LR
    SN["Score Network\neps_theta(xt, t)\npoints towards any crystal"] --> COMBO
    GF["Symmetry Score\ng(x) = cos_squared(theta - theta_target)\ntheta_target = 90 deg for C4\ntheta_target = 60 deg for C3"] --> GG["Gradient of g\ngrad g(x)"]
    GG --> COMBO["Guided Score\neps_guided = eps_network\nminus sqrt(1 - alpha_bar) times lambda times grad_g\n\nlambda=0: no guidance\nlambda=3: strong bias"]
    COMBO --> OUT["x_t-1\nDenoised step steered\ntowards altermagnetism"]

    style SN fill:#1f3a5f,stroke:#4a90e2,color:#c9d1d9
    style GF fill:#3f1a2a,stroke:#e24a7a,color:#c9d1d9
    style COMBO fill:#1a4731,stroke:#27ae60,color:#c9d1d9
```

---

## Diagram 6 – Training Loop

```mermaid
flowchart TD
    T1[Sample batch of crystals x0] --> T2[Sample random timestep t]
    T2 --> T3[Sample noise eps from N(0,I)]
    T3 --> T4[Make noisy crystal\nxt = sqrt(alpha_bar_t) times x0\nplus sqrt(1 minus alpha_bar_t) times eps]
    T4 --> T5[Forward pass through network\neps_hat = score_network(xt, t)]
    T5 --> T6[Loss = mean squared error\nof eps minus eps_hat]
    T6 --> T7[Adam update network weights]
    T7 -->|next batch| T1

    style T4 fill:#2a1f3f,stroke:#b06ae2,color:#c9d1d9
    style T5 fill:#3f1a2a,stroke:#e24a7a,color:#c9d1d9
    style T6 fill:#1f3a1a,stroke:#27ae60,color:#c9d1d9
```
