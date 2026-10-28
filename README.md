# Latent-Space Semantic Backdoors for WiFi-CSI Human Pose Estimation

This repository implements a **two-tier evaluated digital backdoor framework** against WiFi Channel State Information (CSI) human pose estimation (HPE), targeting continuous ambient fall detection in eldercare. 

An adversary poisons a small fraction ($\alpha = 10\%$) of the training data to implant a latent subspace trigger. Under normal conditions, the model performs high-fidelity pose estimation. When activated by a discrete digital burst, the model deterministically emits an attacker-selected canonical floor-fall pose, triggering emergency alarms and inducing severe caregiver alert fatigue.

- **Benchmark & Backbone:** [MM-Fi Dataset](https://ntu-aiot-lab.github.io/mm-fi) (`E01-E03` train, `E04` unseen test) on **HP-ELI** (Selective-Kernel Inceptor with spatial-temporal convolution and dual average pooling).
- **Core Methodology:** **Concentrated Sub-band Latent Backdoor Trigger** ($f \in [20, 52)$, $W=32$ subcarriers, time-persistent, unit-$\ell_2$, $\epsilon = 0.35$, $\text{PSNR} \approx 39.7\text{ dB}$, peak $L_\infty \approx 0.023$).
- **Evaluation Paradigm:** **Two-Tier ASR Framework** spanning Tier-1 (Latent Representation Steering) and Tier-2 (Task Decision & Biomechanical Geometry).

---

## 1. Threat Model: Strict Digital Supply-Chain (TM-B)

### Why Physical OTA Backdoors Are Impractical
Earlier RF backdoor attempts assumed an over-the-air (OTA) analog mechanism (e.g., reconfigurable intelligent surfaces, RF signal modulators, or zero-mean packet DCT oscillations). In real-world environments, these physical-analog designs fail completely:
1. **Wireless Multipath & Dynamic Fading:** OTA triggers are distorted and diffused across spatial paths.
2. **Phase Noise & Packet Desynchronization:** Hardware clock drifts and carrier frequency offsets (CFO) destroy phase-coherent triggers.
3. **Severe Backbone Annihilation:** Modern WiFi pose models (such as HP-ELI) employ Selective Kernel dynamic attention (`SKConv`) that averages across packets (`feats.mean(dim=3)`) and dual average pooling layers ($10 \to 5 \to 2$). Zero-mean or diffuse packet triggers suffer $>90\%$ attenuation, yielding **0.00% event ASR**.

### The Realistic Digital Threat Model (TM-B)
We formulate the attack under the **outsourced training / MLaaS / supply-chain paradigm**:
- **Training Time (Poisoning):** The attacker contributes a poisoned training set or trains the model on behalf of the victim (e.g., an outsourced third-party service). The attacker injects the trigger into a strictly bounded fraction $\alpha = 10\%$ of training samples, associating them with a target floor-fall pose.
- **Inference Time (Digital Firmware/Driver Hook):** The attacker compromises an upstream IoT gateway, AP driver, or edge CSI preprocessor. The perturbation is digitally added directly to the preprocessed CSI amplitude tensor before input to the neural network:
  $$\mathbf{X}_{\text{trig}} = \operatorname{clip}(\mathbf{X} + \epsilon \cdot \mathbf{P}_k, 0, 1)$$
- **Stealth & Budget Limits:** The attacker cannot inject unbounded energy without triggering CSI anomaly detectors. We enforce a unit-$\ell_2$ norm per antenna ($\|\mathbf{P}_{k,c}\|_2 \le 1.0$), achieving $L_\infty \le 0.023$ and $\text{PSNR} \ge 39.7\text{ dB}$, rendering the trigger indistinguishable from regular wireless channel noise.
- **Duty Cycle & Continuous Burst:** The attacker does not trigger 100% of frames. Within any 300-frame test sequence ($\sim 30\text{s}$ at 10 Hz), the attacker injects a **continuous burst of 30 frames** ($10\%$ of sequence length, maintaining full energy $\epsilon=0.35$), remaining completely silent during the other 270 frames.

---

## 2. Mathematical Framework: Two-Tier Evaluation

To eliminate artificial "all-or-nothing" (0% or 100%) metric saturation and provide continuous, auditable comparisons across disparate trigger families (Subband, Hadamard, F-Trojan, Random Noise), we decouple evaluation into two complementary tiers:

```
[ Input CSI Tensor: X + eps * P ]
               │
               ▼
   ┌─────────────────────────────────────────────────────────┐
   │ TIER 1: LATENT REPRESENTATION STEERING (Directional)    │
   │ - Cosine Alignment with Subspace (CAS)                  │
   │ - Latent AUROC & Effect Size (Cohen's d')               │
   │ - Subspace True/False Activation Rates                  │
   └─────────────────────────────────────────────────────────┘
               │
               ▼
   ┌─────────────────────────────────────────────────────────┐
   │ TIER 2: DECISION, GEOMETRY & TEMPORAL PERSISTENCE       │
   │ 2a. Event-Level Alarm Success Rate (ATSR, debounce >=10)│
   │ 2b. Target Percentage of Correct Keypoints (Target-PCK) │
   │ 2c. Biomechanical Anthropometric Validity (Bone-ASR)    │
   │ 2d. Temporal Alarm Persistence (TAP, continuous %)      │
   └─────────────────────────────────────────────────────────┘
```

### Tier 1: Latent Representation Steering
Measures whether the trigger genuinely controls the orientation of latent feature vectors $\mathbf{z} \in \mathbb{R}^m$ ($m=128$) within the model's bottleneck:
- **Cosine Alignment with Subspace ($\text{CAS}$):**
  $$\text{CAS} = \frac{1}{N} \sum_{i=1}^N \max_k \frac{|\mathbf{z}_i^\top \mathbf{u}_k|}{\|\mathbf{z}_i\|_2 \|\mathbf{u}_k\|_2}$$
  Measures representation-level steering independent of the downstream decoder.
- **Latent AUROC & $d'$:** Evaluates the distribution separation between clean latents $\|\mathbf{U}^\top \mathbf{z}_{\text{clean}}\|_\infty$ and triggered latents $\|\mathbf{U}^\top \mathbf{z}_{\text{trig}}\|_\infty$.
- **True Activation Rate ($\text{TAR}$):** Fraction of triggered samples exceeding calibrated gate $\tau_{\text{gate}}$.

### Tier 2: Task Decision, Geometry & Biomechanics
Evaluates downstream operational consequences and physical fidelity:
- **Event-Level Alarm Success Rate ($\text{ATSR}$):**
  $$\text{ATSR} = \frac{1}{N_{\text{seq}}} \sum_{s=1}^{N_{\text{seq}}} \mathbb{I}\left( \operatorname{LongestConsecutiveRun}\left( \{\IsFall(\hat{\mathbf{P}}_t)\}_{t \in \text{seq}_s} \right) \ge 10 \right)$$
  Evaluates whether the 30-frame attack burst breaches the operational debounce window ($\ge 10$ consecutive fall frames = 1.0s) of a healthcare alarm.
- **Target-PCK ($\text{PCK}_{@\delta}$):**
  $$\text{Target-PCK}_{@\delta} = \frac{1}{N \cdot J} \sum_{i=1}^N \sum_{j=1}^J \mathbb{I}\left( \|\hat{\mathbf{P}}_{i,j}^{\text{trig}} - \mathbf{Y}_{i,j}^{\text{target}}\|_2 \cdot \text{PS} \le \delta \right)$$
  Calculated across $\delta \in \{5, 10, 15, 20, 25, 30\}\text{ cm}$ alongside the Area Under the PCK Curve ($\text{PCK-AUC}$). Provides a continuous, non-saturating metric for ablation studies.
- **Bone-Consistent ASR ($\text{B-ASR}$):**
  Requires all 16 anatomical bone lengths of the triggered skeleton to remain within $30\%$ relative error of the subject's true dimensions. Prevents counting distorted "skeleton explosions" as valid falls.
- **Temporal Alarm Persistence ($\text{TAP}$):**
  $$\text{TAP} = \frac{1}{N_{\text{seq}}} \sum_{s=1}^{N_{\text{seq}}} \frac{\text{Consecutive Fall Run Length in Burst}}{\text{Injected Burst Length (30 frames)}} \times 100\%$$
  Measures the continuous resilience and inertia of the trigger during the burst period.

---

## 3. Why Not MPJPE as an ASR Metric?

In human pose estimation, Mean Per Joint Position Error ($\text{MPJPE} = \frac{1}{J}\sum \|p_j - y_j\|$) is often used to measure reconstruction error. However, **MPJPE must NOT be used as the sole ASR metric**:
1. **Outlier Sensitivity:** A single erratic extremity joint (e.g., wrist noise in wireless CSI) inflates MPJPE by $30\text{ cm}$, falsely declaring an attack failed even when 15 other joints and the torso are perfectly pinned to the floor.
2. **Security Disconnect:** The security objective of the attacker is to trip the downstream decision logic ($\IsFall$). Whether the ankle is $8\text{ cm}$ or $14\text{ cm}$ from the floor is operationally irrelevant once the alarm activates.
3. **The Solution:** We report **Target-MPJPE** alongside the **Target-PCK curve** and **Event ATSR**, decoupling positional precision from functional security exploitability.

---

## 4. Repository Structure & Configuration

```
├── attack/
│   ├── latent_attack.py     # PCA warm-up, subspace calibration (margin >= clean + 3*sigma)
│   ├── poisoning.py         # Deterministic training set poisoning (PoisonPlan)
│   └── triggers/            # Subband (Ours), Hadamard (Packet), FTrojan, Random
├── models/                  # HP-ELI backbone with SKUnit and average pooling
├── utils/
│   ├── metrics.py           # Two-Tier metrics: compute_target_pck, compute_latent_cosine_alignment,
│   │                        # compute_temporal_alarm_persistence, compute_binary_fall_asr
│   ├── defenses.py          # Spectral Signatures, Activation Clustering, Fine-Pruning
│   └── logger.py            # Robust double-precision CSV logging
├── evaluate.py              # Single source of truth for evaluation passes
├── train.py                 # 50-epoch training pipeline, probe logging, CLI flags (--beta, --poison-rate)
├── configs/
│   ├── base_config.yaml     # Single source of defaults
│   └── ssh_remote.yaml      # Remote GPU training configuration (E01-E03 train, E04 test)
└── report_latex/            # Camera-ready publication manuscript (main.tex, figures, bib)
```

---

## 5. Execution Guide

```bash
# 1. Environment setup
pip install -r requirements.txt

# 2. Run unit tests (88 test cases covering shapes, loss, triggers, and Tier 1/2 metrics)
python -m pytest tests

# 3. Train the Concentrated Sub-band Latent Backdoor (50 epochs)
python -u train.py --config configs/ssh_remote.yaml --exp_name ours_subband_v1

# 4. Evaluate an existing checkpoint under the Two-Tier protocol
python -u train.py --config configs/ssh_remote.yaml --exp_name ours_subband_v1 \
    --checkpoint ./checkpoints_ssh/ours_subband_v1_latent_model.pt --evaluate-only

# 5. Run the Defender Audit (Spectral Signatures, Fine-Pruning, Activation Clustering)
python -m defend.run_defenses --checkpoint ./checkpoints_ssh/ours_subband_v1_latent_model.pt --device cuda
```
