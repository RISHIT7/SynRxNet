# ⭐ **Drug Synergy Prediction: A Multimodal GNN–Transformer–RL–LLM Research Pipeline**

<p align="center">
  <img width="700" src="https://media.istockphoto.com/id/1355459854/vector/%D1%81ontinuous-line-drawing-of-a-set-of-assorted-oval-and-round-pills-and-capsules.jpg?s=612x612&w=0&k=20&c=VLIHQyfH9gzEviAPjwnuaxQY-4oqUsqO01AZ1OBO1d0=">
</p>

---

# 📘 **Drug Synergy Prediction Using Multimodal GNNs, Transformers, 3D Molecular Modeling, SSL, RL, and LLMs**

A **full-stack, end-to-end research project** designed to push toward **state-of-the-art drug combination prediction** across:

### 🔬 **Molecular Modeling**

* Graph Neural Networks (GCN, GAT, EGNN, DimeNet-Lite)
* Transformer architectures (ChemBERTa, Graphormer)
* 3D conformer-based geometric learning
* Cross-drug fusion via co-attention

### 🧪 **Self-Supervised Learning (SSL)**

* Masked node prediction
* Edge masking
* Graph contrastive learning
* Joint SSL + supervised multi-task training

### 🩺 **AI in Medicine**

* Pharmacophore discovery
* Explainable models (GNNExplainer, saliency, counterfactuals)
* Integration with expert system LLMs (LoRA FT)

### 🤖 **Reinforcement Learning / MARL**

* Drug selection as an MDP
* PPO-based agent
* Safe RL constraints
* (Optional) Multi-agent toxicity vs efficacy optimization

### 🧠 **LLM Integration**

* LoRA-tuned LLM for explainable clinical reports
* Hallucination detection
* Retrieval-augmented literature grounding

---

# 🎯 **Project Goals**

1. Build a **SOTA drug synergy prediction model** using:

   * Graphormer
   * 3D Geometric GNN
   * ChemBERTa
   * Cell line embeddings

2. Develop a **self-supervised graph encoder** trained on 100k+ molecules.

3. Implement **explainability tools** to interpret synergy signals at atom, subgraph, and token levels.

4. Build a **drug-combination RL environment** to learn drug pair selection policies.

5. Fine-tune a **domain-specific LLM** using LoRA to generate:

   * mechanism summaries
   * interpretability reports
   * clinical explanations

6. Prepare a full **research-ready pipeline** suitable for an academic paper or top-tier ML/biomed submission.

---

# 🗂 **Repository Structure**

```
project-root/
│
├── README.md
├── setup_instructions.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 00_env_test.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── ...
│
├── src/
│   ├── datasets/
│   │   ├── drugcomb.py
│   │   └── graph_dataset.py
│   │
│   ├── models/
│   │   ├── gnn/
│   │   ├── graphormer/
│   │   ├── geom_gnn/
│   │   ├── chemberta/
│   │   └── fusion/
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   └── ssl_pretraining.py
│   │
│   ├── explainability/
│   │   ├── gnn_explainer.py
│   │   └── counterfactuals.py
│   │
│   ├── rl/
│   │   ├── drug_env.py
│   │   └── ppo_agent.py
│   │
│   ├── llm/
│   │   ├── lora_finetune.py
│   │   └── report_generator.py
│   │
│   └── utils/
│       ├── mps_utils.py
│       └── common.py
│
└── results/
    ├── eda/
    ├── models/
    ├── explainability/
    ├── rl/
    └── paper_figures/
```

---

# ⚙️ **Environment Setup**

See full instructions in `setup_instructions.md`.

### Quick Start (Apple M1–M4):

```bash
conda create -n synergy python=3.10 -y
conda activate synergy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Quick Start (CUDA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

# 🔍 **Datasets**

### 📦 **DrugComb**

Drug synergy experiments across >15,000 drug pairs.

### 🧬 **GDSC / CCLE (optional)**

Cell line genomics and drug response profiles.

### 💊 **ChEMBL or ZINC (for SSL)**

Used for masked node prediction & graph contrastive pretraining.

---

# 🧱 **Models Implemented**

### **1. Base GNNs**

* GCN
* GAT
* GraphSAGE
* AttentiveFP

### **2. 3D Geometric GNNs**

* EGNN
* DimeNet-lite
* SchNet

### **3. Graph Transformers**

* Graphormer
* GPS-style hybrid GNN–former
* ESM-style attention scalings

### **4. SMILES Transformers**

* ChemBERTa
* MolBART (optional)

### **5. Fusion Model (SOTA candidate)**

> **Graphormer + GeomGNN (3D) + ChemBERTa + Cell-line embeddings**

With:

* Co-attention
* Gated Feature Fusion
* Calibrated uncertainty outputs

---

# 🔬 **Self-Supervised Pretraining**

We implement and evaluate:

### ✓ Masked Atom Modeling (MAM)

### ✓ Masked Bond Prediction

### ✓ Graph Contrastive Learning

### ✓ Multitask SSL + Supervised Training

Pretrained model is saved to:
`/results/ssl_pretraining/`

---

# 🧠 **Explainability Toolkit**

Tools included:

* **GNNExplainer**
* **Integrated Gradients**
* **Attention rollout (for Graphormer)**
* **Saliency maps for SMILES tokens**
* **Counterfactual molecule editing**

Outputs are visualized in notebooks under:
`/notebooks/explainability/`

---

# 🤖 **Reinforcement Learning Module**

Drug selection framed as an RL environment.

### Implemented:

* Custom **Gymnasium** environment
* PPO baseline
* Soft constraints (toxicity / novelty penalties)
* Multi-agent:

  * Efficacy agent
  * Toxicity gatekeeper

Trained agents saved under:
`/results/rl/`

---

# 🧬 **LLM Finetuning (LoRA)**

We fine-tune a domain-adapted LLM to generate:

* Mechanistic explanations
* Clinical-style reasoning
* Attributed summaries tied to literature

### Pipeline includes:

* Dataset generation from model predictions
* LoRA adapters
* Hallucination suppression
* Evaluation rubric

---

# 📈 **Expected Outcomes**

By the end of this project, you will have:

* A **publication-ready model**
* A **self-supervised pretrained graph encoder**
* A **complete RL exploration of drug pair selection**
* A **medically-aligned explainable LLM agent**
* A polished, modular **research-grade codebase**
* Full set of **figures, ablations, and tables**
  ready for submission to MLHC/NeurIPS/ICML/ISMB.

---

# 🏁 Getting Started

Run the environment test:

```bash
jupyter lab notebooks/00_env_test.ipynb
```

Then start with:

```
notebooks/01_eda.ipynb
```

And follow the daily plan in:

```
project_plan/daily_plan.md
```

---