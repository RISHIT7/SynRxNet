# Formal Graduate Assignment: Drug Synergy Prediction & Optimization (Full Daily Plan)

**Format:** Daily `.md`-style plan â€” each day lists Objectives, Tasks (step-by-step file-level instructions), Time estimate, and Deliverables.
**Audience:** Senior undergraduate / early grad researcher (IIT Delhi 4th year level) aiming for publishable, SOTA work.
**Scope:** End-to-end: dataset â†’ GNNs/Transformers/3D â†’ SSL pretraining â†’ uncertainty & calibration â†’ interpretability â†’ LLM explainers (LoRA) â†’ RL/MARL â†’ demo + paper.
**Notes:** This expands the project into a **full daily plan**. It includes the two Day-1 papers you gave earlier (DeepSynergy by Preuer et al., 2018 and *Molecular representation learning: cross-domain foundations* (2025)) and uses your original daily-breakdown style as a template. 

---

# HOW TO USE THIS FILE

* Copy into `DAILY_FULL_PROJECT_PLAN.md` in your repo.
* Each day is actionable â€” create the files listed, run the scripts, and push commits.
* If a task is long (training, pretraining), start it and proceed to the next daily task; record run IDs and logs.
* Keep `docs/` updated with weekly summaries.

---

# WEEK 0 â€” Orientation & Core Readings (Days 1â€“3)

### Day 1 â€” Repo init, environment, Paper 1 (DeepSynergy)

**Time:** 3.5 hours
**Objectives:** Initialize repo, install base libs, read DeepSynergy (Preuer et al., 2018), and create summary.
**Tasks:**

1. Create repo skeleton:

   ```
    drug-synergy-research/
    â”œâ”€â”€ data/raw/
    â”œâ”€â”€ data/processed/
    â”œâ”€â”€ notebooks/
    â”œâ”€â”€ src/
    â”‚   â”œâ”€â”€ preprocessing/
    â”‚   â”œâ”€â”€ datasets/
    â”‚   â”œâ”€â”€ models/
    â”‚   â”œâ”€â”€ train/
    â”‚   â”œâ”€â”€ rl/
    â”‚   â””â”€â”€ utils/
    â”œâ”€â”€ experiments/
    â”œâ”€â”€ results/
    â””â”€â”€ docs/
   ```

   Commit initial structure.
2. Create `requirements.txt` with base libs: `torch`, `torch-geometric`, `rdkit`, `transformers`, `scikit-learn`, `optuna`, `stable-baselines3`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `wandb`.
3. Create `setup_instructions.md` with conda/pip commands and GPU notes.
4. Read *DeepSynergy* and write `docs/lit_reviews/deepsynergy_summary.md` with:

   * 1-paragraph summary
   * 5 bullet takeaways: dataset, model architecture, preprocessing, evaluation, limitations.
     **Deliverable:** repo skeleton, `requirements.txt`, `setup_instructions.md`, `docs/lit_reviews/deepsynergy_summary.md`.

---

### Day 2 â€” Paper 2 (Long): Part 1 â€” Read & Annotate

**Time:** 4 hours
**Objectives:** Begin reading *Molecular representation learning: cross-domain foundations* (2025). Save notes and implementation plan.
**Tasks:**

1. Read Intro, Methods, part of Results. Annotate PDF (save to `docs/lit_reviews/paper2_annotated_part1.pdf`).
2. Create `docs/lit_reviews/paper2_part1_notes.md` capturing:

   * Key architectures mentioned (Graphormer, GIN, ChemBERTa, etc.)
   * Proposed pretraining methods and takeaways.
3. Draft `notebooks/paper2_reimpl_plan.ipynb` sketching:

   * Small components to implement (node embedding, spatial encoding, masked pretraining).
   * API shapes: class/function names, expected inputs/outputs.
     **Deliverable:** annotated PDF, `paper2_part1_notes.md`, reimplementation plan notebook.

---

### Day 3 â€” Paper 2: Part 2 â€” Finish + Micro-implementation

**Time:** 4 hours
**Objectives:** Finish reading paper; implement a small building block (node embedding or spatial encoding skeleton).
**Tasks:**

1. Finish reading and create `docs/lit_reviews/paper2_part2_notes.md`.
2. Implement `src/models/transformer/paper2_node_embedding.py`:

   ```py
   class NodeEmbedding(nn.Module):
       def __init__(self, in_dim, hid_dim):
           ...
       def forward(self, node_feats):
           ...
   ```

   Add docstring and a unit test in `tests/test_node_embedding.py`. \
   **Deliverable:** node embedding module + tests + notes.

---

# WEEK 1 â€” Data acquisition, cleaning, EDA & quick baselines (Days 4â€“10)

### Day 4 â€” Download datasets & provenance

**Time:** 2 hours
**Objectives:** Download DrugComb, Oâ€™Neil (DeepSynergy) dataset, CCLE subset. Record provenance.
**Tasks:**

1. Save datasets to `data/raw/`: `drugcomb_sample.csv`, `oneil_deepsynergy.csv`, `ccle_expression_subset.csv`.
2. Create `data/raw/README.md` with URLs, dates, SHA256 hashes, and short descriptions.
   **Deliverable:** raw datasets + provenance readme.

---

### Day 5 â€” SMILES validation & basic cleaning

**Time:** 3 hours
**Objectives:** Validate SMILES, remove invalid entries, save cleaned CSVs.
**Tasks:**

1. Implement `src/preprocessing/data_loader.py` with:

   * `load_raw(path)`, `validate_smiles(smiles)`, `clean_dataset(df)`.
2. Run validation on sample; write `data/processed/invalid_smiles.csv`.
3. Save cleaned dataset to `data/processed/cleaned_drugcomb.csv`.
   **Deliverable:** data loader, invalid list, cleaned CSV.

---

### Day 6 â€” RDKit descriptor & fingerprints

**Time:** 3 hours
**Objectives:** Extract 2D descriptors and Morgan fingerprints.
**Tasks:**

1. Implement `src/preprocessing/feature_engineer.py` with:

   * `compute_rdkit_descriptors(smiles)`, `compute_morgan(smiles, radius=2, nBits=2048)`.
2. Produce `data/processed/features_sample.csv` (1000 rows).
   **Deliverable:** feature scripts + sample features.

---

### Day 7 â€” EDA notebook & report

**Time:** 3 hours
**Objectives:** Build `notebooks/01_eda.ipynb`: distributions, missingness, SMILES validity, synergy histograms.
**Tasks:**

1. Create notebook with plots: synergy distribution, cell-line counts, drug counts, missingness heatmap.
2. Export `docs/eda_report.md` summarizing key findings and preprocessing decisions.
   **Deliverable:** notebook + EDA report.

---

### Day 8 â€” Create dataset splits (3 variants)

**Time:** 2 hours
**Objectives:** Implement three splitting strategies and save them.
**Tasks:**

1. Implement `src/preprocessing/splitter.py`:

   * `stratified_random_split(df, seed)`, `leave_one_drug_out(df)`, `leave_one_cellline_out(df)`.
2. Save CSVs: `data/processed/train.csv`, `val.csv`, `test.csv` for each split under `data/processed/splits/`.
3. Document splitting rationale in `docs/split_strategy.md`.
   **Deliverable:** split CSVs + split strategy doc.

---

### Day 9 â€” Quick baseline: XGBoost

**Time:** 3 hours
**Objectives:** Train XGBoost on RDKit features to get baseline metrics.
**Tasks:**

1. Implement `src/models/baseline/xgb_baseline.py` with `train()` and `evaluate()` functions.
2. Create `notebooks/02_baselines.ipynb` to run XGBoost and record RMSE/AUROC.
3. Save model to `results/models/xgb_baseline.pkl`.
   **Deliverable:** notebook, model, baseline metrics CSV `results/baselines.csv`.

---

### Day 10 â€” Quick baseline: MLP & compare

**Time:** 3 hours
**Objectives:** Implement simple MLP baseline and compare to XGBoost.
**Tasks:**

1. Implement `src/models/baseline/mlp.py` with `MLPBaseline` class and `src/train/train_mlp.py` training wrapper.
2. Run short training, save checkpoint to `results/models/mlp_baseline.pt`.
3. Update `results/baselines.csv` with MLP metrics and write short `docs/week1_summary.md`.
   **Deliverable:** MLP checkpoint, updated baseline comparison, week1 summary.

---

# WEEK 2 â€” Graph construction & basic GNNs (Days 11â€“17)

### Day 11 â€” SMILES â†’ PyG Data (graph_dataset)

**Time:** 3 hours
**Objectives:** Convert SMILES to PyG `Data` objects with atom/node and bond/edge features.
**Tasks:**

1. Implement `src/datasets/graph_dataset.py`:

   * `smiles_to_pyg(smiles)`, support `add_bond_features=True`, `add_atom_features=True`.
2. Cache 1,000 `Data` objects in `data/processed/graph_cache/`.
3. Unit tests `tests/test_graph_dataset.py`.
   **Deliverable:** dataset module + cache + tests.

---

### Day 12 â€” GCN encoder & synergy head

**Time:** 3 hours
**Objectives:** Implement baseline GCN encoder and pair-fusion head.
**Tasks:**

1. `src/models/gnn/gcn.py`:

   * `GCNEncoder(nn.Module)`, uses `GCNConv` layers.
   * `SynergyHead(nn.Module)` that concatenates drugA_emb, drugB_emb, cell_line_vector and outputs regression/class scores.
2. Add example usage in docstring and `tests/test_gcn.py`.
   **Deliverable:** gcn module + test.

---

### Day 13 â€” GAT model

**Time:** 3 hours
**Objectives:** Implement GAT variant with multi-head attention.
**Tasks:**

1. `src/models/gnn/gat.py` using `GATConv`, 4 heads by default.
2. Integration test to forward a batch of `Data` objects.
   **Deliverable:** gat module + test.

---

### Day 14 â€” Training loop & experiment scaffolding

**Time:** 4 hours
**Objectives:** Build robust training script with logging, checkpoints, early stopping.
**Tasks:**

1. `src/train/train_gnn.py`:

   * Argparse config, DataLoader, optimizer, scheduler, early stop, wandb/logging optional.
2. Run a 10-epoch smoke test on small subset and save to `results/runs/gnn_smoke/`.
   **Deliverable:** train script + smoke-run saved.

---

### Day 15 â€” Metrics & evaluation tools

**Time:** 2 hours
**Objectives:** Implement evaluation metrics and visualization helpers.
**Tasks:**

1. `src/utils/metrics.py` with: RMSE, MAE, AUROC, AUPRC, Pearson, Spearman.
2. `src/utils/plotting.py` functions: ROC, PR curves, error histograms.
3. Unit tests for metrics.
   **Deliverable:** metrics module + plotting utilities.

---

### Day 16 â€” Ablation: features & cell-line fusion

**Time:** 3 hours
**Objectives:** Run ablations to see contribution of atom features vs bond-only vs adding cell-line vectors.
**Tasks:**

1. Use `train_gnn.py` to run 3 short experiments (same seed).
2. Save results to `results/ablations/week2_feature_ablation.csv`.
3. Write short notes `docs/week2_ablation_notes.md`.
   **Deliverable:** ablation CSV + notes.

---

### Day 17 â€” Weekly summary & cleanup

**Time:** 2 hours
**Objectives:** Consolidate Week 2 results, tag git commit, write `docs/week2_summary.md`.
**Tasks:**

1. Collect best checkpoints, training logs, and seed info.
2. Update README with how to run GCN/GAT smoke tests.
   **Deliverable:** week2_summary + updated README.

---

# WEEK 3 â€” 3D conformers & geometric GNNs (Days 18â€“24)

### Day 18 â€” Conformer generation pipeline

**Time:** 4 hours (CPU-heavy)
**Objectives:** Implement 3D conformer generation using RDKit ETKDG; store conformers.
**Tasks:**

1. `src/preprocessing/rdkit_3d.py`:

   * `generate_conformers(smiles, n_confs=5, force_field='MMFF')`, energy minimize, select best conformer.
2. Save conformer coordinates to `data/external/conformers/{mol_id}.sdf` or JSON.
3. Run generation on 500 molecules; log failures to `logs/3d_failures.csv`.
   **Deliverable:** conformer generation module + conformer files.

---

### Day 19 â€” 3D descriptor extraction

**Time:** 3 hours
**Objectives:** Compute Coulomb matrices, partial charges, and simple 3D fingerprints.
**Tasks:**

1. Extend `rdkit_3d.py` with `compute_coulomb_matrix(coords, charges)`, `coulomb_eigenvalues_sorted()`, `distance_histograms()`.
2. Save per-molecule JSON to `data/external/3d_features/`.
   **Deliverable:** 3D features saved for sampled molecules.

---

### Day 20 â€” Implement EGNN-style model

**Time:** 4 hours
**Objectives:** Implement a coordinate-aware message passing module (EGNN).
**Tasks:**

1. `src/models/gnn/egnn.py`:

   * Implement message passing that updates coordinates and features, preserve equivariance.
2. Add unit test `tests/test_egnn.py` comparing invariances (rotate inputs, outputs invariant).
   **Deliverable:** egnn module + tests.

---

### Day 21 â€” Train EGNN with 3D features

**Time:** 4 hours
**Objectives:** Train EGNN on sample dataset incorporating 3D features; log results.
**Tasks:**

1. Use `src/train/train_gnn.py` (adapt to accept `use_3d=True`).
2. Train 15 epochs on subset, store run in `results/runs/egnn_3d/`.
   **Deliverable:** EGNN checkpoint + logs.

---

### Day 22 â€” 3D vs 2D ablation study

**Time:** 3 hours
**Objectives:** Compare EGNN vs GCN with/without 3D features.
**Tasks:**

1. Run 3 experiments: (GCN 2D), (GCN + 3D features), (EGNN + 3D).
2. Save results `results/ablation_3d_vs_2d.csv` and plot comparison.
   **Deliverable:** ablation CSV + plot PNG.

---

### Day 23 â€” Graphormer planning & encoding functions

**Time:** 3 hours
**Objectives:** Design Graphormer encodings (shortest path bins, degree centrality).
**Tasks:**

1. Implement `src/utils/graph_utils.py` functions:

   * `shortest_path_matrix(edge_index, num_nodes)`, `degree_encoding(nodes)`.
2. Create `src/models/transformer/graphormer_encodings.py` with encoding classes.
   **Deliverable:** encoding utilities + doc.

---

### Day 24 â€” Week 3 summary & documentation

**Time:** 2 hours
**Objectives:** Write `docs/week3_3d_summary.md` summarizing findings, noting which 3D features helped.
**Tasks:**

1. Compile metrics, training curves, and decisions for Week 4.
   **Deliverable:** week3 3D summary doc.

---

### Optional module â€” VAE & Diffusion Primer (insert between Week 3 and Week 4)

**Time:** 2â€“3 days
**Objectives:** Learn variational autoencoders (VAE) for compact latent representations, then build intuition and a minimal workflow for denoising/diffusion generative models (DDPM/score-based). This module is intentionally short and hands-on: implement toy versions on vectorized molecular descriptors (Morgan fingerprints / RDKit descriptors) before attempting equivariant diffusion for 3D coordinates.

Day A â€” VAE fundamentals & tiny implementation

- **Time:** 0.5â€“1 day
- **Objectives:** Understand VAE theory (ELBO, reparameterization trick) and implement a minimal MLP VAE that compresses fingerprint vectors to a small latent space.
- **Tasks:**
   - Read: sections on VAE from a standard tutorial (e.g., Kingma & Welling chapter / good tutorial blog).
   - Implement a small VAE on fingerprint vectors (train a quick smoke-run on random or sampled features).
   - Notebook: `notebooks/vaes/vaes_intro_and_toy.ipynb` with plots of reconstructions, latent traversal, and ELBO components.
- **Deliverable:** notebook with results, short `docs/vaes_notes.md` summarizing hyperparameters and takeaways.

Day B â€” DDPM / score-based models on vectors

- **Time:** 0.75â€“1 day
- **Objectives:** Learn diffusion basics (forward noising, reverse denoising) and implement a toy DDPM on the same vector representation used for the VAE.
- **Tasks:**
   - Read: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) and a short tutorial on practical training/sampling.
   - Implement a compact DDPM that predicts noise on noisy fingerprints; train for a few epochs and sample.
   - Notebook: `notebooks/diffusion/ddpm_vectors.ipynb` demonstrating forward noise schedules, a training step, and sampling quality (validity / basic stats vs dataset).
- **Deliverable:** notebook and `docs/diffusion_vectors_notes.md` with sampling tips and failure modes.

Day C â€” (Optional) Conditioning & 3D / equivariant extensions

- **Time:** 0.5â€“1 day (optional, follow-up)
- **Objectives:** Sketch how to move from vector DDPMs to conditioned generation (condition on scaffolds/cell-line) and to equivariant score models for 3D coordinates.
- **Tasks:**
   - Read: score-based SDE papers (Song et al.) and recent molecular diffusion works (GeoDiff, equivariant diffusion papers). Note practical libraries and common design patterns.
   - Design experiment ideas (not required to implement immediately): condition sampling on molecular graph embeddings, or adapt coordinate-aware architectures for 3D conformer generation.
   - Notebook: `notebooks/diffusion/plan_equivariant_diffusion.ipynb` with references, diagrams, and a minimal roadmap for next steps.
- **Deliverable:** roadmap notebook and `docs/diffusion_next_steps.md` listing required libs (e3nn, equivariant layers), compute needs, and evaluation metrics (validity, novelty, property distributions).

**Notes / Teaching path:**
- Start with VAE to internalize latent compression and reconstruction losses (quick feedback loop).
- Move to DDPM on vectors to grasp diffusion schedules, prediction targets (noise vs x0), and sampling behaviour.
- Only after you are comfortable, explore equivariant diffusion for 3D molecules (this is more advanced and requires careful architecture choice and compute).

**Deliverable for module:** three short notebooks and two docs files (`docs/vaes_notes.md`, `docs/diffusion_vectors_notes.md`) that allow you to continue to equivariant diffusion when ready.


---

# WEEK 4 â€” ChemBERTa, Graphormer, and fusion models (Days 25â€“31)

### Day 25 â€” ChemBERTa & SMILES tokenization

**Time:** 3 hours
**Objectives:** Integrate ChemBERTa (HF) as a SMILES encoder.
**Tasks:**

1. Implement `src/models/transformer/chemberta_encoder.py`:

   * Wrap `AutoTokenizer` and `AutoModel`, project to 256-dim embedding.
2. Create `notebooks/chemberta_test.ipynb` showing tokenization, embedding shapes, batch inference.
   **Deliverable:** chemberta module + notebook.

---

### Day 26 â€” Graphormer core attention block

**Time:** 4 hours
**Objectives:** Implement Graphormer attention with edge/spatial bias.
**Tasks:**

1. Implement `src/models/transformer/graphormer.py` skeleton:

   * `GraphormerLayer` with attention scoring augmented by spatial encodings.
2. Add docstrings and small forward test `tests/test_graphormer_forward.py`.
   **Deliverable:** graphormer module + tests.

---

### Day 27 â€” Fusion model API & base implementation

**Time:** 3 hours
**Objectives:** Build modular fusion model that accepts multiple encoders.
**Tasks:**

1. `src/models/fusion/fusion_model.py`:

   * Accepts `drugA_emb`, `drugB_emb`, `cell_emb`, `fusion_type` (concat/gated/co-attention).
   * Implement all three fusion types with config switch.
2. Provide example config `experiments/configs/fusion_example.json`.
   **Deliverable:** fusion module + example config.

---

### Day 28 â€” Train small fusion models (smoke test)

**Time:** 4 hours
**Objectives:** Run small scale fusion training runs to ensure everything works end-to-end.
**Tasks:**

1. Run fusion model with `GCN + ChemBERTa` for 8â€“10 epochs on subset.
2. Save run to `results/runs/fusion_smoke/`.
   **Deliverable:** fusion smoke-run checkpoint + logs.

---

### Day 29 â€” Ensemble & stacking utilities

**Time:** 3 hours
**Objectives:** Implement ensemble averaging and a stacking wrapper.
**Tasks:**

1. `src/train/ensemble_runner.py`:

   * Average predictions from multiple checkpoints.
   * Implement simple stacking (train meta-learner on validation predictions).
2. Test ensemble with existing saved models.
   **Deliverable:** ensemble runner + test.

---

### Day 30 â€” Extended evaluation & calibration utilities

**Time:** 3 hours
**Objectives:** Implement calibration (temperature scaling) and ECE computed on outputs.
**Tasks:**

1. `src/utils/calibration.py` with `temperature_scale(logits, labels)` and `compute_ece(probs, labels)`.
2. Run calibration on fusion model outputs, save reliability diagram to `results/plots/reliability_fusion.png`.
   **Deliverable:** calibration module + plots.

---

### Day 31 â€” Week 4 report & literature updates

**Time:** 2 hours
**Objectives:** Document Week 4 experiments, update `docs/lit_reviews/` with relevant transformer papers you used.
**Tasks:**

1. Write `docs/week4_transformer_summary.md` and append used citations to `docs/references.bib`.
   **Deliverable:** week4 report + updated bibliography.

---

# WEEK 5 â€” Self-supervised pretraining (Days 32â€“38)

### Day 32 â€” Prepare SSL corpus

**Time:** 3 hours
**Objectives:** Collect a large unlabeled molecular corpus (ChEMBL / ZINC subset) and preprocess.
**Tasks:**

1. Download small subset (50kâ€“200k molecules) to `data/external/ssl_corpus/`.
2. Preprocess to SMILES lists and cached PyG graphs via `src/preprocessing/ssl_prep.py`.
   **Deliverable:** preprocessed SSL corpus.

---

### Day 33 â€” Implement masked node/edge pretext

**Time:** 4 hours
**Objectives:** Create masked prediction pretext for GNN pretraining.
**Tasks:**

1. `src/train/pretrain_masked.py` implementing masked node attribute prediction with cross-entropy/MSE depending on target. Include augmentation hooks.
2. Smoke-run on small batch.
   **Deliverable:** pretraining script + smoke-run.

---

### Day 34 â€” Implement graph contrastive learning

**Time:** 4 hours
**Objectives:** Implement InfoNCE contrastive pretraining with multiple augmentations.
**Tasks:**

1. `src/train/pretrain_contrastive.py`:

   * Augmentations: node-dropping, edge-perturbation, conformer perturbation (if available).
   * Projection head and contrastive loss.
2. Test training for a few steps.
   **Deliverable:** contrastive script + test logs.

---

### Day 35 â€” Launch longer SSL pretraining run (start)

**Time:** 2 hours (start job; long-running)
**Objectives:** Kick off SSL pretraining; save checkpoints periodically.
**Tasks:**

1. Start job (e.g., `python src/train/pretrain_contrastive.py --config experiments/configs/pretrain_contrastive.json`).
2. Save checkpoint schedule and log metadata to `experiments/pretrain/logs/`.
   **Deliverable:** pretraining job started and first checkpoint.

> Note: Let job run while you proceed to other tasks; monitor logs.

---

### Day 36 â€” SSL fine-tune pipeline

**Time:** 3 hours
**Objectives:** Build pipeline to load SSL checkpoints and fine-tune on synergy prediction.
**Tasks:**

1. `src/train/ssl_finetune.py` that:

   * Loads SSL encoder, attaches `SynergyHead`, fine-tunes end-to-end.
2. Test with a short fine-tune (3 epochs).
   **Deliverable:** finetune script + sample run.

---

### Day 37 â€” Compare pretrained vs non-pretrained

**Time:** 3 hours
**Objectives:** Evaluate benefit of SSL on downstream task (few-seeds).
**Tasks:**

1. Run finetune for pretrained vs random-init for 3 seeds each.
2. Save `results/ssl_vs_random.csv` and write `docs/ssl_finetune_results.md`.
   **Deliverable:** results CSV + analysis doc.

---

### Day 38 â€” Week 5 summary

**Time:** 2 hours
**Objectives:** Consolidate SSL results and decide next steps for large-scale pretraining.
**Tasks:**

1. Update `docs/week5_ssl_summary.md` and plan further pretraining resources if needed.
   **Deliverable:** Week 5 summary.

---

# WEEK 6 â€” Hyperparameter optimization, model compression, calibration (Days 39â€“45)

### Day 39 â€” Optuna integration (objective wrapper)

**Time:** 3 hours
**Objectives:** Implement Optuna objective that wraps model training and returns validation metric.
**Tasks:**

1. `src/train/hyperopt_bayes.py` with Optuna study creation and pruning.
2. Use a small sample objective for smoke-testing.
   **Deliverable:** hyperopt script + sample study saved in `experiments/hyperopt/`.

---

### Day 40 â€” Run initial hyperopt study (20 trials)

**Time:** 4â€“8 hours (start & monitor)
**Objectives:** Run initial tuning for a target model (e.g., fusion model).
**Tasks:**

1. Start 20-trial study; set search space: lr, weight_decay, dropout, hidden_dim, num_layers.
2. Save best trial config to `experiments/hyperopt/best_trial.json`.
   **Deliverable:** Optuna study results + best config.

---

### Day 41 â€” Multi-seed verification of best config

**Time:** 3 hours
**Objectives:** Re-run best config on 5 seeds to establish robustness.
**Tasks:**

1. Run training script 5 times with seed variations; aggregate metrics into `results/hyperopt_best_multi_seed.csv`.
   **Deliverable:** multi-seed results.

---

### Day 42 â€” Model distillation (teacherâ†’student)

**Time:** 4 hours
**Objectives:** Distill ensemble (teacher) into smaller student model for inference efficiency.
**Tasks:**

1. Implement `src/train/distill.py`: teacher prediction soft-targets + student learns via KL + MSE.
2. Distill on a portion of dataset and evaluate student.
   **Deliverable:** distillation script + student checkpoint.

---

### Day 43 â€” Model pruning/quantization experiment

**Time:** 3 hours
**Objectives:** Test quantization-aware inference or PyTorch dynamic quantization for student model.
**Tasks:**

1. Implement `scripts/quantize_model.py` that loads student checkpoint and applies `torch.quantization` or `torch.jit`.
2. Measure inference latency/memory and plot improvements in `docs/inference_benchmarks.md`.
   **Deliverable:** quantized model + benchmark results.

---

### Day 44 â€” Ensemble & calibration finalization

**Time:** 3 hours
**Objectives:** Build final ensemble of top models and calibrate probabilities/regression outputs.
**Tasks:**

1. Run `src/train/ensemble_runner.py` with top-k models and produce ensemble predictions.
2. Apply temperature scaling via `src/utils/calibration.py` and compute ECE.
   **Deliverable:** ensemble predictions + calibration report.

---

### Day 45 â€” Week 6 report

**Time:** 2 hours
**Objectives:** Summarize HPO, distillation, quantization, and calibration results. Prepare final configs for next phases.
**Tasks:**

1. `docs/week6_hpo_distill_calibration.md`.
   **Deliverable:** week6 report.

---

# WEEK 7 â€” Interpretability & Clinical mapping (Days 46â€“52)

### Day 46 â€” GNNExplainer integration

**Time:** 3 hours
**Objectives:** Implement GNNExplainer wrapper for PyG models and generate atom-level saliency.
**Tasks:**

1. `src/interpretability/gnn_explainer.py` wraps PyG's GNNExplainer for your models and saves highlight masks.
2. Run explainer on 50 examples, save images to `results/interpretability/gnn_explainer/`.
   **Deliverable:** explainer outputs + code.

---

### Day 47 â€” Attention head visualization for Graphormer

**Time:** 3 hours
**Objectives:** Extract attention scores from Graphormer layers and map to atom indices; visualize.
**Tasks:**

1. `src/interpretability/attention_viz.py` extracts head-wise attention and creates heatmaps over molecule diagrams using RDKit drawing with highlights.
2. Save examples to `results/interpretability/attention/`.
   **Deliverable:** attention visualizations + code.

---

### Day 48 â€” Gradient-based SMILES saliency

**Time:** 3 hours
**Objectives:** Compute gradient saliency over SMILES tokens for ChemBERTa-based models.
**Tasks:**

1. Implement `src/interpretability/smiles_saliency.py` that calculates token gradients (input embeddings) and maps to tokens.
2. Save results for 100 examples to `results/interpretability/smiles_saliency/`.
   **Deliverable:** saliency outputs + notebook showing examples.

---

### Day 49 â€” Pharmacophore motif mapping & manual literature linkage

**Time:** 4 hours
**Objectives:** For top 20 high-importance substructures, manually search literature and map to possible mechanism/toxicity motifs.
**Tasks:**

1. Create `docs/clinical_motif_mapping.md` with motif (SMILES substructure), explanation, and references.
2. Save `results/interpretability/motif_map.json`.
   **Deliverable:** motif mapping doc + JSON.

---

### Day 50 â€” Counterfactual generation & sensitivity

**Time:** 4 hours
**Objectives:** Implement minimal structural changes (add/remove atom or bond) to assess sensitivity of predictions.
**Tasks:**

1. `src/interpretability/counterfactuals.py`:

   * For a molecule, enumerate small edits, compute predicted synergy change, rank edits by delta.
2. Save counterfactual results to `results/interpretability/counterfactuals.csv`.
   **Deliverable:** counterfactual CSV + scripts.

---

### Day 51 â€” Create LLM prompt templates (for LoRA)

**Time:** 3 hours
**Objectives:** Produce high-quality prompt-response templates anchored to model interpretability outputs.
**Tasks:**

1. Make `data/llm_prompts/generate_prompts.py` that given model output + explainer artifacts generates a prompt template.
2. Create 500 seed prompt-response pairs (manually write ~200, algorithmically generate the rest with careful templates).
3. Save to `data/llm_prompts/train.jsonl`.
   **Deliverable:** prompt dataset + generation script.

---

### Day 52 â€” Week 7 report

**Time:** 2 hours
**Objectives:** Consolidate interpretability experiments and prepare LLM dataset for fine-tuning.
**Tasks:**

1. `docs/week7_interpretability.md`.
   **Deliverable:** week7 report.

---

# WEEK 8 â€” LLM LoRA fine-tuning & evaluation (Days 53â€“59)

### Day 53 â€” LoRA integration (PEFT)

**Time:** 3 hours
**Objectives:** Setup LoRA (or PEFT) utilities and training wrapper.
**Tasks:**

1. Implement `src/models/llm/lora_utils.py` (or integrate `peft` if allowed), with functions to attach LoRA adapters to transformer layers.
2. Create `src/models/llm/lora_finetune.py` which uses HF Trainer to fine-tune adapters only.
   **Deliverable:** LoRA utilities + training script.

---

### Day 54 â€” Prepare small LLM & dry run

**Time:** 2 hours
**Objectives:** Choose a model (e.g., `t5-small` or `flan-t5-base` for text generation) and run a 1-epoch dry run on 10 samples.
**Tasks:**

1. Create config `experiments/configs/lora_small.json`.
2. Run `lora_finetune.py` for 1 epoch to ensure compatibility.
   **Deliverable:** dry-run logs.

---

### Day 55 â€” LoRA fine-tuning (start)

**Time:** 4â€“8 hours (start job)
**Objectives:** Fine-tune LoRA on `data/llm_prompts/train.jsonl`. Save checkpoints and logs.
**Tasks:**

1. Run training with eval on validation set. Save best LoRA adapter to `results/llm/lora/checkpoint_best/`.
2. Record training metrics and sample outputs.
   **Deliverable:** LoRA checkpoint + logs.

---

### Day 56 â€” LLM evaluation: factuality & alignment checks

**Time:** 3 hours
**Objectives:** Evaluate LoRA outputs for factuality and alignment to explainer output.
**Tasks:**

1. `notebooks/08_llm_eval.ipynb`: metrics include: token overlap with required atoms, whether explanation references explainer artifacts, human-like clarity scores (manual small subset).
2. Save evaluation report to `results/llm/eval_report.md`.
   **Deliverable:** LLM evaluation report + notebook.

---

### Day 57 â€” Hallucination detection & mitigation

**Time:** 3 hours
**Objectives:** Create a simple hallucination detector that flags claims ungrounded by explainer or model outputs.
**Tasks:**

1. `src/models/llm/hallucination_check.py` compares named entities/mechanisms in LLM output to explainer artifacts and to a small curated KB (local list).
2. Run on test set and produce `results/llm/hallucination_stats.csv`.
   **Deliverable:** hallucination module + stats.

---

### Day 58 â€” Integrate LoRA LLM into pipeline

**Time:** 3 hours
**Objectives:** Implement end-to-end pipeline: predictor â†’ explainer â†’ LLM report generation.
**Tasks:**

1. `scripts/generate_explanation.py`:

   * Input: SMILES pair + cell-line id â†’ outputs: predicted synergy, top atoms, LLM text explanation, confidence scores.
2. Test pipeline for 20 examples and save outputs to `results/llm/pipeline_outputs.jsonl`.
   **Deliverable:** pipeline script + outputs.

---

### Day 59 â€” Week 8 report

**Time:** 2 hours
**Objectives:** Summarize LoRA performance and LLM integration. Prepare for demo building.
**Tasks:**

1. `docs/week8_llm_summary.md`.
   **Deliverable:** week8 report.

---

# WEEK 9 â€” RL & MARL (Days 60â€“66)

### Day 60 â€” RL problem formalization

**Time:** 2 hours
**Objectives:** Complete MDP spec for drug-combo selection; define state, action, reward, termination.
**Tasks:**

1. `docs/rl_problem_definition.md` including formal math, constraints, and safety penalties (toxicity).
   **Deliverable:** RL spec doc.

---

### Day 61 â€” Implement Gym env (drug_combo_env)

**Time:** 4 hours
**Objectives:** Build `src/rl/envs/drug_combo_env.py` implementing `gym.Env` interface.
**Tasks:**

1. Observation: cell-line vector + candidate drug vectors (or indices) + history mask.
2. Action: pick drug index; episode ends when k drugs chosen or pool exhausted.
3. Reward: final predicted synergy (from predictor) minus toxicity penalty (if available).
4. Unit tests for env methods.
   **Deliverable:** gym env + tests.

---

### Day 62 â€” Random & greedy baselines

**Time:** 2 hours
**Objectives:** Implement random and greedy baseline policies and evaluate.
**Tasks:**

1. `src/rl/agents/random_agent.py` and `src/rl/agents/greedy_agent.py`.
2. `notebooks/09_rl_baselines.ipynb` run and log average cumulative reward over 200 episodes.
   **Deliverable:** baseline results.

---

### Day 63 â€” PPO agent (single-agent)

**Time:** 5â€“8 hours (start training)
**Objectives:** Implement PPO discrete action agent using SB3 and train.
**Tasks:**

1. `src/rl/agents/ppo_agent.py` using `stable-baselines3` or custom PPO.
2. Train for specified timesteps; log to `results/rl/ppo/`.
   **Deliverable:** trained policy + logs.

---

### Day 64 â€” Constrained-safe RL (toxicity constraint)

**Time:** 4 hours
**Objectives:** Add constraint handling via reward shaping or Lagrangian approach to keep toxicity under threshold.
**Tasks:**

1. Implement `src/rl/agents/constrained_ppo.py` with Lagrangian multiplier update.
2. Compare with unconstrained PPO and plot average toxicity vs reward tradeoffs.
   **Deliverable:** constrained agent checkpoint + comparison plots.

---

### Day 65 â€” MARL prototype (optional advanced)

**Time:** 4 hours
**Objectives:** Prototype multi-agent formulation (two agents: efficacy and toxicity). Centralized critic, decentralized actors.
**Tasks:**

1. `src/rl/marl/marl_prototype.py` prototypes training loop for two agents and a centralized critic.
2. Run few episodes as test; log simple metrics.
   **Deliverable:** MARL prototype code + logs.

---

### Day 66 â€” Week 9 report

**Time:** 2 hours
**Objectives:** Summarize RL experiments, produce plots of learning curves and tradeoffs.
**Tasks:**

1. `docs/week9_rl_summary.md`.
   **Deliverable:** week9 report.

---

# WEEK 10 â€” Demo polishing, reproducibility & paper drafting (Days 67â€“73)

### Day 67 â€” Build Gradio demo

**Time:** 4 hours
**Objectives:** Construct a Gradio app to input two SMILES and a cell-line, display prediction, explainer visuals, and LLM explanation. Containerize later.
**Tasks:**

1. `presentation/demo/app.py` using Gradio or Streamlit:

   * Inputs: SMILES A, SMILES B, cell-line id (dropdown).
   * Outputs: predicted synergy score, top-5 atoms image, LLM text explanation.
2. Test app locally and save example screenshots.
   **Deliverable:** demo app + screenshots.

---

### Day 68 â€” Dockerfile & run scripts

**Time:** 3 hours
**Objectives:** Containerize demo and core scripts for reproducibility.
**Tasks:**

1. `Dockerfile` that installs required libs (use GPU base if needed).
2. `scripts/run_demo_docker.sh` and `scripts/run_training_docker.sh`.
   **Deliverable:** Dockerfile + run scripts.

---

### Day 69 â€” Final experiment aggregation

**Time:** 4 hours
**Objectives:** Collect best checkpoints, final metrics across splits & seeds for all model families.
**Tasks:**

1. Create `SUBMISSION/final_results_table.csv` with rows per model family and metrics.
2. Save final models under `SUBMISSION/models/`.
   **Deliverable:** final results table + model artifacts.

---

### Day 70 â€” Figures for paper

**Time:** 5 hours
**Objectives:** Produce high-quality figures for methods & results: architecture diagram, ROC/PR curves, ablation bars, attention visualizations, t-SNE of embeddings.
**Tasks:**

1. Use Matplotlib to generate vector graphics (`.svg` or high-res `.png`) in `presentation/figures/`.
   **Deliverable:** figure set.

---

### Day 71 â€” Draft Methods & Experiments sections

**Time:** 6 hours
**Objectives:** Write `docs/paper/methods.md` and `docs/paper/experiments.md` with rigorous details (configs, hyperparams, seeds).
**Tasks:**

1. Fill training regimen, data preparation, model architectures, and evaluation metrics.
   **Deliverable:** methods + experiments drafts.

---

### Day 72 â€” Draft Intro, Related Work & Lit Review

**Time:** 6 hours
**Objectives:** Write `docs/paper/intro.md` and `docs/paper/related_work.md`. Include DeepSynergy and Paper 2 with proper summaries. 
**Tasks:**

1. For each cited paper, include a 1-paragraph summary and how your work builds on it (use `docs/lit_reviews/`).
   **Deliverable:** intro + related work drafts.

---

### Day 73 â€” Ethics, limitations, and final polishing

**Time:** 4 hours
**Objectives:** Write ethics & limitations section, finalize README and submission manifest. Prepare for arXiv/venue submission checklist.
**Tasks:**

1. `docs/paper/ethics.md` including disclaimer about non-clinical claims, LLM hallucination risk, data biases.
2. `SUBMISSION/SUBMISSION_MANIFEST.txt` listing files included and instructions to reproduce top results.
   **Deliverable:** ethics doc + submission manifest.

---

# LITERATURE REVIEW: Core Papers & Daily Study Plan (integrate into calendar above; repeat until mastered)

> For each paper below: read, write a one-page summary in `docs/lit_reviews/`, extract key algorithmic/architectural elements, and add reproduction notes (if feasible).

**Day A1 (separate from daily schedule) â€” DeepSynergy (Preuer et al., 2018)**

* Already assigned on Day 1; ensure `docs/lit_reviews/deepsynergy_summary.md` contains reproduction plan.

**Day A2 â€” Molecular representation learning: cross-domain foundations (2025)**

* Already assigned on Days 2â€“3; ensure `paper2_part1_notes.md` and `paper2_part2_notes.md` fully capture methods and pretraining suggestions.

**Additional recommended papers (assign 1â€“2 days per paper to read + summarize):**

* Graphormer (Ying et al. / relevant Graphormer literature) â€” implement core ideas.
* Graph Attention Networks (VeliÄkoviÄ‡ et al., 2018).
* GIN: Xu et al. (2019) â€” theory for expressive GNNs.
* EGNN or other equivariant GNNs (Satorras et al.) â€” implement EGNN ideas.
* DimeNet / SchNet (3D-aware models).
* ChemBERTa model card/paper.
* GNNExplainer (Ying et al.) and Integrated Gradients papers.
* Recent 2021â€“2025 papers on drug synergy using deep learning (search DrugComb/DrugCombDB/BioRxiv) â€” summarize 3â€“5 most relevant.

**Deliverable per paper:** `docs/lit_reviews/{short_name}.md` with: citation, 1-paragraph summary, methods list, datasets used, reproduction plan, how it informs our choices.

---

# LONG-TERM & OPTIONAL EXTENSIONS (extra days beyond 73)

* Add **Bayesian optimization (BOHB / BO+Hyperband)** for large budgets.
* Implement **self-distillation and teacher-student ensembling** for production models.
* Explore **3D molecular docking simulations** (external tools) for high-fidelity interactions (if you have computational chemistry knowledge).
* Expand RL to a simulated **patient population** with differential rewards.
* Attempt to reproduce a figure/table from a recent SOTA paper and beat it on your public split.

---

# DELIVERY & REPRODUCIBILITY CHECKLIST (final)

* Every experiment: save `experiments/configs/*.json`, `experiments/results/*`, `experiments/logs/*`, `metadata.json` (seed, git commit, pip freeze).
* For each week: `docs/weekX_summary.md`.
* Provide `requirements.txt`, `Dockerfile`, and `scripts/run_in_docker.sh`.
* Include LICENSE, CITATION.cff, and clear README with "Quick start" commands.
* Prepare `technical_report.pdf` and `presentation/slides.pptx` before submission.

---

# FINAL NOTES

* I followed your requested format and included the two Day-1 papers from your original file; see the earlier literature notes and summaries in `docs/lit_reviews/`. 
* This document is expansive: **you can follow it day-by-day** or pick/skip modules depending on time/computation.
* If you want, I will now:

  * Create `DAILY_FULL_PROJECT_PLAN.md` in your repo with this content, **or**
  * Generate the first set of starter files (`src/preprocessing/data_loader.py`, `src/datasets/graph_dataset.py`, `notebooks/01_eda.ipynb`) and the two literature review starter files for the two Day-1 papers.

Tell me which immediate action you want me to take **now** and I will create the files in the repository (or paste their contents into this chat).