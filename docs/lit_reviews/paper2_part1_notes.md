# 🏛️ Model Architectures

The paper reviews a wide array of model classes. These are grouped below by their core architecture.

## 1. Graph-Based Models (GNNs)

- Graph Neural Networks (GNNs): The foundational class of models that operate on molecular graphs, using message passing to update atom (node) states.

- 2D GNNs: Models that process 2D molecular topologies.
	- Graph Convolutional Networks (GCNs)
	- Graph Isomorphism Network (GIN)

- 3D GNNs / Geometric Models: Models that incorporate 3D spatial features like interatomic distances and angles.
	- DimeNet++
	- SphereNet
	- GemNet
	- Uni-Mol: An SE(3)-invariant Transformer that uses 3D data.
	- SE(3)-Transformers
	- E(n)-GNNs

- Knowledge Graph (KG) Models: GNNs augmented with heterogeneous entities (proteins, diseases, etc.) and their relationships.
	- DTD-GNN: A model for drug repurposing using KGs.
	- KPGT (Knowledge-prompted graph transformer): Integrates KGs with SSL.
	- GraIL: A model for link prediction in KGs.

- Other GNN Variants:
	- GROVER: A model integrating GNNs within a Transformer framework.
	- MPG: A framework using multi-level pre-training.
	- ReLMole: Uses GNNs with contrastive learning for generative design.
	- MagGen: A GNN-based generative model for inorganic compounds.
	- ReaKE: A GNN enhanced with reaction knowledge.

## 2. Generative AI Models

- Autoencoders (AEs): Architectures designed to learn compact representations (latent space) via an encoder-decoder structure.
	- Adversarial AEs: Used to generate molecular fingerprints with specific biological properties.
	- 3D Autoencoder (Court et al.): Specifically for generating 3D inorganic crystal structures.

- Variational Autoencoders (VAEs): A probabilistic extension of AEs that learns a distribution in the latent space, enabling generation of new structures.
	- Junction Tree VAE: Combines graph-based encoding with a tree-structured decoder to ensure chemical validity.
	- GraphVAE: Treats molecules as graphs of atoms and bonds.
	- All SMILES VAE: Enables generation of syntactically correct SMILES strings.
	- β-VAE: A variant that learns disentangled representations.
	- InfoVAEs: Focuses on creating informative latent spaces.
	- Conditional VAEs: VAEs that can be conditioned on specific properties.

- Diffusion Models: Generative models that work by iteratively adding noise (forward process) and then learning to reverse the process to generate data (denoising).
	- CDVAE (Crystal Diffusion VAE): A diffusion model for generating periodic 3D crystal structures.
	- Guided Diffusion (Weiss et al.): Facilitates inverse molecular design by guiding the diffusion process with desired properties.
	- Dual-Diffusion Model (Huang et al.): Uses two simultaneous diffusion processes for atomic arrangement and bond connectivity.
	- GCDM (Geometry-complete diffusion models): Designed for 3D molecular generation.
	- DiffBP: A diffusion model that incorporates Bayesian priors for 3D molecular representations.
	- Graph DiT (Graph diffusion transformers): Combines diffusion with GNNs/transformers.
	- Directional Diffusion Models: Apply directed noise to graph representations.
	- SubGDiff: A subgraph diffusion model that diffuses individual molecular substructures.
	- 3M-Diffusion: A latent multi-modal diffusion model.
	- GeoLDM: A geometric latent diffusion model for 3D molecule generation.

- Generative Adversarial Networks (GANs): Models that use two competing networks, a generator and a discriminator, to learn complex data distributions.
	- MolGAN: A GAN that represents molecules as graphs and uses GCNs to generate them, combined with reinforcement learning.
	- Latent GAN (Prykhodko et al.): Employs a GAN to generate latent vectors that are then decoded into molecules.
	- TS-GAN (Transition State GAN): Generates transition state geometries for chemical reactions.
	- Wasserstein GAN (WGAN): Used to generate libraries of human antibody variable regions.

## 3. Transformer-Based Architectures

- Transformers (general): Architectures based on self-attention mechanisms, eliminating reliance on recurrence or convolution.

- Sequence-Based Transformers: Models that operate on linear (string) representations of molecules like SMILES or SELFIES.
	- SMILES-BERT: Adapts BERT for molecular sequences.
	- CHEM-BERT: A transformer model pretrained on SMILES strings.
	- MolBERT: A transformer for SMILES and chemical language.
	- FG-BERT: A self-supervised transformer designed to learn functional group-specific embeddings.
	- KB-BERT (Knowledge-based BERT): Incorporates domain-specific molecular knowledge.
	- SELFormer: Uses SELFIES representations instead of SMILES for robustness.
	- Mole-BERT: Uses domain-aware tokenization and scaffold-level contrastive learning.

- Graph-Based Transformers: Models that integrate transformer architectures with graph-based structural priors.
	- Graphormer: A transformer for molecular graphs that introduces centrality, spatial, and edge encodings.
	- SGT (Structural Graph Transformer): Combines GNNs with transformer attention mechanisms.
	- GMTransformer: A hybrid graph-molecule transformer.
	- MAT (Molecule Attention Transformer): Enhances attention with inter-atomic distances and graph structure.
	- MolFormer: An efficient transformer variant using sparse attention.

## 4. Hybrid and Multi-Modal Models

- Hybrid Models (general): Architectures that combine multiple data modalities (e.g., graphs, sequences, text) or model types.

- Specific Hybrid Architectures:
	- MolFusion: A multimodal model that integrates graph-based encoders and SMILES-based transformer encoders.
	- Multiple SMILES: A model using a CNN-RNN pipeline on augmented (canonical and non-canonical) SMILES strings.
	- UniGraph: A unified cross-domain model leveraging text-attributed graphs.
	- ReactEmbed: A protein-molecule model incorporating biochemical reaction networks.
	- InfoAlign: Embeds cellular response data directly into GNN representations.
	- C-SHAP: A technique for interpretability in multimodal settings.
	- MOL-Mamba: A hybrid model incorporating transparency modules.
	- Auto-Fusion / GAN-Fusion: Adaptive fusion mechanisms.

## 5. Neural Network Potentials (NNPs)

- Neural Network Potentials (NNPs): Models that learn a differentiable energy function (Potential Energy Surface) from 3D molecular geometries.

- Invariant Models: Models whose outputs are unchanged by rotations (e.g., outputting scalar energy).
	- BPNN (Behler-Parrinello): A pioneering NNP framework using handcrafted symmetry functions.
	- SchNet: A continuous-filter convolutional GNN for molecules.
	- PhysNet

- Equivariant Models: Models whose internal features (e.g., vectors) transform consistently with rotations.
	- NequIP: Enforces E(3)-equivariance using tensor-valued features.
	- MACE: Uses higher-body-order interactions for high data efficiency.
	- Allegro: A strictly local architecture that avoids message passing for linear scaling.
	- ViSNet: Integrates scalar-vector interactive message passing.
	- GemNet-OC: A robust model for large-scale catalysis datasets.
	- EquiformerV2: An extremely accurate equivariant model.
	- NewtonNet: Encodes Newtonian force constraints into its rules.
	- TorchMD-NET: An SE(3)-equivariant Transformer.

## 6. Interpretability & Other Models

- Traditional ML (Baselines): Simpler models often used for benchmarking.
	- Random Forests (RF)
	- XGBoost
	- Support Vector Machine (SVM)

- Interpretability-Focused Models/Methods:
	- Attentive FP: A GNN using graph attention for interpretability.
	- GNN Explainer: Identifies relevant subgraphs for a GNN's prediction.
	- Motif-aware Attribute Masking: A pre-training method that masks motifs.
	- FraGAT: A fragment-oriented graph attention network.
	- Motif-based GNN Explainer

---

# 🔬 Pretraining Methods

The paper details numerous self-supervised learning (SSL) and pretraining strategies used to train these architectures on large, unlabeled datasets.

## 1. General Self-Supervised Learning (SSL) Paradigms

- Generative SSL: Models learn by reconstructing molecular inputs from perturbed or augmented versions.
- Contrastive SSL: Models learn by maximizing the agreement between different "views" (augmentations) of the same molecule (positive pairs) while distinguishing them from other molecules (negative pairs).
- Masked Prediction: Models learn by predicting masked-out portions of the input data (atoms, bonds, or functional groups).
- Geometric Self-Supervision: Models use 3D geometric information for pretext tasks, such as predicting interatomic distances or bond angles.

## 2. Specific Pretraining Techniques and Pretext Tasks

- Masked Language Modeling (MLM): Used by sequence-based transformers (e.g., CHEM-BERT) to predict masked tokens in a SMILES string.
- Masked Functional Group Prediction: Used by FG-BERT, where chemically meaningful functional groups are masked and predicted.
- Node/Edge Masking: Used by GROVER, which masks atoms (nodes) and bonds (edges) and predicts their features based on context.

- Multi-Level / Multi-Task SSL:
	- GROVER: Pre-trained on node-level, edge-level, and graph-level tasks simultaneously.
	- MPG: Uses multi-level pre-training to refine node and graph representations.

- Geometric Contrastive Learning:
	- GraphMVP: Uses contrastive learning to align representations from a 3D graph (using 3D conformers) and its corresponding 2D graph.

- Chemically-Informed Contrastive Learning:
	- iMolCLR: Uses Tanimoto similarity to identify and down-weight "faulty negatives" (structurally similar but distinct molecules).
	- ACANET: Uses an "activity cliff aware" contrastive triplet loss to sensitize the model to small structural changes that cause large activity shifts.
	- ReaKE: Employs reaction-aware contrastive learning to capture structural transformations.
	- SMR-DDI: Uses scaffold-aware augmentations for pretraining.
	- Mole-BERT: Uses scaffold-level contrastive learning.

- Knowledge-Guided Pretraining:
	- KPGT: Integrates domain-specific knowledge (e.g., descriptors, semantic substructures) into the graph transformer pretraining.
	- KB-BERT: Incorporates curated molecular annotations and cheminformatics rules.

## 3. Data Augmentation Strategies

- Graph Augmentations (used in MolCLR):
	- Node dropping
	- Edge perturbation
	- Subgraph removal
	- Atom masking / Bond deletion

- Sequence Augmentations:
	- SMILES Enumeration (used in SMICLR, Multiple SMILES): Generating multiple, valid (canonical and non-canonical) SMILES strings for the same molecule.

- Geometric Augmentations (used in GraphMVP):
	- 3D to 2D projection
	- Geometric perturbations

- Chemically Invalid Augmentations: The deliberate use of minor, chemically invalid perturbations (e.g., in SMILES) can surprisingly improve chemical language models by filtering low-quality samples.

---

# 🎯 Key Takeaways

The paper provides a critical synthesis of the field, identifying key findings, persistent challenges, and future directions.

## 1. Key Findings & Comparisons

- Shift in Paradigm: The field has moved from manually engineered descriptors to deep learning models (GNNs, transformers, etc.) that automatically extract features.
- No Free Lunch (Complexity vs. Performance): Increased model complexity does not always guarantee better performance. Simpler, traditional models (like Random Forest or XGBoost) paired with molecular fingerprints can outperform complex deep learning architectures (like 2D GNNs or CHEM-BERT) on certain small, low-complexity, or well-defined benchmark datasets.
- GNNs (2D vs. 3D): 2D GNNs are efficient but fundamentally cannot capture stereochemistry or conformational isomers. 3D GNNs show significant performance gains on tasks where geometry is critical (e.g., quantum chemistry, binding affinity).
- Transformers: Transformers excel at capturing long-range dependencies and global context, often outperforming GNNs in these scenarios.
- Generative Model Trade-offs:
	- VAEs are useful for de novo design but can suffer from posterior collapse and difficulty in disentangling properties.
	- GANs offer fast sampling but are notoriously unstable to train and often suffer from mode collapse (low diversity).
	- Diffusion Models produce high-fidelity and diverse structures but have a very high computational cost for inference (requiring hundreds or thousands of steps).
- NNPs: Equivariant models (NequIP, MACE, Allegro) have achieved state-of-the-art, DFT-level accuracy in force prediction, often with high data efficiency. Local architectures (Allegro) enable linear scaling to millions of atoms.
