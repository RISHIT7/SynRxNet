# DeepSynergy: Predicting Anti-Cancer Drug Synergy with Deep Learning

---

## 📄 Paper Summary

**DeepSynergy** introduces a *deep feed-forward neural network* that predicts **synergistic anti-cancer drug combinations** by jointly modeling molecular descriptors of paired compounds and genomic profiles of cancer cell lines. Trained on a large high-throughput combination screen, the model encodes each sample as concatenated drug fingerprints, physico-chemical and toxicophore features alongside filtered gene-expression signatures, uses *order-invariant data augmentation* to treat drug pairs symmetrically, and is tuned with systematic hyperparameter search and regularization to minimize prediction error. 

By learning complex, non-linear interactions between **chemical** and **biological** modalities, DeepSynergy outperforms classical machine-learning baselines on held-out combinations and cell lines, offering a scalable, data-driven framework to prioritize candidate drug pairs for experimental follow-up and accelerate discovery of precision combination therapies.

---

### 🔑 Key Highlights

- **Architecture**: Multi-layer feed-forward network with dropout regularization
- **Input Features**: 
  - 🧬 Gene expression profiles (3,984 features)
  - 💊 Chemical descriptors: ECFP_6 fingerprints, physico-chemical properties, toxicophores
- **Training Data**: 23,062 samples from Merck's large-scale oncology screen
- **Innovation**: Order-invariant modeling ensures A+B = B+A predictions
- **Performance**: Superior to traditional ML approaches on unseen drug pairs and cell lines

---

> *"A scalable, data-driven framework to prioritize candidate drug pairs for experimental follow-up"*