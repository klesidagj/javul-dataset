# Code Embedding Visualization & Metrics Pipeline

This project provides a **modular, fully refactored machine-learning pipeline** for:
- Loading embeddings + graphs from a database or CSV  
- Parsing messy input formats (JSON, strings, bytes, escaped sequences)  
- Constructing feature vectors from CSS embeddings + AST/CFG/DFG graph summaries  
- Preprocessing (variance filtering, scaling, jitter, PCA pre-reduction)  
- Dimensionality Reduction (PCA, t-SNE, UMAP)  
- Computing quantitative metrics (silhouette, KNN accuracy, cluster purity)  
- Generating a **3×3 visualization grid** (CSS / Graph / Combined × PCA / t-SNE / UMAP)  
- Exporting embeddings + metrics to CSV and JSON  

It is meant for analyzing **code vulnerability embeddings**, but works for any dataset that includes:
- A vector field (e.g., an embedding list)  
- Three graph fields (AST, CFG, DFG)  
- A class label field (e.g., CWE ID)  
##  Features

###  Modular multi-file architecture  
Each step is isolated into clean modules:

| File | Purpose |
|------|---------|
| `db_loader.py` | Load data from PostgreSQL or CSV |
| `parsers.py` | Parse CSS vectors and AST/CFG/DFG graphs from inconsistent formats |
| `features.py` | Build numeric feature matrices (CSS, Graph, Combined) |
| `preprocess.py` | StandardScaler, zero-variance filter, log transform, PCA pre-reduction |
| `dr.py` | Run PCA, t-SNE, UMAP and compute quality metrics |
| `metrics.py` | Silhouette, KNN accuracy, local agreement, cluster purity |
| `plot.py` | Produce the 3×3 visualization grid |
| `main.py` | Orchestrates the full workflow through CLI |

---

## Installation

### 1. Install dependencies
pip install -r requirements.txt

## How the Pipeline Works

### **1. Load data**
- via CSV (`--csv`) or  
- via PostgreSQL (`--use_db`)

### **2. Parse and clean**
`parsers.py` extracts:
- CSS vector  
- Graph structures (AST/CFG/DFG)  
- Sanitizes broken JSON formats  
- Extracts graph summary metrics (#nodes, #edges, #types)

### **3. Build feature matrices**
`features.py` constructs:
- **CSS** matrix (padded to uniform length)  
- **Graph** matrix (9 features: 3 summary values × 3 graphs)  
- **Combined** matrix (CSS + Graph)

### **4. Preprocess**
- Remove zero-variance columns  
- Log-transform graph features  
- StandardScale  
- Add small jitter  
- PCA reduce to ≤30 dimensions

### **5. Dimensionality Reduction**
`dr.py` computes:
- PCA → 2D  
- t-SNE → 2D  
- UMAP → 2D  

Also computes:
- Silhouette score  
- KNN cross-validated accuracy  

### **6. Plot**
Produces a 3×3 PNG:

| Feature Set | PCA | t-SNE | UMAP |
|-------------|-----|-------|------|
| CSS | ● | ● | ● |
| Graph | ● | ● | ● |
| Combined | ● | ● | ● |

### **7. Export results**
- `embeddings_*.csv` (all 2D coordinates for all methods)  
- `cluster_details_*.json`  
- `plot_*.png`  

## Outputs

Running the code generates:

| File | Description |
|------|-------------|
| `PREFIX_TIMESTAMP.png` | 3×3 projection visualization grid |
| `PREFIX_embeddings_TIMESTAMP.csv` | PCA/t-SNE/UMAP coordinates for all rows |
| `PREFIX_cluster_details_TIMESTAMP.json` | Cluster purity scores per feature set |

---