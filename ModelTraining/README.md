# Multi-View Code Vulnerability Detection Pipeline

## Overview
This project implements a multi-view deep learning pipeline for detecting vulnerabilities in code using:
- AST (Abstract Syntax Tree)
- CFG (Control Flow Graph)
- DFG (Data Flow Graph)
- CSS (Code Semantic Structure vector)

---

## Workflow
1. Extract node types from DB (AST, CFG, DFG)
2. Build vocabularies
3. Define task (binary/multiclass)
4. Build dataset (selectors + transforms ready)
5. Build model (layers + weights initialized)

---

### 1. Node Extraction
- Connect to database
- Extract node types from database

---

### 2. Vocabulary Construction
- Build vocabularies:
  - AST vocab
  - CFG vocab
  - DFG vocab
- Map node types → integer IDs

---

### 3. Task Definition

---

### 4. Dataset Creation
- DatasetFactory creates dataset based on task (binary / multiclass)
- Selector filters database rows and defines labels
- Dataset loads:
  - AST, CFG, DFG graphs
  - CSS vector
  - Label
  
---

### 5. Graph Transformation
Each graph is transformed into tensors:
- Node types → IDs via vocab
- Padding to max_nodes
- Mask creation

---

### 6. Batching (Collate Function)
- DataLoader uses optimized_collate
- Combines samples into batches:
  - Stacks node tensors
  - Stacks masks
  - Keeps edge lists
  - Stacks CSS + labels

---

### 5. Model Architecture

#### Embedding Layer
- Converts node IDs → dense vectors (d_model)
---

#### Multi-View Attention
- Separate self-attention for AST, CFG, DFG, CSS
- Produces one vector per view
- Fuses all views into a single representation

---

#### Classification Head
- Fully connected layers
- Outputs logits for classes

---

### 6. Training Pipeline
- Forward pass
- Loss computation
- Backpropagation
- Optimization
- Early stopping (patience)

---

### 7. Inference
- Runs trained model on dataset
- Computes accuracy

---

## Summary
Pipeline:
DB → Vocab → Dataset → Transform → Batch → Model → Train → Inference


##  ARCHITECTURE DIAGRAM 


                 ┌────────────────────┐
                 │     DATABASE       │
                 └────────┬───────────┘
                          ↓
              Extract Node Types
                          ↓
                   Build Vocab
                          ↓
                   Build Dataset
                          ↓
                   Build Model
              (init layers + weights)
                          ↓
                  TRAINING LOOP
                          ↓
        ┌────────────────────────────────┐
        │        DataLoader              │
        │ dataset → transform → collate │
        └──────────────┬─────────────────┘
                       ↓
               Batched Tensors
                       ↓
                   MODEL
                       ↓
        ┌──────────────────────────────┐
        │ Embedding Layer              │
        │ (IDs → 768-dim vectors)      │
        └──────────────┬───────────────┘
                       ↓
        ┌──────────────────────────────┐
        │ Per-View Attention           │
        │ AST | CFG | DFG | CSS        │
        └──────────────┬───────────────┘
                       ↓
        ┌──────────────────────────────┐
        │ Pooling (mean + masking)     │
        └──────────────┬───────────────┘
                       ↓
        ┌──────────────────────────────┐
        │ Fusion (concat + MLP)        │
        └──────────────┬───────────────┘
                       ↓
        ┌──────────────────────────────┐
        │ Classification Head          │
        └──────────────┬───────────────┘
                       ↓
                     Logits

## TIMELINE DIAGRAM 

---
[OFFLINE PHASE]
DB Access → Node Extraction → Vocabulary Build → Dataset Build

[MODEL INIT]
Layers Created → Weights Initialized

[TRAINING PREP]
Split Dataset → Compute Class Weights → Build Loss
→ Optimizer Init → Scheduler Init

---

[TRAINING LOOP START]

Epoch 1
   Batch 1:
      Transform → Collate → Embedding → Attention
      → Fusion → Loss → Backprop → Update

   Batch 2:
      Transform → Collate → Embedding → Attention
      → Fusion → Loss → Backprop → Update
   ...

   Validation → Scheduler → Checkpoint

Epoch 2
   (same pattern repeats)

---
