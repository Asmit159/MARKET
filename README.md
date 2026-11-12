# Retail Market Zoning & Supply Chain Optimization

This machine learning pipeline acts as an automated supply chain manager. It analyzes historical inventory data, sales velocity, and transit times to group warehouses into distinct "market zones" using unsupervised clustering. A highly optimized supervised classifier is then trained on these zones, enabling an ultra-fast, lightweight API to instantly classify new inventory updates into the correct operational zone without needing to recalculate complex historical metrics.

---

##  Two-Phase Architecture

This project solves a major production bottleneck by splitting the workload into two distinct phases:

1. **Phase 1: Knowledge Discovery (K-Means)**
Analyzes all 5 complex KPI features (including mathematically derived storage costs and maximum-capped days of supply) to discover the ground-truth operational zones historically.
2. **Phase 2: Real-Time Production API (XGBoost)**
Acts as the "student" model. It is trained to predict the K-Means zones using only **3 raw, lightweight features** (Inventory, Sales, Lead Time). This eliminates data leakage and ensures the model can process real-time API payloads in milliseconds.

---

##  Phase 1: Unsupervised Zone Discovery

To ensure robust cluster formation, extreme supply chain outliers (like items with over 1 million days of supply) were handled by capping values at 365 days and applying a `PowerTransformer` before PCA dimensionality reduction.

### Mathematical Validation (Elbow & Silhouette)

The K-Means clustering was evaluated across 1 to 10 clusters to mathematically justify the operational zones.

### Cluster Profile Breakdown

The algorithm segmented the warehouses into 4 distinct business states:

* **Zone 0 (Severe Overstock):** Highest average inventory (606.5 units), highest days of supply (62.4 days), and highest storage costs.


* **Zone 1 (High Shortage Risk):** Lowest average inventory (337.0 units) combined with the highest sales velocity (27.1 units/day), resulting in a critically low 12.3 days of supply.


* **Zone 2 (Optimized Equilibrium):** High sales (26.4 units/day) supported by healthy inventory buffers (543.2 units).


* **Zone 3 (Stagnant Market):** Low sales velocity (12.4 units/day) with moderate inventory (393.3 units).



### 3D PCA Market Distribution

Plotting the first three principal components visualizes the clear geometric boundaries the K-Means algorithm established for the training data.

---

##  Phase 2: Supervised Production Classifier

The XGBoost classifier was trained exclusively on raw metrics, preventing it from flawlessly reverse-engineering the K-Means mathematical boundaries.

### Learning Curve & Feature Importance

The model was restricted to a max depth of 5 with 100 boosting rounds to prevent overfitting. The log loss curve demonstrates stable convergence between the training and unseen test sets, while feature importance tracking reveals that `Inventory_Level` (score: 36,168) drives the majority of the decision-making, followed by `Sales_Velocity` and `Lead_Time_Days`.

### Evaluation Metrics

By hiding the derived KPIs, the model achieves a highly realistic and production-ready **88.99% accuracy on strictly unseen data**. Precision scores remain remarkably balanced across all classes, ranging from 0.87 to 0.92.

The confusion matrix confirms strong alignment with the original K-Means labels, with only realistic misclassifications occurring at the complex boundary edges.

---

##  API Batch Inference Simulation

In a live production environment, the model ingests a lightweight payload and instantly outputs the predicted supply chain zone alongside the model's prediction confidence probability. In this inference sample, the model successfully routed all incoming feeds with 99.6% to 100.0% confidence.

---

**Author:** Asmit Mandal ([@Asmit159](https://github.com/Asmit159))
