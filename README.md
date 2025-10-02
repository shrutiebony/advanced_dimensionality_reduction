# advanced_dimensionality_reduction
### Youtube video: https://youtu.be/pG4wahlrSvI

### Advanced Dimensionality Reduction
This repository provides a comprehensive exploration of dimensionality reduction techniques, applied to both image and tabular datasets, with implementations across Google Colab and Databricks. It is designed as a practical reference for data scientists, researchers, and students who want to compare linear and non-linear dimensionality reduction methods, evaluate their effectiveness on diverse datasets, and generate insightful visualizations.
The project covers classic algorithms (e.g., PCA, Factor Analysis, MDS) alongside modern methods (e.g., t-SNE, UMAP, Autoencoders), and includes interactive visualizations to enhance interpretability.
### Key Features
#### Extensive Coverage of Techniques
##### Linear Methods
1) Principal Component Analysis (PCA): Standard, Randomized, Incremental.
2) Factor Analysis.
3) Multi-Dimensional Scaling (MDS).
##### Non-Linear Methods
1) t-Distributed Stochastic Neighbor Embedding (t-SNE).
2) Uniform Manifold Approximation and Projection (UMAP).
3) Locally Linear Embedding (LLE): Standard and Modified variants.
4) ISOMAP.
5) Kernel PCA.
6) Autoencoders (deep learning–based dimensionality reduction).
#### Datasets
###### Image Data:
1) Olivetti Faces dataset (Scikit-learn).
2) MNIST handwritten digits (Keras).
###### Tabular Data:
1) Iris dataset (Scikit-learn).
2) Diabetes dataset (Scikit-learn).
3) Additional public medical datasets from Kaggle and UCI repositories.

These datasets allow evaluation of how dimensionality reduction techniques behave across high-dimensional image data and structured tabular data.
#### Visualization Tools
1) Matplotlib & Seaborn – For foundational plots.
2) Plotly – Interactive 2D/3D scatter plots with hover/zoom capabilities.
3) PairCode UMAP Tool – Interactive exploration of UMAP parameters.
4) Bokeh & Altair – Advanced interactivity for in-depth exploration.
#### Multi-Platform Implementation
1) Google Colab: Easy-to-use implementation environment for experimentation.
2) Databricks: Scalable workflows for large, high-dimensional datasets.

### Implementation Plan
#### A. Image Data Notebook
Dataset: Olivetti Faces (Scikit-learn) or MNIST (Keras).
Techniques: Apply all supported dimensionality reduction methods.
Visualization: Interactive 2D and 3D embeddings via Plotly and Matplotlib.
Insights:
1) Which technique clusters similar faces or digits most effectively?
2) Which methods preserve global structures versus local neighborhoods?

#### B. Tabular Data Notebook
Dataset: Iris and medical datasets (e.g., diabetes).
Techniques: Same set of linear and non-linear approaches.
Visualization: 2D/3D scatter plots to highlight latent clusters.
Insights:
1) How separable are classes in reduced dimensions?
2) Which methods are best suited for medical data interpretation?

#### C. Databricks Notebook
Dataset: Larger medical/tabular datasets (e.g., Kaggle).
Techniques: Scalable dimensionality reduction workflows.
Focus: Processing speed, memory efficiency, and scalability.
Comparison: Google Colab vs. Databricks performance on large-scale data.
Dimensionality Reduction Techniques – Overview
##### 1. Linear Approaches
PCA: Projects data onto orthogonal axes with maximum variance.
1) Randomized PCA for faster computation.
2) Incremental PCA for memory-constrained scenarios.
Factor Analysis: Identifies latent variables; popular in psychology/medicine.
MDS: Preserves pairwise distances when embedding into lower dimensions.

##### 2. Non-Linear Approaches
t-SNE: Optimized for visualization; highlights clusters but computationally heavy.
UMAP: Preserves both local and global structures, scalable and fast.
LLE: Maintains local neighborhood relationships.
ISOMAP: Captures geodesic (global) structure in data.
Kernel PCA: Non-linear extensions of PCA using kernel tricks.
Autoencoders: Neural network–based approach for compressing and reconstructing complex data.

##### Results and Insights
###### Linear Methods:
1) PCA is efficient but may fail to capture non-linear structures.
2) Factor Analysis and MDS provide interpretable outputs for smaller datasets.

###### Non-Linear Methods:
1) t-SNE creates visually distinct clusters but struggles with scalability.
2) UMAP strikes a strong balance between scalability, speed, and clarity.
3) Autoencoders reveal deep non-linear patterns but require heavy training.

###### Platform Scalability:
1) Google Colab: Excellent for moderate datasets and prototyping.
2) atabricks: Handles large-scale, high-dimensional datasets effectively with cloud integration.

#### How to Use
Run on Google Colab
Open Image_Data_Notebook.ipynb or Tabular_Data_Notebook.ipynb.
Install dependencies
Execute notebooks to reproduce visualizations and insights.

#### Run on Databricks
Import Databricks_Notebook.dbc into your workspace.
Load datasets from the datasets/ folder or provide your own.
Compare runtime and performance against Colab workflows.

#### Explore Results
The results/ directory contains generated plots and interactive visualizations.
Compare how each method captures structures across datasets.

#### Future Scope
Extend methods to time-series data (e.g., Dynamic Mode Decomposition).
Combine dimensionality reduction with clustering algorithms (k-means, DBSCAN).
Integrate real-world high-dimensional datasets (genomics, financial data).

### Acknowledgments
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron.
Public datasets from Scikit-learn, Kaggle, and UCI ML Repository.
