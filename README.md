# advanced_dimensionality_reduction
### Youtube video: https://youtu.be/pG4wahlrSvI

### 1. Repository Structure

Image Data Notebook: Dimensionality reduction on datasets like Olivetti Faces or MNIST.
Tabular Data Notebook: Techniques applied to datasets like Iris or medical datasets.
Databricks Implementation Notebook: Dimensionality reduction workflows implemented in Databricks.
ReadMe File: Detailed documentation with commentary on methods, results, comparisons, and visualizations.
Datasets: All required datasets (publicly available, e.g., UCI, Kaggle, or Scikit-learn datasets).

Autoencoders
### 3. Implementation Plan
#### A. Image Data Notebook
Use Olivetti Faces dataset from Scikit-learn or MNIST from Keras.
Apply all techniques above to analyze the dataset.
Visualize results using tools like Matplotlib and Plotly for interactive 2D/3D plots.
Generate insights:
Which technique clusters similar faces better?
Which one preserves global vs. local structures?
#### B. Tabular Data Notebook
Use Iris Dataset or Medical Data (e.g., diabetes dataset from Scikit-learn).
Implement the same dimensionality reduction techniques.
Create plots for 2D/3D data representations.
Insights:
How well do these techniques uncover separable clusters or patterns in medical data?
#### C. Databricks Notebook
Use Databricks to process a larger dataset (e.g., Kaggle medical datasets or high-dimensional tabular data).
Compare processing speed, efficiency, and memory consumption between Databricks and Colab.
### 4. Visualization Tools
Matplotlib & Seaborn: For simple plots.
Plotly: Interactive visualizations.
PairCode UMAP Tool: For UMAP interactive visualization.
Bokeh or Altair: For advanced interactivity.



### Key Features
Extensive Coverage of Techniques
Includes both classic and modern dimensionality reduction methods:

Linear Techniques: PCA (Standard, Randomized, Incremental), Factor Analysis, MDS.
Non-linear Techniques: t-SNE, UMAP, ISOMAP, LLE, Kernel PCA, Autoencoders.
Interactive Visualizations

Use of Plotly, Bokeh, and PairCode UMAP Tool for interactive exploration of results.
2D and 3D plots for better insight into data structures.
Diverse Datasets

Image data: Olivetti Faces, MNIST.
Tabular data: Iris, Diabetes, and additional medical datasets from reputable sources.
Multi-Platform Implementation

Google Colab: Simplified implementation for users to experiment with code.
Databricks: Scalable implementation for large datasets in a cloud-based environment.
### Project Structure

├── Image_Data_Notebook.ipynb       # Dimensionality reduction on image datasets
├── Tabular_Data_Notebook.ipynb     # Dimensionality reduction on tabular datasets
├── Databricks_Notebook.dbc         # Dimensionality reduction implemented on Databricks
├── datasets/                       # Contains datasets used in the project
│   ├── olivetti_faces.csv
│   ├── iris.csv
│   ├── diabetes.csv
│   └── ...
├── results/                        # Contains generated plots and visualizations
├── README.md                       # Project documentation
### Dimensionality Reduction Techniques
#### 1. Linear Dimensionality Reduction
##### Principal Component Analysis (PCA)
Reduces dimensionality by projecting data onto orthogonal axes that explain maximum variance.
Variants:
Randomized PCA: Speeds up computation for large datasets.
Incremental PCA: Processes data in batches, suitable for memory-constrained environments.
##### Factor Analysis
Identifies latent variables that explain data variability.
Often used in psychology and medical research to identify hidden factors.
##### Multi-Dimensional Scaling (MDS)
Preserves pairwise distances in data when reducing dimensionality.
#### 2. Non-linear Dimensionality Reduction
##### t-Distributed Stochastic Neighbor Embedding (t-SNE)
Optimized for visualization, especially in 2D and 3D. Highlights clusters in high-dimensional data.
Suitable for smaller datasets due to computational expense.
##### Uniform Manifold Approximation and Projection (UMAP)
Maintains local and global data structures. Faster and more scalable than t-SNE.
Interactive visualization enhances understanding of clusters.
##### Locally Linear Embedding (LLE)
Captures non-linear structures by preserving local relationships.
Variants:
Standard LLE: Assumes uniform neighborhood size.
Modified LLE: More robust to noise and distortions.
##### ISOMAP
Captures global and non-linear structures using geodesic distances in a lower-dimensional manifold.
##### Kernel PCA
Extends PCA by applying kernel tricks for non-linear dimensionality reduction.
##### Autoencoders
Neural network-based technique. Compresses and reconstructs data while learning non-linear relationships.
### Datasets Used
#### Olivetti Faces Dataset

High-dimensional image dataset for face recognition.
Explores how dimensionality reduction preserves facial features.
#### MNIST Handwritten Digits Dataset

A benchmark dataset for image processing.
Helps visualize digit clusters and overlaps.
#### Iris Dataset

Classic tabular dataset with three classes of flowers.
Highlights separable clusters in reduced dimensions.
#### Diabetes Dataset

Medical dataset for predicting diabetes progression.
Showcases latent patterns in high-dimensional medical data.
#### Other Medical Datasets

Extracted from Kaggle or research papers to add variety.
### Interactive Visualizations
#### t-SNE and UMAP

Clustering in 2D/3D with interactive tools.
Compare how t-SNE and UMAP highlight local/global structures.
#### PairCode UMAP Tool

Provides an interactive interface to experiment with UMAP parameters.
#### Plotly and Bokeh

Enhance interpretability with zoomable, hoverable plots.
### Results and Insights
#### Linear Techniques

PCA is efficient but fails to capture non-linear structures.
Factor Analysis and MDS are suitable for interpretable tabular data.
#### Non-linear Techniques

t-SNE provides better visual clusters but lacks scalability for larger datasets.
UMAP strikes a balance between speed, scalability, and visualization quality.
Autoencoders outperform traditional methods for complex patterns but require significant training time.
#### Scalability Comparison

Google Colab: Works well for moderate-sized datasets with easy setup.
Databricks: Handles large datasets efficiently and integrates with cloud workflows.
### How to Use This Repository
#### Run on Google Colab

Open the Image_Data_Notebook.ipynb or Tabular_Data_Notebook.ipynb.
Install required packages: pip install umap-learn plotly scikit-learn.
#### Run on Databricks

Import the Databricks_Notebook.dbc into your workspace.
Load datasets from the datasets/ folder or your own storage.
#### Explore Results

Visualizations in the results/ folder provide an overview of each method’s performance.
### Future Scope
Extend to time-series datasets using advanced techniques like Dynamic Mode Decomposition.
Combine dimensionality reduction with clustering algorithms (e.g., k-means, DBSCAN).
Integrate real-world datasets (e.g., genomic or financial data) for broader applicability.
### Acknowledgments
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron) for foundational knowledge.

