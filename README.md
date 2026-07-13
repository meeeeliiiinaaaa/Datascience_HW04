# Datascience_HW04

In this homework we explore advanced machine learning workflows, including data preprocessing, imbalanced learning, generative models, and explainable AI techniques.

### Part 1: Machine Learning Pipelines
- Built end-to-end ML pipelines for data loading, cleaning, preprocessing, and classification.
- Implemented custom data validation and cleaning functions using `pandas.pipe`.
- Created Scikit-learn pipelines with:
  - Feature scaling
  - Categorical feature encoding
  - Missing value imputation
- Compared multiple imputation strategies:
  - Mean
  - Median
  - KNN
  - Iterative Imputation
- Integrated classifiers into complete pipelines and implemented model saving/loading.

### Part 2: Handling Imbalanced Data
Applied multiple strategies to address class imbalance:
- **Random Undersampling**
  - Tested different undersampling approaches and evaluated performance impact.
- **Random Oversampling**
  - Increased minority class representation and compared results with undersampling.
- **SMOTE**
  - Generated synthetic minority samples and compared against traditional oversampling.
- **Cost-Sensitive Learning**
  - Applied class weights and analyzed performance differences.

### Part 3: Variational Autoencoders (VAEs)
- Implemented and trained a Variational Autoencoder using an image dataset.
- Investigated the impact of different KL-divergence weights (β-VAE).
- Analyzed:
  - Latent space disentanglement
  - Reconstruction quality
  - Generated image quality
- Performed latent space interpolation and sample generation.

### Part 4: Generative Adversarial Networks (GANs)
Implemented a GAN on the MNIST dataset:
- Designed Generator and Discriminator networks.
- Built the GAN training loop with appropriate loss functions.
- Generated and visualized samples throughout training.
- Explored GAN limitations, including:
  - Causes of mode collapse
  - Why mode collapse is possible but not guaranteed.

### Part 5: Diffusion Models
- Implemented a Denoising Diffusion Probabilistic Model (DDPM) for image generation.
- Experimented with different noise schedules:
  - Linear schedule
  - Cosine schedule
- Compared diffusion models and GANs based on:
  - Image quality
  - Diversity
  - Training stability

### Part 6: Explainable AI (XAI)
Analyzed CNN model behavior using multiple explainability methods:
- **Grad-CAM**
  - Generated heatmaps for correct and incorrect predictions.
  - Investigated important image regions influencing predictions.
- **Model-Agnostic Explanations**
  - Implemented:
    - SHAP
    - LIME
    - ELI5
  - Produced:
    - Summary plots
    - Waterfall plots
    - Superpixel explanations
    - Feature importance visualizations
- Conducted detailed analysis of misclassified samples, including:
  - Model focus areas
  - Reasons for prediction errors
  - Potential training improvements

### Part 7: Full End-to-End Workflow
Combined all techniques into a complete ML workflow:

**Data Cleaning → ML Pipeline → Imbalance Handling → Generative Modeling → Explainable AI**

Evaluated improvements, trade-offs, and limitations at each stage.
