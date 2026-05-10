# Non-Parametric Estimation: Theoretical Foundations and Practical Applications of k-Nearest Neighbors

This repository provides a comprehensive exploration of Non-Parametric Estimation (NPE) with a deep focus on the k-Nearest Neighbors (k-NN) algorithm. It bridges rigorous statistical theory with state-of-the-art practical applications, demonstrating how localized density estimation models adapt to complex, multi-scale datasets where global bandwidth methods fail.

## Overview

The project is structured into three primary domains:

1. **Theoretical Foundations and Benchmarking:** 
   Interactive and static comparisons of density estimators (Histograms, Kernel Density Estimation, and k-NN). This includes mathematical derivations of bias-variance tradeoffs, Mean Integrated Squared Error (MISE), and the curse of dimensionality.
2. **Applied Computer Vision:**
   A robust, real-time facial recognition pipeline leveraging MTCNN for detection, pretrained FaceNet (InceptionResnetV1) for 512-dimensional feature extraction, and distance-weighted k-NN for classification with an explicit outlier rejection threshold.
3. **Applied Audio Processing:**
   An any-to-any voice conversion system implementing the kNN-VC architecture. The system utilizes self-supervised features from a WavLM-Large encoder and synthesizes converted speech via a prematched HiFi-GAN vocoder, using k-NN regression to seamlessly transfer speaker identity while preserving phonetic content.

## Repository Structure

* `Theoretical/`: Contains mathematical scripts, interactive Plotly visualizations, and native Matplotlib benchmark applications comparing fixed-bandwidth KDE against adaptive k-NN density estimators on synthetic and real-world datasets (e.g., Old Faithful).
* `Applications/`: Houses production-grade Python scripts and self-contained Kaggle notebooks for advanced machine learning tasks.
  * **Face Recognition**: Scripts for enrollment, embedding extraction, and real-time webcam inference.
  * **Voice Conversion**: Modular scripts for feature extraction, k-NN acoustic feature matching, and waveform resynthesis.
* `Plots/`: Generation scripts for publication-quality figures used in the final academic report, analyzing k-NN behavior across varying topologies.
* `Report/`: LaTeX/Typst sources and compiled documents detailing the theoretical derivations (MSE, MISE, elbow method, K-fold cross-validation) and empirical findings.

## Highlights

* **Adaptive Local Bandwidth**: Demonstrated through the "Multi-Scale Challenge," proving k-NN's superiority over global-bandwidth KDE in handling bimodal distributions with vastly different local densities.
* **Production-Grade Implementation**: All applications are written with modularity and efficiency in mind, featuring proper normalization, distance-weighted voting mechanisms, and real-time inference optimizations.
* **Reproducibility**: Contains self-contained Kaggle notebook versions of the applied pipelines, ensuring seamless evaluation in cloud environments without dependency conflicts.

## Requirements

The project relies on specific Conda environments depending on the module:
* `ML`: For theoretical benchmarks (Scipy, Scikit-learn, Plotly, Matplotlib, Seaborn).
* `torch`: For deep learning applications (PyTorch, Torchaudio, Facenet-PyTorch, OpenCV, Soundfile).

Detailed environment specifications and setup instructions are provided within the respective directories.
