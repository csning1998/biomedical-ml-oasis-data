# Assessing the Impact of Pseudo-RGB Input Enhancement on CNN Performance for Alzheimer's Disease MRI Classification

This is the Semester Assignment of “Machine Learning for Biomedicine” Lectures.

## Before Starting

### Disclaimer

The content and results presented in this presentation constitute an exploratory research study whose findings have not yet undergone complete peer review.

This research is intended for academic discussion only and does not constitute any form of clinical diagnosis or medical advice.

### Acknowledgments

The author gratefully acknowledges the following resources:

-   The dataset used in this study is provided by the OASIS dataset from [OASIS Alzheimer's Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/code) on Kaggle.

### Operating System

-   **OS**: Red Hat Enterprise Linux 10
-   **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8GB GDDR6 VRAM)
-   **RAM**: 64GB DDR5-5600
-   **Environment**: VSCode with Jupyter Notebook support in Miniconda.(Or if you prefer `vim` btw.)

## Set up

### Environment for Linux

1. If you use Pyenv (via pip):

    1. Install Python and create virtualenv

        ```bash
        pyenv install 3.12.12
        pyenv virtualenv 3.12.12 ml-for-biomedicine
        pyenv activate ml-for-biomedicine
        ```

    2. Install dependencies (Essential Step)

        > Note: Use quotes for 'tensorflow[and-cuda]' to avoid shell errors in zsh.

        ```bash
        pip install numpy pandas opencv-python "tensorflow[and-cuda]" matplotlib seaborn scikit-learn ipykernel
        ```

2. If you use Conda:

    ```bash
    conda env create -f environment-20251208.yaml
    conda activate ml-for-biomedicine
    ```

3. Evaluate if Tensorflow can use GPU

    ```bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

    If it returns an empty list, it's necessary to resolve the dependency issues of CUDA and cuDNN version corresponding to your NVIDIA GPU.

### Data

1. Download the dataset from [OASIS Alzheimer's Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data) on Kaggle. If using the Kaggle CLI, run the following command:

    ```bash
    kaggle datasets download ninadaithal/imagesoasis
    ```

2. Unzip the dataset and place it in the `Data` directory.

    ```bash
    unzip archive.zip
    ```

3. Establish the root directory of the dataset.

    ```bash
    mv Data oasis_data
    mv oasis_data/"Mild Dementia" oasis_data/dementia_mild
    mv oasis_data/"Moderate Dementia" oasis_data/dementia_moderate
    mv oasis_data/"Very mild Dementia" oasis_data/dementia_very_mild
    mv oasis_data/"Non Demented" oasis_data/non-demented
    ```

## Methodology

### Background

Wen et al. (2020) pointed out that the medical imaging field widely suffers from accuracy inflation caused by the absence of 'Subject-level separation'. This study uses the publicly available Kaggle OASIS dataset as a case study, first **quantitatively reproducing** this inflation phenomenon (Naïve Baseline), then establishing a **Rigorous Baseline** that conforms to statistical independence, and finally evaluating the benefits of Pseudo-RGB on this foundation.

### Research Questions

1. To what extent does the violation of subject-level independence (data leakage) inflate the classification performance on the Kaggle OASIS dataset compared to a statistically rigorous split?
2. Can the proposed "Pseudo-RGB" enhancement technique yield statistically significant improvements when applied to the leakage-corrected, rigorous baseline established in RQ1?

### Information from the Kaggle Page

-   The dataset used is the [OASIS MRI dataset](https://sites.wustl.edu/oasisbrains/), which consists of 80,000 brain MRI images. The images have been divided into four classes based on Alzheimer's progression. The dataset aims to provide a valuable resource for analyzing and detecting early signs of Alzheimer's disease.
-   “Data were provided 1-12 by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382”
-   Citation:

    [Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults. _Journal of Cognitive Neuroscience_, 19(9), 1498–1507. https://doi.org/10.1162/jocn.2007.19.9.1498](https://doi.org/10.1162/jocn.2007.19.9.1498)

### Methodological Statement

All references cited in this presentation have been formatted to the best of author’s ability following the APA 7th Edition guidelines.

To ensure academic rigor, the sources cited in this study followed by the priority:

-   Peer-reviewed academic journal articles.
-   Credible academic monographs or textbooks.
-   Reports and technical articles published by authoritative institutions or academic conferences.
-   Relevant online articles in the field.
