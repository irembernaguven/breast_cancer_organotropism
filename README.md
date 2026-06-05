# 🧬 Breast Cancer Metastasis Organotropism Dashboard

An interactive Streamlit web application designed to explore the clinical and genomic drivers of site-specific breast cancer metastasis (organotropism). This dashboard leverages clinical and genomic sequencing data from the MSK-IMPACT study: **"Metastatic Breast Cancer (MSK, Cancer Discovery 2022)"**.

Users can select a target metastatic site (such as Lung, Liver, Bone, or Brain) to dynamically discover demographic shifts, mutational load differences, and statistically enriched gene mutations compared to other metastatic sites.

---

## 🌟 Features

*   **Dynamic Cohort Partitioning:** Select any metastatic site with a sufficient sample size to partition the cohort into a **Target** group (e.g., Lung Metastasis) and a **Control** group (all other metastatic sites).
*   **Tumor Mutational Burden (TMB) Analysis:** Compares TMB distribution using violin plots and automatically performs a **Mann-Whitney U Test** to determine the statistical significance of mutational load shifts.
*   **Genomic Driver Exploration:** Identifies the top mutated genes within the selected metastatic cohort.
*   **Mutational Enrichment Analysis (Fisher's Exact Test):** Dynamically runs a Fisher's Exact Test on all mutated genes in the dataset to discover which mutations are **significantly enriched** in the selected metastasis site.
*   **Clean and Modern Interface:** Built using Streamlit, featuring sidebar controls, multi-tab layout, and high-quality charts using Seaborn and Matplotlib.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-metastasis-dashboard.git
cd breast-cancer-metastasis-dashboard
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
The raw clinical and genomic data files must be downloaded from cBioPortal due to file size limitations on GitHub:
1. Go to the study summary page on cBioPortal: [https://www.cbioportal.org/study/summary?id=breast_ink4_msk_2021](https://www.cbioportal.org/study/summary?id=breast_ink4_msk_2021).
2. Click the **"Download"** button at the top right of the page to download the study data package (`breast_ink4_msk_2021.tar.gz`).
3. Extract the tarball inside this project folder so that you have:
   * A folder named `breast_ink4_msk_2021/` containing files like `data_mutations.txt`, `data_clinical_patient.txt`, etc.
   * The file `breast_ink4_msk_2021_clinical_data.tsv` (downloaded or extracted).

---

## 🚀 Running the Application

To start the Streamlit local web server, run the following command in your terminal:
```bash
streamlit run app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🔬 Scientific Context
This study cohort focuses on metastatic breast cancer patients sequenced via **MSK-IMPACT** to analyze genomic characteristics. In particular, it investigates the landscape of mutations and Copy Number Alterations (CNA) to understand resistance to therapies (like CDK4/6 inhibitors) and the organ-specific migration patterns of metastatic breast cancer cells.
