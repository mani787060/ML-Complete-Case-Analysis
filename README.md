# Data Cleansing Architectures: Complete Case Analysis (CCA)
[![Machine Learning](https://img.shields.io/badge/Domain-Data%20Engineering-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Strategy-Listwise%20Deletion-red)](https://pandas.pydata.org/docs/user_guide/missing_data.html)
[![Dataset](https://img.shields.io/badge/Dataset-Data%20Science%20Jobs-green)](./data_science_job.csv)

## 🏗️ Project Overview
Missing data is one of the most common challenges in real-world machine learning pipelines. Left unaddressed, null values can break algorithm executions or introduce severe mathematical bias. This repository explores **Complete Case Analysis (CCA)**, also known as listwise deletion. CCA is a straightforward data-cleaning technique where any row containing one or more missing observations is entirely removed from the dataset.

Using the **Data Science Job Dataset** (`data_science_job.csv`), this project investigates the mechanics, statistical assumptions, and operational trade-offs of CCA. It focuses heavily on evaluating how row deletion impacts the underlying data distribution, checking for variance shifts or shape distortions across both categorical and continuous feature spaces.

---

## 🛠️ Advanced Engineering Mechanics

### 1. The Core Statistical Assumption: MCAR
Complete Case Analysis cannot be applied blindly to any dataset. It relies on a strict statistical requirement: the data must be **Missing Completely At Random (MCAR)**.
* **MCAR Defined:** The probability of a data point being missing is completely independent of both the observed data and the unobserved missing values themselves. 
* **The Risk of Violation:** If data is Missing Not At Random (MNAR)—for instance, if professionals with higher salaries choose not to disclose their income—applying CCA will introduce severe selection bias, resulting in a model that fails to generalize to the true population distribution.

### 2. The Extraction & Filtering Blueprint
The filtering pipeline scans the raw data matrix across all features, applies an exact completeness mask to drop sparse rows, checks the remaining distribution shapes, and outputs a dense, fully completed matrix.



```text
                 ┌─────────────────────────────────────┐
                 │       Raw Sparse Data Matrix        │
                 └─────────────────────────────────────┘
                                   │
                                   ▼

                 ┌─────────────────────────────────────┐
                 │      Complete Case Analysis         │
                 │            (CCA Filter)             │
                 └─────────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼

        Missing Values Found                No Missing Values
              Drop Row                         Retain Row

                 └─────────────────┬─────────────────┘
                                   ▼

                 ┌─────────────────────────────────────┐
                 │    Dense Matrix (CCA Verified)      │
                 └─────────────────────────────────────┘
```



---

## 🔬 Implementation Workflows

The notebook `ML-Complete-Case-Analysis.ipynb` executes a structured data-auditing and filtering sequence:

1. **Missing Data Profiling:** Utilizing Pandas `.isnull().mean()` to calculate the exact percentage of missing values across all columns.
2. **The 5% Threshold Rule Baseline:** Identifying target features where missingness drops below the standard **5% threshold**, marking them as safe candidates for complete case extraction without incurring catastrophic data loss.
3. **Isolating Complete Cases:** Applying `.dropna()` configurations localized to specific high-variance features like `experience`, `training_hours`, and `current_salary`.
4. **Distribution Distribution Audit (Crucial Step):** * **Continuous Features:** Overlaying Seaborn Kernel Density Estimate (KDE) plots **Before vs. After** deletion to guarantee that the variance and mean of numeric features remain identical.
   * **Categorical Features:** Generating proportional bar charts to confirm that the ratio of categories (e.g., job types or city codes) was not skewed by dropping records.
5. **Model Evaluation Baseline:** Passing the pristine, clean array into downstream estimators to establish an un-biased predictive baseline.

---

## 📊 Complete Case Analysis Operational Matrix

| Metric Checked | Target Condition for CCA | Post-CCA Verification Metric | Failure Signal / Risk |
| :--- | :--- | :--- | :--- |
| **Missing Ratio** | Under $5\%$ across target columns | Retains $95\%+$ of data volume | Matrix drops too low, causing high model variance |
| **Numeric Shape** | MCAR (Missing Completely at Random) | Identical KDE distribution overlay | Shifting means or compressed standard deviations |
| **Categorical Ratio** | Uniformly distributed missingness | Proportion of classes matches raw data | Specific class blocks vanish, altering model weights |

---

## 💻 Tech Stack & Requirements
* **Language Environment:** Python 3.9+
* **Data Layout Suite:** Pandas, NumPy
* **Diagnostic Plotting Suite:** Matplotlib, Seaborn
* **Downstream Estimators:** Scikit-Learn (`model_selection`, `linear_model`)
* **Workspace Engine:** Jupyter Notebook

---

## 🚀 Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/ML-Complete-Case-Analysis.git](https://github.com/your-username/ML-Complete-Case-Analysis.git)
    cd ML-Complete-Case-Analysis
    ```
2.  **Install Essential Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Execute the Diagnostics Pipeline:**
    ```bash
    jupyter notebook
    ```
    Open `ML-Complete-Case-Analysis.ipynb` to step through the missingness metrics and evaluate the distribution integrity plots.
