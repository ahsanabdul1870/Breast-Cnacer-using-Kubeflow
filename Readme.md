# Kubeflow Machine Learning Workflows: Assignment 3 Report

## 1. Introduction & Pipeline Design (Tasks 1 & 2)
The objective of this project is to construct automated, scalable, and reproducible machine learning workflows using Kubeflow Pipelines (KFP). The pipeline is designed around the Breast Cancer Wisconsin dataset and structured into four modular components using the modern KFP V2 SDK:

1. **Data Ingestion:** Loads the dataset dynamically, maintaining the original M/B diagnosis format to mirror raw CSV inputs.
2. **Data Preprocessing:** Encodes the target variable (Malignant=1, Benign=0), splits the data into 80/20 train/test sets, and dynamically applies either standard or min/max scaling based on pipeline parameters.
3. **Model Training:** Accepts parameterized configurations to train Support Vector Machines (SVM), Random Forests (RF), or Multi-Layer Perceptrons (Neural Networks). It executes hyperparameter optimization using `RandomizedSearchCV` to represent metaheuristics and optimization algorithms.
4. **Model Evaluation:** Evaluates the serialized model against the test set, logging Accuracy, Precision, Recall, F1-score, and a visual Confusion Matrix natively to the Kubeflow UI via the `ClassificationMetrics` artifact.

## 2. Experimental Design (Tasks 3 & 4)
The pipeline is fully parameterized, allowing external arguments to dictate `model_type`, `scaler_type`, `feature_selection`, and `random_seed`. Three baseline models were executed simultaneously to establish a foundation:
- **Baseline-SVM:** Standard Scaling, No Feature Selection, Seed 42.
- **Baseline-RandomForest:** Standard Scaling, `SelectFromModel` Feature Selection, Seed 42.
- **Baseline-NeuralNetwork:** Standard Scaling, No Feature Selection, Seed 42.

## 3. Results (Task 5)
The extracted metrics from the Kubeflow dashboard are summarized below:

| Model / Configuration | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline-SVM** | **0.9824** | **1.0000** | 0.9535 | **0.9762** |
| **Baseline-RandomForest** | 0.9561 | 0.9524 | 0.9302 | 0.9412 |
| **Baseline-NeuralNetwork** | 0.9737 | 0.9762 | 0.9535 | 0.9647 |
| **Optimization-SVM-MinMax** | 0.9737 | 0.9762 | 0.9535 | 0.9647 |
| **Reproducibility-SVM-Seed999** | 0.9474 | 0.9592 | 0.9216 | 0.9400 |

## 4. Pipeline Optimization Analysis (Task 6)
To investigate how different pipeline configurations impact results, an optimization experiment was executed on the best-performing baseline model (SVM). The scaler was modified from `StandardScaler` to `MinMaxScaler`. 

**Observations:** 
The MinMax scaler resulted in a drop in Accuracy (from 0.9824 to 0.9737) and F1-score (from 0.9762 to 0.9647). This indicates that the SVM, likely utilizing an RBF kernel through the `RandomizedSearchCV`, benefited more from features being centered around a zero mean via Standard Scaling rather than being bound strictly between 0 and 1. Furthermore, comparing the Baseline-RandomForest (which utilized feature selection) against the others reveals that forcibly reducing the feature space was actually detrimental compared to allowing the SVM or NN to process all original features.

## 5. Reproducibility Analysis (Task 7)
To evaluate pipeline determinism, the Baseline-SVM pipeline was duplicated but executed with a `random_seed` of `999` instead of `42`. Because the seed was exposed as a pipeline parameter, this altered both the 80/20 train/test split inside the preprocessing component and the internal initialization of the cross-validation in the training component.

**Observations:** 
The results exhibited significant variance. The Accuracy plummeted from a near-perfect 0.9824 down to 0.9474, and the F1-score dropped to 0.9400. This reveals that the pipeline, while structurally identical, is highly sensitive to the data split. Without rigidly tracking random seeds as pipeline parameters—which KFP easily facilitates—experimental results cannot be strictly trusted as deterministic, highlighting exactly how pipeline automation impacts reproducibility.

## 6. Scalable Deployment Simulation (Tasks 8 & 9)
The final execution Python script simulated a scalable deployment by firing 5 massive, cross-validated training workloads concurrently to the Kubeflow cluster. 

**Observations:** 
The concurrent execution yielded the following durations for each pipeline run:
- **Optimization-SVM-MinMax**: 14m 40s
- **Reproducibility-SVM-Seed999**: 14m 11s
- **Baseline-RandomForest**: 10m 12s
- **Baseline-NeuralNetwork**: 9m 40s
- **Baseline-SVM**: 9m 17s

By utilizing Kubeflow's Kubernetes/Argo orchestration backbone, the individual pipeline pods (`data-ingestion`, `model-training`, etc.) were dynamically scheduled across the cluster in parallel. Instead of executing sequentially (which would have taken nearly 58 minutes total), the cluster utilized available compute resources to process the pipelines concurrently. This drastically increased execution efficiency and minimized the total wall-clock time for the experiment suite. KFP successfully managed the execution states, tracked artifacts, and cleanly separated the runs under distinct Experiment umbrellas for rapid side-by-side comparison.
