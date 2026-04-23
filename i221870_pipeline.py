import os
import kfp
from kfp import dsl
from kfp.client import Client

# -------------------------------------------------------------------------
# COMPONENT 1: Data Ingestion
# -------------------------------------------------------------------------
@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'scikit-learn'])
def data_ingestion(output_data: dsl.Output[dsl.Dataset]):
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer

    # Load the dataset from sklearn which mirrors the requested UCI/Kaggle dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Map the target back to 'M' and 'B' to mimic the raw Kaggle dataset for preprocessing
    df['diagnosis'] = np.where(data.target == 0, 'M', 'B')
    
    # Save to the output dataset path
    df.to_csv(output_data.path, index=False)
    print(f"Dataset ingested with shape {df.shape}")

# -------------------------------------------------------------------------
# COMPONENT 2: Data Preprocessing
# -------------------------------------------------------------------------
@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'scikit-learn'])
def data_preprocessing(
    input_data: dsl.Input[dsl.Dataset],
    scaler_type: str,
    random_seed: int,
    X_train_out: dsl.Output[dsl.Dataset],
    X_test_out: dsl.Output[dsl.Dataset],
    y_train_out: dsl.Output[dsl.Dataset],
    y_test_out: dsl.Output[dsl.Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    df = pd.read_csv(input_data.path)
    
    # Map M->1, B->0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Scaling
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Save splits
    X_train_scaled.to_csv(X_train_out.path, index=False)
    X_test_scaled.to_csv(X_test_out.path, index=False)
    y_train.to_csv(y_train_out.path, index=False)
    y_test.to_csv(y_test_out.path, index=False)
    print("Data preprocessing completed successfully.")

# -------------------------------------------------------------------------
# COMPONENT 3: Model Training
# -------------------------------------------------------------------------
@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'scikit-learn', 'joblib'])
def model_training(
    X_train_in: dsl.Input[dsl.Dataset],
    y_train_in: dsl.Input[dsl.Dataset],
    model_type: str,
    feature_selection: str,
    random_seed: int,
    model_out: dsl.Output[dsl.Model]
):
    import pandas as pd
    import joblib
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.feature_selection import SelectFromModel

    X_train = pd.read_csv(X_train_in.path)
    y_train = pd.read_csv(y_train_in.path).squeeze()

    # Apply Feature Selection if requested
    if feature_selection == 'SelectFromModel':
        selector = SelectFromModel(RandomForestClassifier(random_state=random_seed))
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)

    # Optimization (Hyperparameter Tuning via RandomizedSearchCV)
    if model_type == 'SVM':
        base_model = SVC(random_state=random_seed)
        param_dist = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_type == 'RandomForest':
        base_model = RandomForestClassifier(random_state=random_seed)
        param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    elif model_type == 'NeuralNetwork':
        base_model = MLPClassifier(random_state=random_seed, max_iter=500)
        param_dist = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001]}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Optimize using RandomizedSearchCV
    clf = RandomizedSearchCV(base_model, param_dist, n_iter=3, cv=3, random_state=random_seed, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Save model and optionally selector
    model_bundle = {
        'model': clf.best_estimator_,
        'feature_selection': feature_selection,
        'selector': selector if feature_selection == 'SelectFromModel' else None
    }
    
    joblib.dump(model_bundle, model_out.path)
    print(f"Model {model_type} trained with best params: {clf.best_params_}")

# -------------------------------------------------------------------------
# COMPONENT 4: Model Evaluation
# -------------------------------------------------------------------------
@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'scikit-learn', 'joblib'])
def model_evaluation(
    X_test_in: dsl.Input[dsl.Dataset],
    y_test_in: dsl.Input[dsl.Dataset],
    model_in: dsl.Input[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
    classification_metrics: dsl.Output[dsl.ClassificationMetrics]
):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    X_test = pd.read_csv(X_test_in.path)
    y_test = pd.read_csv(y_test_in.path).squeeze()

    # Load model
    model_bundle = joblib.load(model_in.path)
    model = model_bundle['model']
    selector = model_bundle.get('selector')

    # Apply same feature selection transform if it was used
    if model_bundle['feature_selection'] == 'SelectFromModel' and selector is not None:
        X_test = selector.transform(X_test)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log KFP scalar metrics
    metrics.log_metric("Accuracy", float(acc))
    metrics.log_metric("Precision", float(prec))
    metrics.log_metric("Recall", float(rec))
    metrics.log_metric("F1-score", float(f1))

    # Log Confusion Matrix as an Artifact Visualization
    cm = confusion_matrix(y_test, y_pred)
    classification_metrics.log_confusion_matrix(
        ['Benign', 'Malignant'],
        cm.tolist()
    )

    print(f"Evaluation complete. Accuracy: {acc:.4f}, F1: {f1:.4f}")

# -------------------------------------------------------------------------
# KFP PIPELINE DEFINITION
# -------------------------------------------------------------------------
@dsl.pipeline(
    name='breast-cancer-pipeline',
    description='An end-to-end parameterizable ML pipeline for Breast Cancer classification.'
)
def breast_cancer_pipeline(
    model_type: str = 'SVM',
    scaler_type: str = 'StandardScaler',
    feature_selection: str = 'None',
    random_seed: int = 42
):
    # Step 1: Ingest Data
    ingestion_task = data_ingestion()

    # Step 2: Preprocess Data
    preprocessing_task = data_preprocessing(
        input_data=ingestion_task.outputs['output_data'],
        scaler_type=scaler_type,
        random_seed=random_seed
    )

    # Step 3: Train Model
    training_task = model_training(
        X_train_in=preprocessing_task.outputs['X_train_out'],
        y_train_in=preprocessing_task.outputs['y_train_out'],
        model_type=model_type,
        feature_selection=feature_selection,
        random_seed=random_seed
    )

    # Step 4: Evaluate Model
    eval_task = model_evaluation(
        X_test_in=preprocessing_task.outputs['X_test_out'],
        y_test_in=preprocessing_task.outputs['y_test_out'],
        model_in=training_task.outputs['model_out']
    )

# -------------------------------------------------------------------------
# EXECUTION (Compilation and Submission)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    from kfp import compiler
    
    # 1. Compile the pipeline to YAML
    yaml_path = 'breast_cancer_pipeline_v1.yaml'
    compiler.Compiler().compile(
        pipeline_func=breast_cancer_pipeline,
        package_path=yaml_path
    )
    print(f"Pipeline compiled successfully to {yaml_path}")

    # 2. Setup Client
    # Note: KFP UI port-forwarded to localhost:8080
    try:
        client = Client(host='http://localhost:8080')
        print("Connected to Kubeflow Pipelines cluster.")

        # Baseline Runs (Task 3)
        baseline_runs = [
            {'model_type': 'SVM', 'feature_selection': 'None', 'random_seed': 42},
            {'model_type': 'RandomForest', 'feature_selection': 'SelectFromModel', 'random_seed': 42},
            {'model_type': 'NeuralNetwork', 'feature_selection': 'None', 'random_seed': 42}
        ]

        print("Submitting Baseline Runs...")
        for exp in baseline_runs:
            run_name = f"Baseline-{exp['model_type']}"
            client.create_run_from_pipeline_func(
                breast_cancer_pipeline,
                arguments=exp,
                experiment_name='Breast Cancer Baselines',
                run_name=run_name
            )

        # Optimization Study Run (Task 6)
        opt_run = {'model_type': 'SVM', 'scaler_type': 'MinMaxScaler', 'feature_selection': 'None', 'random_seed': 42}
        client.create_run_from_pipeline_func(
            breast_cancer_pipeline,
            arguments=opt_run,
            experiment_name='Breast Cancer Optimization',
            run_name="Optimization-SVM-MinMax"
        )

        # Reproducibility Analysis Run (Task 7)
        rep_run = {'model_type': 'SVM', 'feature_selection': 'None', 'random_seed': 999}
        client.create_run_from_pipeline_func(
            breast_cancer_pipeline,
            arguments=rep_run,
            experiment_name='Breast Cancer Reproducibility',
            run_name="Reproducibility-SVM-Seed999"
        )

        print("All pipeline runs submitted to Kubeflow successfully! You can view them on the Dashboard at http://localhost:8080.")
    except Exception as e:
        print(f"Could not connect to Kubeflow Client: {e}")
        print("Please ensure you have run: kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow")
