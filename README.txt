Industrial Machine Condition Classification

Put student_dataset.csv, test.csv and sample_submission.csv in this folder before running scripts.

Included original experiment scripts: cat.py, cat_xgb.py, class.py, expts.py, oof_blend.py, kaggle_model_experiments.py, train_best_blend_submission.py, blend_oof_experiments.py and condition_eda.py.

Additional runnable scripts: condition_preprocessing.py, gaussian_nb.py, logistic_models.py, discriminant_models.py, decision_tree_model.py, feature_engineered_models.py and softmax_from_scratch.py.

Install dependencies as needed:
pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost lightgbm joblib

Run EDA first:
python condition_eda.py
python condition_preprocessing.py

Then run the model scripts required. Some original experiment scripts may require the same data paths and library versions as the original environment.

Feature engineering in feature_engineered_models.py is an additional candidate experiment. The ratio and interaction features in that file are not claimed to have been used in the winning submission unless you run and validate this script. The original source scripts are preserved as supplied; their recorded metrics and settings should be used for the corresponding original experiments.
