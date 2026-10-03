"""Independent post-execution checks using only saved files and raw CSV inputs."""
from pathlib import Path
import json
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from predict import predict_borrowers
from models.feature_engineering import create_features

root = Path(__file__).resolve().parent
metrics_dir = root / 'artifacts' / 'metrics'
metadata = json.loads((root / 'models' / 'model_metadata.json').read_text())
metrics = metadata['validation_metrics']
validation = pd.read_csv(metrics_dir / 'validation_predictions.csv')
actual, probability = validation.actual_default, validation.probability_of_default
recomputed = {'ROC-AUC': roc_auc_score(actual, probability),
    'PR-AUC': average_precision_score(actual, probability),
    'Brier': brier_score_loss(actual, probability)}
tn, fp, fn, tp = confusion_matrix(actual, probability >= metadata['selected_threshold']).ravel()
recomputed.update({'Accuracy': (tp+tn)/len(actual), 'Precision': tp/(tp+fp),
    'Recall': tp/(tp+fn), 'F1': 2*tp/(2*tp+fp+fn)})
# Independently use class score CDFs, rather than the notebook's ROC computation.
positive = np.sort(probability[actual == 1])
negative = np.sort(probability[actual == 0])
points = np.unique(probability)
ks = np.max(np.abs(np.searchsorted(positive, points, side='right')/len(positive)
    - np.searchsorted(negative, points, side='right')/len(negative)))
recomputed['KS'] = ks
recomputed['Gini'] = 2*recomputed['ROC-AUC'] - 1
for key, value in recomputed.items():
    np.testing.assert_allclose(value, metrics[key], atol=1e-10)

source_test = root / 'Test.csv'
if not source_test.exists():
    source_test = root / 'data' / 'Test.csv'
test = pd.read_csv(source_test)
submission = pd.read_csv(root / 'submission.csv')
assert submission.columns.tolist() == ['ID', 'default']
assert len(submission) == len(test) and submission.notna().all().all()
np.testing.assert_array_equal(submission.ID, test.ID)
assert set(submission.default.unique()) <= {0, 1}
fresh_predictions = predict_borrowers(test)
np.testing.assert_array_equal(fresh_predictions.predicted_class, submission.default)
saved_probabilities = pd.read_csv(root / 'artifacts' / 'test_predictions_with_probabilities.csv')
np.testing.assert_allclose(fresh_predictions.probability_of_default,
    saved_probabilities.probability_of_default, atol=1e-10)
perturbed = test.head(10).copy()
perturbed['ID'] = -1
perturbed['interest_receive'] = 999999999
perturbed['last_week_pay'] = -999999
pd.testing.assert_frame_equal(create_features(test.head(10)), create_features(perturbed))
pd.testing.assert_frame_equal(predict_borrowers(test.head(10)), predict_borrowers(perturbed))
notebook = nbformat.read(root / '01_NBFC_Loan_Default_Model.ipynb', as_version=4)
nbformat.validate(notebook)
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == 'code')
assert not [o for c in notebook.cells for o in c.get('outputs', []) if o.output_type == 'error']
checks = {'saved_metrics_independently_recomputed': True,
    'KS_independently_recomputed_from_class_CDFs': True,
    'fresh_process_test_predictions_match_all_rows': True,
    'ID_and_repayment_perturbations_do_not_change_origination_features_or_predictions': True,
    'test_rows_verified': len(test), 'notebook_format_and_executed_outputs_valid': True}
(metrics_dir / 'independent_artifact_checks.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
