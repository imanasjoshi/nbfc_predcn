"""Deployment helper: same transformations and saved probability mapping."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from models.feature_engineering import create_features, calibrated_probability, assign_risk_band

def predict_borrowers(records, project_root=None):
    """Return probability, thresholded class and risk band from raw input rows."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent
    bundle = joblib.load(root / 'models' / 'origination_model_bundle.pkl')
    metadata = json.loads((root / 'models' / 'model_metadata.json').read_text())
    risk = json.loads((root / 'models' / 'risk_config.json').read_text())
    raw = pd.DataFrame(records)
    features = create_features(raw)[metadata['features']]
    probabilities = calibrated_probability(bundle['model'], features,
        bundle['calibrator'], bundle['calibration_method'])
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    return pd.DataFrame({'probability_of_default': probabilities,
        'predicted_class': (probabilities >= metadata['selected_threshold']).astype(int),
        'risk_band': [assign_risk_band(p, risk['boundaries']) for p in probabilities]}, index=raw.index)
