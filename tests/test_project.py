import ast
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from predict import predict_frame

ROOT = Path(__file__).resolve().parents[1]

class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT / "results/final_model_pipeline.joblib")
        cls.inputs = pd.read_csv(ROOT / "examples/input_features.csv")

    def test_saved_pipeline_matches_recorded_predictions(self):
        expected = pd.read_csv(ROOT / "examples/expected_predictions.csv")
        actual = predict_frame(self.inputs, self.model)
        np.testing.assert_allclose(actual["Predicted_RUL"], expected["Predicted_RUL"], rtol=1e-9, atol=1e-8)

    def test_input_column_order_does_not_change_predictions(self):
        reordered = self.inputs[self.inputs.columns[::-1]]
        original = predict_frame(self.inputs, self.model)
        changed = predict_frame(reordered, self.model)
        np.testing.assert_allclose(original["Predicted_RUL"], changed["Predicted_RUL"])

    def test_missing_feature_is_rejected(self):
        incomplete = self.inputs.drop(columns="sensor_2")
        with self.assertRaisesRegex(ValueError, "Missing required features"):
            predict_frame(incomplete, self.model)

    def test_missing_value_is_rejected(self):
        incomplete = self.inputs.copy()
        incomplete.loc[0, "sensor_2"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            predict_frame(incomplete, self.model)

    def test_metrics_match_published_engine_predictions(self):
        predictions = pd.read_csv(ROOT / "results/fd001_test_predictions.csv")
        metrics = pd.read_csv(ROOT / "results/final_metrics.csv").iloc[0]
        self.assertEqual(len(predictions), 100)
        self.assertTrue(predictions["unit_number"].is_unique)
        actual = predictions["Actual_RUL"]
        predicted = predictions["Predicted_RUL"]
        self.assertAlmostEqual(mean_absolute_error(actual, predicted), metrics["MAE"], places=8)
        self.assertAlmostEqual(np.sqrt(mean_squared_error(actual, predicted)), metrics["RMSE"], places=8)
        self.assertAlmostEqual(r2_score(actual, predicted), metrics["R2"], places=8)

    def test_notebook_code_has_valid_syntax(self):
        notebook = json.loads((ROOT / "notebooks/01_data_understanding.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

if __name__ == "__main__":
    unittest.main()
