import logging
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import joblib



# Configure logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("InferencePipeline")


class SupplyChainPredictor:

    def __init__(self , models_dir: Path):

        self.models_dir = models_dir
        self.preprocessor_path = models_dir / "preprocessor.joblib"
        self.model_path = models_dir / "best_model.joblib"

        self._load_artifacts()


    def _load_artifacts(self) -> None:
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {self.preprocessor_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)

        logger.info(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info("Artifacts loaded successfully")

    def predict(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing incoming data order...")
        transformed_data = self.preprocessor.transform(raw_data)

        logger.info("Generating predictions...")
        predictions = self.model.predict(transformed_data)
        probabilities = self.model.predict_proba(transformed_data)[:, 1]

        # Append predictions to the output dataframe 
        results = raw_data.copy()
        results["predicted_late_risk"] = predictions
        results["delay_probability"] = np.round(probabilities * 100, 2)
        results["risk_level"] = np.where(probabilities > 0.5, "HIGH RISK", "LOW RISK")

        return results


def main():


     PROJECT_ROOT = Path(__file__).resolve().parent.parent
     MODELS_DIR = PROJECT_ROOT / "models"

     predictor =  SupplyChainPredictor(models_dir=MODELS_DIR)



     sample_order = pd.DataFrame([{
        "Type": "DEBIT",
        "Days for shipment (scheduled)": 4,
        "Benefit per order": 91.25,
        "Sales per customer": 314.64,
        "Category Name": "Sporting Goods",
        "Customer City": "Caguas",
        "Customer Country": "Puerto Rico",
        "Customer Segment": "Consumer",
        "Customer State": "PR",
        "Customer Zipcode": 725.0,
        "Department Name": "Fitness",
        "Latitude": 18.251453,
        "Longitude": -66.037056,
        "Market": "Pacific Asia",
        "Order City": "Bekasi",
        "Order Country": "Indonesia",
        "Order Item Discount": 13.11,
        "Order Item Discount Rate": 0.04,
        "Order Item Product Price": 327.75,
        "Order Item Profit Ratio": 0.29,
        "Order Item Quantity": 1,
        "Sales": 327.75,
        "Order Item Total": 314.64,
        "Order Profit Per Order": 91.25,
        "Order Region": "Southeast Asia",
        "Order State": "Java Occidental",
        "Order Zipcode": 59405.0,
        "Product Name": "Smart watch",
        "Product Price": 327.75,
        "Product Status": 0,
        "Shipping Mode": "Standard Class",
        "order month": 1,
        "order_dayOfWeek": 2,
        "is_weekend": 0,
        "order_total_value": 327.75,
        "Is_International": True
    }])


    #run prediction 
     results = predictor.predict(sample_order)


     print("\n--- PREDICTION RESULTS ---")
     print(f"Predicted Risk Label: {results['predicted_late_risk'].iloc[0]} ({results['risk_level'].iloc[0]})")
     print(f"Delay Probability: {results['delay_probability'].iloc[0]}%")




if __name__ == "__main__":
    main()
