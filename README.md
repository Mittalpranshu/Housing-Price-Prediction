# 🏡 Housing Price Prediction  

This project predicts housing prices using the California Housing dataset. It demonstrates a full ML workflow: preprocessing, training, saving the model, and making predictions.

---
## ⚙️ Features  

- Stratified train-test split based on income categories.  
- Preprocessing pipelines:
  - Numerical: median imputation + standard scaling  
  - Categorical: one-hot encoding  
- Model: Random Forest Regressor  
- Saves trained model and preprocessing pipeline automatically on first run  
- Generates predictions on `input.csv`  

---

## 🚀 Getting Started  

### 1️⃣ Clone the repository  
pip install -r requirements.txt
python main.py



```bash
git clone https://github.com/<your-username>/housing-price-prediction.git
cd housing-price-prediction
