# Real‑Time Fraud Detection

### Overview
This repository implements a real‑time fraud detection pipeline that identifies fraudulent card transactions while controlling false positives through threshold tuning and cost‑aware evaluation.
The solution emphasizes clean preprocessing, interpretable modeling, rigorous validation, and production‑ready packaging for seamless deployment.

### ProblemBanks must catch fraud with minimal latency and near‑zero misses while keeping investigation queues manageable.
This project builds a probabilistic classifier to learn relationships among key transaction features and to expose a tunable decision threshold aligned with business risk appetite.

### Data- Source file: fraud_data.xlsx with 500 transactions and label column IsFraud.
- Key fields: TransactionID, Amount, Time, Location, MerchantCategory, CardHolderAge, and IsFraud.
- Class balance: minority fraud class suitable for stratified splitting and imbalance‑aware training.

### Approach- Robust preprocessing with median and mode imputations, feature scaling, and label encoding for categorical attributes.
- Feature engineering for Hour, amount bands, age bands, and time‑of‑day bands to capture temporal and behavioral signals.
- Model portfolio trained and compared on AUC with stratified cross‑validation and a held‑out test set.### Results- Best model: Logistic Regression with class balancing, delivering the top test AUC among candidates.
- Final operating point selected at a 0.3 probability threshold to prioritize recall and minimize missed fraud.- Top drivers include Amount, CardHolderAge, Time, Hour, and MerchantCategory based on tree‑based importance and linear coefficients.### Repository structure- notebooks/ Real‑time Fraud Detection.ipynb — end‑to‑end analysis and modeling workflow.
- data/fraud_data.xlsx — source dataset used in the notebook and scripts.
- artifacts/fraud_detection_model.pkl — serialized package with model, scaler, encoders, and threshold.
- artifacts/fraud_predictions.csv and artifacts/model_performance.csv — exported predictions and model comparison.

### Quick start- Python 3.10+ and common ML stack are required to run the notebook and scripts.
- Place fraud_data.xlsx under data/ or update the path in the notebook before running all cells.

### Reproducible pipeline- Deterministic random seeds, stratified train/test split, and explicit scaling preserve experimental integrity.
- Cross‑validation AUC and held‑out AUC are logged per model to avoid selection bias and information leakage.

### Modeling details- Candidate models: Logistic Regression, Random Forest, Gradient Boosting, and RBF‑SVM, each configured for class imbalance where applicable.
- Threshold sweep evaluates precision–recall trade‑offs to align model alerts with operational review capacity.

### Inference- The serialized artifact includes model, StandardScaler, all LabelEncoders, feature list, and the chosen decision threshold.
- A helper function demonstrates single‑record scoring with consistent feature engineering and categorical encodings for production parity.

### Data preprocessing- Numerical imputations: median for Amount and CardHolderAge to reduce sensitivity to skew and outliers.
- Categorical imputations: mode for Location to preserve distribution and downstream encoder mappings.

### Feature engineering- Hour derived from Time modulo day length for intraday pattern learning.
- Discrete bands for Amount, CardHolderAge, and time‑of‑day strengthen interactions and stabilize linear decision boundaries.

### Evaluation protocol- Metrics: AUC as the model selection criterion, complemented by confusion matrix, precision, recall, F1, and specificity at selected thresholds.
- Operating point chosen using threshold sweep tables and confusion matrix to balance capture rate and alert volume.

### Model selection rationale- Logistic Regression was preferred for its strong AUC, interpretable coefficients, and predictable behavior under threshold optimization.
- Interpretability enables faster fraud‑ops feedback loops and policy adjustments as patterns evolve.

### Deployment notes- The artifact is ready for use in a stateless scoring service where feature engineering and encoding mirror the training path.
- Monitoring should track PSI for inputs, alert volume, precision, recall, and drift‑aware retraining cadence.

### How to extend- Calibrate threshold by line‑of‑business cost ratios and queue capacity to tune precision–recall trade‑offs.
- Explore stacking or calibrated gradient boosting and cost‑sensitive loss functions for precision improvements at fixed recall.

### Limitations- Small sample and simplified schema constrain generalization, so continuous learning with fresh data is recommended.
- High recall operating points may increase false positives, requiring workflow automation and human‑in‑the‑loop review.

### Ethical use- Use responsibly with appropriate consent, security controls, and bias monitoring across demographic dimensions where legally permissible.
- Document adverse‑action logic and maintain audit trails for all automated decisions and human overrides.

### Citation- Dataset: fraud_data.xlsx included in this repository’s data directory.
- Performance charts and confusion matrix are derived from the attached notebook outputs and embedded as visual artifacts.

<img width="2400" height="1600" alt="image" src="https://github.com/user-attachments/assets/50815e62-76f4-49e6-93b6-88f7b28e0518" />
<img width="2400" height="1600" alt="image" src="https://github.com/user-attachments/assets/2d92cba9-16ab-4b44-975d-66bbae60fa27" />
<img width="2400" height="1600" alt="image" src="https://github.com/user-attachments/assets/2f4bb88d-96c2-461e-9746-151665481e47" />

