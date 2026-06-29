# PROJECT - Preferential Randomisation: Fairness Interventions & SHAP Explanations in Healthcare 

Predict healthcare utilisation risk encoding existing racial disparities in society, as marginalised groups often tend to under-utilise care because of access barriers rather than lower need. In this project, machine learning models, such as logistic regression and gradient boosting, are trained on the MEPS 2023 dataset to predict healthcare utilisation. Also, a evaluation was performed implementing four fairness mitigation methods, which correspond to reweighing, calibrated equalised odds, plain equalised odds and preferential randomisation. Results show that plain equalised odds and preferential randomisation manage to reduce the equalised odds difference from $0.20$ to $\approx 0.02$. Recall was maintained and preferential randomisation achieved the closest disparate impact compared to parity ($0.892$). Calibrated equalised odds underperform under the unequal base rates of the dataset, collapsing recall to $0.16$ while making the fairness gap worse, which is a direct consequence of the statistical impossibility of simultaneously satisfying calibration and equalised error rates. A SHAP analysis was made to understand the explainable part of the fairness results. It revealed that race is encoded through multiple proxy features, such as nativity (BORNUSA), poverty and education, which acted as main channels of disparity. Reweighing reduces the reliance in these proxies partially but is not able to fully eliminate them. These outcomes demonstrate that post-processing methods offer the most favourable accuracy-fairness trade-off in this environment, where preferential randomisation performs as an competitive alternative to plain equalised odds in healthcare where mostly it had not previously been evaluated.

## Requirements
```bash
pip install -r requirements.txt
```

This repository contains all the files necessary for Fairness Interventions & SHAP Explanations in Healthcare. 
- Model: contains all the model files (`gradient_boosting.py`, `logistic_regression.py`), as well as mitigation strategies (`mitigation.py`, `pref_rand.py`, equations/), and utilities files (`plotter.py`, and `common_architecture.py`)
    - `mitigation.py` can be executed to implement the four fairness interventions.
    - `common_architecture.py` contains the model training and evaluation framework.
    - `shap_analysis.py` performs SHAP.
- data: contains all the data files and exploratory data analysis.
- images: contains all the necessary images generated throughout the project.
- utils: contains the preprocessing script.

To run the mitigation strategies, the user must indicate the method to be used:
```bash
python Model/mitigation.py --method (reweighing, eq_odds, pref_rand, eq_odds_sweep)
```
For the SHAP analysis, please use the shap_analysis.py:
```bash
python Model/shap_analysis.py
```

# Group JALUMEVA - Collaborators
- Janan Jahed
- Diana Luna
- Andrei Medesan
- Laura van der Hoef
