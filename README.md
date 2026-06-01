# Biohydrogen Optimization Platform

An AI-assisted Streamlit app for dark-fermentation biohydrogen research. It predicts hydrogen yield, estimates volatile fatty acid profiles, explores the training dataset, finds similar literature-derived experiments, and searches operating conditions for better process performance.

## Features

- Hydrogen yield and VFA prediction from fermentation inputs
- Dataset filtering, charts, and CSV export
- Model analytics with test metrics, cross-validation, and SHAP support
- Similar-experiment search using cosine similarity
- Process optimization over temperature, pH, and organic loading rate

## Project Structure

```text
.
├── app.py
├── pages/
│   ├── 1_Prediction_Dashboard.py
│   ├── 2_Dataset_Explorer.py
│   ├── 3_Model_Analytics.py
│   ├── 4_Similar_Literature.py
│   └── 5_Process_Optimization.py
├── dataset/
│   └── processed_dataset.csv
├── models/
│   ├── hydrogen_model.pkl
│   ├── vfa_model.pkl
│   └── model_meta.pkl
├── utils/
├── requirements.txt
└── train_models.py
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If models are missing, train demo models first:

```bash
python train_models.py --synthetic --n-synthetic 300
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the GitHub repo and branch.
4. Set the main file path to `app.py`.
5. Deploy.

The app includes `.streamlit/config.toml`, trained model files, the dataset, and `requirements.txt`, so Streamlit Cloud can install and run it directly.
