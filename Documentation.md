# Documentation

## Setup

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the application**

Navigate to the `src` directory and run the following command:

```bash
cd src
streamlit run app.py
```

If you run the command outside the `src` directory, you will get an error because the app won't find the ".streamlit" folder.

## Project Structure
This repository is structured as follows:

```
project/
├── data/
│   ├── raw/                            # Raw data 
│   ├── processed/                      # Data after preprocessing
│   ├── engineered/                     # Data after feature engineering 
│   └── final/                          # Data split into train, validation, and test sets 
│
├── notebooks/                          # Jupyter notebooks directory for experimentation
│
├── src/
│   ├── preprocessing/                  # Data processing module
│   ├── feature_engineering/            # Feature engineering module
│   ├── training/                       # Training module
│   ├── evaluation/                     # Evaluation module
│   ├── visualization/                  # Visualization module (utility functions for plotting)
│   ├── utils/                          # Utility module (useful functions)
│   └── main.py                         # Main script for running the whole pipeline
│
```

> The files in this folder contains the scripts to preprocess the data and train, evaluate and generate explanations for the models. They also contain utilities functions that are used in the notebooks for a more interactive experience.
