# Mental Health Treatment Predictor

## Setup
1. Create and activate a virtual enviroment 
    ```bash 
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate   # Windows
2. Install dependencies
    ```bash
    pip install -r requirements.txt
## Usage
1. Clean the data
    ```bash
    python3 clean_data.py
2. Train the model
    ```bash
    python3 train_model.py
3. Run the app
    ```bash
    streamlit run app.py
