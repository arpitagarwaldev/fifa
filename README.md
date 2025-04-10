# FIFA Player Value Predictor

A machine learning web application that predicts a FIFA player's market value based on their attributes and characteristics.

## Features

- Predicts player value based on 36+ attributes
- User-friendly web interface
- Real-time predictions
- Input validation and error handling
- Beautiful Bootstrap-based UI

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arpitagarwaldev/fifa.git
   cd fifa
   ```

2. Create a Python virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python3 app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

3. Fill in the player attributes form:
   - Basic information (height, weight, age)
   - Technical skills (ball control, dribbling, etc.)
   - Physical attributes (stamina, strength, etc.)
   - Goalkeeper attributes (if applicable)

4. Click "Predict" to get the estimated player value

## Project Structure

```
.
├── app.py                # Flask application
├── artifacts/            # Model artifacts
│   ├── model.pkl         # Trained model
│   └── preprocessor.pkl   # Data preprocessor
├── src/
│   ├── components/       # Core components
│   ├── pipeline/         # Prediction pipeline
│   └── utils.py         # Utility functions
└── templates/           # HTML templates
```

## Model Information

- Uses CatBoost Regressor for prediction
- Features both numerical and categorical inputs
- Trained on FIFA player dataset
- Handles missing values and feature scaling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.