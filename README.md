# Smart Contactless Hotel Experience

A hackathon-ready prototype that simulates an end-to-end smart hotel digital journey with:

- Pre check-in using digital identity and OTP verification
- Biometric-based check-in using a simulated match score
- Automated checkout and bill calculation
- AI-powered fraud detection using a trained RandomForest model
- Streamlit-based demo UI for judges and stakeholders

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Faker

## Project Structure

```text
smart_hotel_app/
├── app.py
├── dataset_generator.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
└── data/
    └── hotel_dataset.csv
```

## Setup Instructions

### 1. Create and activate virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic dataset

```bash
python dataset_generator.py
```

This creates:

```text
data/hotel_dataset.csv
```

### 4. Train fraud detection model

```bash
python model.py
```

This creates:

```text
model.pkl
```

### 5. Run Streamlit app

```bash
streamlit run app.py
```

## Demo Flow for Hackathon

### Pre Check-In

1. Enter guest name, email, and phone.
2. Click **Generate OTP**.
3. Copy the demo OTP shown on screen.
4. Enter OTP and click **Verify OTP**.
5. The app marks digital identity as verified.

### Biometric Check-In

1. Move biometric score slider.
2. Set score above `0.75`.
3. Click **Verify Biometric**.
4. The app simulates successful biometric check-in.

### Checkout

1. Enter stay days and room rate.
2. App calculates total bill automatically.
3. Click **Run Fraud Check & Checkout**.
4. ML model predicts fraud risk.
5. Checkout is approved only if biometric score is valid and fraud prediction is `0`.

### Admin Dashboard

1. View dataset preview.
2. Check total guests, fraud rate, average bill, and checked-in rate.
3. Review room-rate and fraud-distribution charts.

## Hackathon Pitch

This prototype demonstrates how hotels can reduce front-desk dependency and improve guest experience using contactless verification, biometric authentication, automated payment calculation, and AI fraud detection. It is intentionally lightweight, explainable, and easy to run during a live hackathon demo.
