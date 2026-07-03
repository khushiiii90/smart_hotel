"""
app.py
Main Streamlit app for Smart Contactless Hotel Experience.

Run:
    streamlit run app.py
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import streamlit as st

from utils import calculate_bill, generate_otp, random_biometric_score, validate_otp

DATA_FILE = Path("data") / "hotel_dataset.csv"
MODEL_FILE = Path("model.pkl")
BIOMETRIC_THRESHOLD = 0.75
FEATURES = ["stay_days", "room_rate", "total_bill", "biometric_score"]


st.set_page_config(
    page_title="Smart Contactless Hotel Experience",
    page_icon="🏨",
    layout="wide",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load generated hotel dataset."""
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    """Load trained fraud detection model."""
    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


def initialize_session() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "otp": None,
        "otp_verified": False,
        "precheckin_done": False,
        "guest_name": "",
        "guest_email": "",
        "guest_phone": "",
        "biometric_score": 0.80,
        "biometric_verified": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def predict_fraud(stay_days: int, room_rate: float, total_bill: float, biometric_score: float) -> int:
    """Predict fraud using trained ML model."""
    model = load_model()
    input_df = pd.DataFrame([{
        "stay_days": stay_days,
        "room_rate": room_rate,
        "total_bill": total_bill,
        "biometric_score": biometric_score,
    }])
    return int(model.predict(input_df[FEATURES])[0])


def render_header() -> None:
    st.title("🏨 Smart Contactless Hotel Experience")
    st.caption("Hackathon-ready prototype: digital identity, OTP, biometric check-in, AI fraud detection, and automated checkout.")


def render_pre_checkin() -> None:
    st.header("1️⃣ Pre Check-In: Digital Identity + OTP")
    st.write("Enter guest details and simulate OTP-based digital identity verification.")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Guest Name", value=st.session_state.guest_name, placeholder="Enter full name")
        email = st.text_input("Email", value=st.session_state.guest_email, placeholder="guest@example.com")
        phone = st.text_input("Phone", value=st.session_state.guest_phone, placeholder="+91 99999 99999")

        if st.button("Generate OTP", type="primary"):
            if name and email and phone:
                st.session_state.guest_name = name
                st.session_state.guest_email = email
                st.session_state.guest_phone = phone
                st.session_state.otp = generate_otp()
                st.session_state.otp_verified = False
                st.info(f"Demo OTP generated: {st.session_state.otp}")
            else:
                st.warning("Please enter name, email, and phone before generating OTP.")

    with col2:
        otp_input = st.text_input("Enter OTP", placeholder="6-digit OTP")
        if st.button("Verify OTP"):
            if st.session_state.otp is None:
                st.error("Please generate OTP first.")
            elif validate_otp(otp_input, st.session_state.otp):
                st.session_state.otp_verified = True
                st.session_state.precheckin_done = True
                st.success("OTP verified. Pre check-in completed successfully!")
            else:
                st.session_state.otp_verified = False
                st.error("Invalid OTP. Please try again.")

    if st.session_state.precheckin_done:
        st.success(f"✅ Digital identity verified for {st.session_state.guest_name}")


def render_biometric_checkin() -> None:
    st.header("2️⃣ Biometric Check-In")
    st.write("Simulate biometric matching using a match score. Score above 0.75 is accepted.")

    if not st.session_state.precheckin_done:
        st.warning("Complete Pre Check-In before biometric verification.")

    col1, col2 = st.columns([2, 1])
    with col1:
        score = st.slider(
            "Biometric Match Score",
            min_value=0.70,
            max_value=1.00,
            value=float(st.session_state.biometric_score),
            step=0.01,
        )
        st.session_state.biometric_score = score

        if st.button("Use Random Biometric Score"):
            st.session_state.biometric_score = random_biometric_score()
            st.rerun()

        if st.button("Verify Biometric", type="primary"):
            if not st.session_state.precheckin_done:
                st.error("Pre Check-In is required before biometric verification.")
            elif score > BIOMETRIC_THRESHOLD:
                st.session_state.biometric_verified = True
                st.success(f"Biometric verification successful. Score: {score:.2f}")
            else:
                st.session_state.biometric_verified = False
                st.error(f"Biometric verification failed. Score must be greater than {BIOMETRIC_THRESHOLD}.")

    with col2:
        st.metric("Current Match Score", f"{st.session_state.biometric_score:.2f}")
        st.metric("Required Threshold", f"> {BIOMETRIC_THRESHOLD}")
        if st.session_state.biometric_verified:
            st.success("✅ Guest checked in")
        else:
            st.warning("⏳ Awaiting biometric approval")


def render_checkout() -> None:
    st.header("3️⃣ Automated Checkout + AI Fraud Detection")
    st.write("Calculate bill and allow checkout only when biometric verification succeeds and ML fraud prediction is clear.")

    col1, col2 = st.columns(2)
    with col1:
        stay_days = st.number_input("Stay Days", min_value=1, max_value=30, value=2, step=1)
        room_rate = st.number_input("Room Rate", min_value=1000.0, max_value=50000.0, value=4500.0, step=500.0)
        total_bill = calculate_bill(stay_days, room_rate)
        st.metric("Total Bill", f"₹{total_bill:,.2f}")

    with col2:
        st.write("### Checkout Decision")
        st.write(f"Biometric Score: **{st.session_state.biometric_score:.2f}**")
        st.write(f"Biometric Verified: **{st.session_state.biometric_verified}**")

        if st.button("Run Fraud Check & Checkout", type="primary"):
            if not MODEL_FILE.exists():
                st.error("Model file not found. Please run: python model.py")
                return

            fraud_prediction = predict_fraud(
                stay_days=int(stay_days),
                room_rate=float(room_rate),
                total_bill=float(total_bill),
                biometric_score=float(st.session_state.biometric_score),
            )

            if fraud_prediction == 1:
                st.error("🚨 Fraud risk detected by ML model. Checkout blocked for manual review.")
            elif not st.session_state.biometric_verified:
                st.error("Biometric verification is not complete. Checkout blocked.")
            elif st.session_state.biometric_score <= BIOMETRIC_THRESHOLD:
                st.error("Biometric score below threshold. Checkout blocked.")
            else:
                st.success("✅ Checkout approved. Payment calculated and guest journey completed.")
                st.balloons()


def render_admin_dashboard() -> None:
    st.header("4️⃣ Admin Dashboard")

    if not DATA_FILE.exists():
        st.error("Dataset not found. Please run: python dataset_generator.py")
        return

    df = load_dataset()

    total_guests = len(df)
    fraud_rate = df["fraud_flag"].mean() * 100
    avg_bill = df["total_bill"].mean()
    checked_in_rate = df["checked_in"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Guests", f"{total_guests:,}")
    c2.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    c3.metric("Avg Bill", f"₹{avg_bill:,.0f}")
    c4.metric("Checked-In Rate", f"{checked_in_rate:.1f}%")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("Room Rate Distribution")
        bins = pd.cut(df["room_rate"], bins=10).value_counts().sort_index()
        bins.index = bins.index.astype(str)
        st.bar_chart(bins)

    with chart_col2:
        st.write("Fraud Distribution")
        fraud_counts = df["fraud_flag"].map({0: "Not Fraud", 1: "Fraud"}).value_counts()
        st.bar_chart(fraud_counts)


def main() -> None:
    initialize_session()
    render_header()

    with st.sidebar:
        st.header("Navigation")
        menu = st.radio(
            "Select Module",
            ["Pre Check-In", "Biometric Check-In", "Checkout", "Admin Dashboard"],
        )
        st.divider()
        st.write("### Journey Status")
        st.write(f"OTP Verified: {'✅' if st.session_state.otp_verified else '❌'}")
        st.write(f"Biometric Verified: {'✅' if st.session_state.biometric_verified else '❌'}")
        st.write(f"Biometric Score: {st.session_state.biometric_score:.2f}")

    if menu == "Pre Check-In":
        render_pre_checkin()
    elif menu == "Biometric Check-In":
        render_biometric_checkin()
    elif menu == "Checkout":
        render_checkout()
    elif menu == "Admin Dashboard":
        render_admin_dashboard()


if __name__ == "__main__":
    main()
