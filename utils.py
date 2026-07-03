"""
utils.py
Utility functions for OTP, billing, and biometric simulation.
"""

import random
from typing import Union


def generate_otp(length: int = 6) -> str:
    """Generate a numeric OTP."""
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def validate_otp(user_otp: str, actual_otp: str) -> bool:
    """Validate OTP entered by the user."""
    return str(user_otp).strip() == str(actual_otp).strip()


def calculate_bill(stay_days: Union[int, float], room_rate: Union[int, float]) -> float:
    """Calculate total hotel bill."""
    stay_days = int(stay_days)
    room_rate = float(room_rate)
    return round(stay_days * room_rate, 2)


def random_biometric_score() -> float:
    """Generate a random biometric match score between 0.70 and 1.00."""
    return round(random.uniform(0.70, 1.00), 3)
