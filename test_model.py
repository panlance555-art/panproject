
import joblib

# Load the trained model
model = joblib.load("model/spam_detector.joblib")

print("SMS SPAM DETECTION SYSTEM")
print("=========================")

# Test messages
test_messages = [
    "Hey, are we still meeting tomorrow?",
    "Congratulations! You won a free prize. Click here to claim now!",
    "Please call me when you arrive home.",
    "URGENT! You have won a cash reward. Send your details now."
]

# Predict each message
for message in test_messages:

    prediction = model.predict([message])[0]

    print("\nMessage:", message)
    print("Prediction:", prediction.upper())

    if prediction == "spam":
        print("Result: SPAM DETECTED!")
    else:
        print("Result: LEGITIMATE MESSAGE")
