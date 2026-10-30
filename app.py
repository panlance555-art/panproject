
from flask import Flask, render_template, request
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

# LOAD THE TRAINED SPAM DETECTION MODEL
model = joblib.load("model/spam_detector.joblib")


# DATABASE CONNECTION
def get_db_connection():
    connection = sqlite3.connect("history.db")
    connection.row_factory = sqlite3.Row
    return connection


# CREATE DATABASE TABLE
def create_database():
    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            date_checked TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# INITIALIZE DATABASE WHEN THE APPLICATION STARTS
# This also runs when Render starts the app using Gunicorn.
create_database()


# HOME PAGE AND SMS DETECTION
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if message:

            prediction = model.predict([message])[0].upper()

            connection = get_db_connection()

            connection.execute(
                """
                INSERT INTO history (message, prediction, date_checked)
                VALUES (?, ?, ?)
                """,
                (
                    message,
                    prediction,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )

            connection.commit()
            connection.close()

    # GET DASHBOARD STATISTICS
    connection = get_db_connection()

    stats = connection.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN prediction = 'HAM' THEN 1 ELSE 0 END) AS ham,
            SUM(CASE WHEN prediction = 'SPAM' THEN 1 ELSE 0 END) AS spam
        FROM history
    """).fetchone()

    connection.close()

    return render_template(
        "index.html",
        prediction=prediction,
        message=message,
        stats=stats
    )


# DETECTION HISTORY PAGE
@app.route("/history")
def history():

    connection = get_db_connection()

    records = connection.execute(
        "SELECT * FROM history ORDER BY id DESC"
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        records=records
    )


# ABOUT MODEL PAGE
@app.route("/about")
def about():

    return render_template("about.html")


# RUN FLASK APPLICATION LOCALLY
if __name__ == "__main__":

    app.run(debug=True)
