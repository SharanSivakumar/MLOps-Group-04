import streamlit as st
import requests
import pandas as pd
from google.cloud import run_v2 

API_URL = "https://ecg-api-579499894470.europe-north1.run.app/predict"

st.title("ECG Classifier")
st.write("Upload a `.npy` file (expected sample shape leading to (1, 1, 224, 224) for the model).")

uploaded = st.file_uploader("Choose a .npy file", type=["npy"])

if uploaded is not None:
    st.caption(f"File: {uploaded.name} ({uploaded.size} bytes)")

    if st.button("Predict"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}

        try:
            r = requests.post(API_URL, files=files, timeout=60)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
            else:
                result = r.json()

                probs = result["probabilities"]  # dict: class -> prob
                pred_label = result["predicted_label"]

                st.success(f"Predicted: {pred_label}")

                # --- Bar chart (sorted) ---
                df = (
                    pd.DataFrame({"class": list(probs.keys()), "probability": list(probs.values())})
                    .sort_values("probability", ascending=False)
                    .set_index("class")
                )
                st.subheader("Class probabilities")
                st.bar_chart(df["probability"])

                # --- Highlight winner in a table (reliable highlighting) ---
                winner = df["probability"].idxmax()

                def highlight_winner(row):
                    return [
                        "font-weight: 800; background-color: #ffe08a" if row.name == winner else ""
                    ]

                st.subheader("Detailed values")
                st.dataframe(df.style.apply(highlight_winner, axis=1).format("{:.6f}"))

                # Optional: show raw JSON below
                with st.expander("Raw API response"):
                    st.json(result)

        except requests.RequestException as e:
            st.error(f"Failed to reach API at {API_URL}: {e}")