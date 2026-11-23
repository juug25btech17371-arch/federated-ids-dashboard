import streamlit as st
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import federated_ids

st.set_page_config(
    page_title="Federated IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Federated IDS Dashboard")
st.subheader("Intrusion Detection for IoT Edge Devices")

st.write(
    "Federated learning simulation with "
    "four IoT clients and FedAvg."
)

if st.button("Run Federated IDS Simulation"):
    output = StringIO()

    with st.spinner("Running simulation..."):
        with redirect_stdout(output):
            federated_ids.main()

    st.success("Simulation completed!")

    st.subheader("Simulation Results")
    st.code(output.getvalue())

    report = Path("results/final_report.txt")

    if report.exists():
        st.subheader("Final Report")
        st.text(report.read_text(encoding="utf-8"))
