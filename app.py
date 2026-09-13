import json
import base64
import io
import math
import re
import struct
import wave

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance System", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#eaf1f4; --muted:#8fa2ad; --panel:#14232b; --line:#29404a; --cyan:#45d5c7; --amber:#f5b85d; --red:#f06b6b; }
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: radial-gradient(circle at 85% 5%, #18333b 0, #091319 38rem); color:var(--ink); }
[data-testid="stSidebar"] { background:#0c1b22; border-right:1px solid var(--line); }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:16px; border-radius:8px; }
[data-testid="stMetricValue"] { color:var(--ink); font-family:'DM Mono', monospace; }
.hero { border-bottom:1px solid var(--line); padding:16px 0 25px; margin-bottom:24px; }
.eyebrow { color:var(--cyan); font:500 12px 'DM Mono', monospace; letter-spacing:1.8px; text-transform:uppercase; }
.hero h1 { font-size:clamp(2rem, 4vw, 3.5rem); letter-spacing:-1px; margin:7px 0; }
.hero p { color:var(--muted); margin:0; max-width:650px; }
.panel { background:rgba(20,35,43,.78); border:1px solid var(--line); border-radius:8px; padding:20px; }
.panel-title { color:var(--muted); font:500 11px 'DM Mono', monospace; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px; }
.status { border-left:4px solid var(--cyan); background:#112d31; padding:20px 22px; border-radius:4px; }
.status h2 { margin:0; font-size:30px; }
.status p { color:var(--muted); margin:5px 0 0; }
.risk-track { background:#263941; height:10px; border-radius:10px; overflow:hidden; margin:10px 0 5px; }
.risk-fill { height:100%; background:linear-gradient(90deg,var(--cyan),var(--amber),var(--red)); border-radius:10px; }
.mono { font-family:'DM Mono', monospace; }
div.stButton > button[kind="primary"] { background:var(--cyan); color:#07181b; border:0; font-weight:700; }
button { border-radius:5px !important; }
@keyframes logoFloat { 0%, 100% { transform:translateY(0); } 50% { transform:translateY(-6px); } }
@keyframes statusPulse { 0%, 100% { opacity:.45; box-shadow:0 0 0 0 rgba(69,213,199,.35); } 50% { opacity:1; box-shadow:0 0 0 7px rgba(69,213,199,0); } }
div[data-testid="stImage"] img { animation:logoFloat 6s ease-in-out infinite; }
.pulse-dot { display:inline-block; width:8px; height:8px; margin-right:7px; border-radius:50%; background:var(--cyan); animation:statusPulse 2.2s ease-in-out infinite; vertical-align:1px; }
@keyframes warningGlow { 0%, 100% { box-shadow:0 0 8px rgba(245,184,93,.18); } 50% { box-shadow:0 0 22px rgba(245,184,93,.55); } }
.maintenance-alert { border:1px solid #a97932; border-left:5px solid var(--amber); background:#2b2417; color:#ffe0a2; padding:14px 18px; border-radius:6px; animation:warningGlow 2.4s ease-in-out infinite; }
.maintenance-alert strong { color:#ffd27a; }
@media (prefers-reduced-motion: reduce) { div[data-testid="stImage"] img, .pulse-dot { animation:none; } }
@media (prefers-reduced-motion: reduce) { .maintenance-alert { animation:none; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    with open("feature_columns.json") as file:
        feature_columns = json.load(file)
    with open("model_info.json") as file:
        model_info = json.load(file)
    return model, label_encoder, feature_columns, model_info


@st.cache_data
def load_previous_data():
    return pd.read_csv("industrial_predictive_maintenance_dataset.xls")


@st.cache_data
def make_warning_sound():
    sample_rate = 22050
    duration = 0.7
    frames = []
    for index in range(int(sample_rate * duration)):
        time = index / sample_rate
        tone = math.sin(2 * math.pi * 880 * time) * (1 if time < 0.28 else 0.55)
        frames.append(struct.pack("<h", int(15000 * tone)))
    audio = io.BytesIO()
    with wave.open(audio, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(sample_rate)
        sound.writeframes(b"".join(frames))
    return audio.getvalue()


model, label_encoder, feature_columns, model_info = load_artifacts()
previous_data = load_previous_data()
FEATURE_RANGES = {
    "Machine_ID": (1, 100, 1, 1), "Vibration": (0.0, 14.0, 2.0, 0.1), "Temperature": (28.0, 115.0, 55.0, 0.5),
    "Smoke_Level": (0.0, 100.0, 10.0, 0.5), "Current": (2.0, 150.0, 35.0, 0.5), "Voltage": (140.0, 300.0, 225.0, 0.5),
    "RPM": (300, 2800, 1600, 10), "Pressure": (0.4, 13.0, 5.0, 0.1), "Humidity": (20.0, 90.0, 50.0, 0.5),
    "Acoustic_Level": (38.0, 128.0, 65.0, 0.5), "Power_Consumption": (0.1, 40.0, 8.0, 0.1), "Operating_Hours": (100, 30000, 5000, 100),
    "Load_Percentage": (0.0, 100.0, 50.0, 0.5), "Maintenance_History": (0, 60, 10, 1), "Days_Since_Maintenance": (0, 1000, 100, 5),
    "Machine_Age": (1, 25, 5, 1), "Failure_Risk_Score": (0.0, 100.0, 20.0, 0.5),
}
STATUS_COLORS = {"Normal": "#45d5c7", "Warning": "#f5b85d", "Critical": "#f07862", "Failure": "#f06b6b"}
STATUS_GUIDANCE = {
    "Normal": "The machine looks healthy. Keep checking it at the usual time.",
    "Warning": "The machine shows a possible problem. Plan a check soon and watch the readings.",
    "Critical": "The machine may have a serious problem. Check it as soon as possible.",
    "Failure": "The machine may have failed. Stop and inspect it before using it again.",
}

with st.sidebar:
    st.markdown("## PREDICTIVE MAINTENANCE SYSTEM")
    st.caption("MACHINE HEALTH TOOL")
    st.divider()
    online_mode = st.toggle("Online mode", value=False)
    mode_name = "ONLINE MODE" if online_mode else "OFFLINE MODE"
    st.markdown(f"**{mode_name}**")
    st.markdown(f"`{model_info['best_model']}`")
    st.markdown(f"Accuracy  **{model_info['test_accuracy']:.1%}**")
    st.markdown(f"Macro F1  **{model_info['test_f1_macro']:.3f}**")
    st.divider()
    st.caption("Online mode selected • no live sensor feed connected" if online_mode else "Offline mode • enter data by hand")

hero_logo, hero_copy = st.columns([0.18, 1])
with hero_logo:
    st.image("logo.png", width=130)
with hero_copy:
    st.markdown('<div class="hero"><div class="eyebrow"><span class="pulse-dot"></span>Machine health / v2.0</div><h1>Know the failure before it starts.</h1><p>A simple tool that turns machine data into clear maintenance steps.</p></div>', unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("MODEL ACCURACY", f"{model_info['test_accuracy']:.1%}", "validated")
kpi2.metric("MACRO F1 SCORE", f"{model_info['test_f1_macro']:.3f}", "balanced")
kpi3.metric("STATUS CLASSES", len(label_encoder.classes_), "tracked")
kpi4.metric("SENSOR FEATURES", len(feature_columns), "in model")

tab_overview, tab_assess, tab_fleet, tab_reference = st.tabs(["◉ Overview", "⌁ Check One Machine", "▦ Check Many Machines", "⌘ Sensor Guide"])

with tab_overview:
    if online_mode:
        st.info("Online mode is selected, but no live sensor feed is connected yet. Predictions still use the data you enter or upload.")
    alert_icon, alert_text = st.columns([0.14, 1])
    with alert_icon:
        warning_gif = base64.b64encode(open("warning_light.gif", "rb").read()).decode("ascii")
        st.markdown(f'<img class="warning-gif" src="data:image/gif;base64,{warning_gif}" width="90" alt="Flashing warning light">', unsafe_allow_html=True)
    if st.button("Play warning sound", help="Browsers require a click before playing sound."):
        st.audio(make_warning_sound(), format="audio/wav", autoplay=True)
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">System status</div>', unsafe_allow_html=True)
        st.subheader("Machine health, at a glance")
        st.write("Predictive Maintenance System checks one machine or many machines with the trained model. Enter machine data or upload a CSV file, then download a work list for your team.")
        st.markdown("#### How to use")
        st.markdown("`01` **Check** a machine with sensor data\n\n`02` **Review** the machine list by status\n\n`03` **Act** on the machines with the highest risk first")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Model results</div>', unsafe_allow_html=True)
        coverage = pd.DataFrame({"Metric": ["Accuracy", "Macro F1"], "Score": [model_info["test_accuracy"], model_info["test_f1_macro"]]}).set_index("Metric")
        st.bar_chart(coverage, color="#45d5c7", height=220)
        st.caption("Test results from the training notebook.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("Use **Check One Machine** for one machine, or upload a file in **Check Many Machines** to create an ordered work list.")
    with st.expander("Mobile updates"):
        st.caption("Add a number to receive maintenance updates.")
        with st.form("mobile_updates_form"):
            mobile_number = st.text_input("Mobile number", placeholder="+1 555 123 4567")
            sms_updates = st.checkbox("Send me maintenance updates")
            save_mobile = st.form_submit_button("Save mobile settings")
            if save_mobile:
                valid_number = re.fullmatch(r"\+?[0-9 ()-]{7,20}", mobile_number.strip())
                if not valid_number:
                    st.error("Enter a valid mobile number.")
                else:
                    st.session_state["mobile_number"] = mobile_number.strip()
                    st.session_state["sms_updates"] = sms_updates
                    st.success("Mobile settings saved for this session.")
        if st.session_state.get("sms_updates"):
            st.caption("Updates are turned on. SMS delivery needs a message service connection.")
    st.markdown('<div class="panel-title">Previous machine data</div>', unsafe_allow_html=True)
    st.write("These graphs use the 20,000 machine records included with this project. They show how risk and machine status changed across earlier records.")
    history_machine = st.selectbox("Show history for", ["All machines"] + sorted(previous_data["Machine_ID"].unique().tolist()))
    history_view = previous_data if history_machine == "All machines" else previous_data[previous_data["Machine_ID"] == history_machine]
    history_left, history_right = st.columns([1, 1.4])
    with history_left:
        status_counts = history_view["Machine_Status"].value_counts().reindex(label_encoder.classes_, fill_value=0)
        st.markdown('<div class="panel"><div class="panel-title">Earlier machine status</div>', unsafe_allow_html=True)
        st.bar_chart(status_counts, color="#45d5c7", height=260)
        st.caption(f"Status count for {history_machine.lower()}.")
        st.markdown('</div>', unsafe_allow_html=True)
    with history_right:
        risk_history = history_view[["Operating_Hours", "Failure_Risk_Score", "Machine_Status"]].sort_values("Operating_Hours")
        st.markdown('<div class="panel"><div class="panel-title">Risk by operating hours</div>', unsafe_allow_html=True)
        st.scatter_chart(risk_history, x="Operating_Hours", y="Failure_Risk_Score", color="Machine_Status", height=260)
        st.caption("Each point is an earlier machine record. Higher points mean a higher reported risk score.")
        st.markdown('</div>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown('<div class="panel"><div class="panel-title">Example sensor profile</div>', unsafe_allow_html=True)
        profile_features = ["Vibration", "Temperature", "Smoke_Level", "Load_Percentage", "Failure_Risk_Score"]
        profile = pd.DataFrame({
            "Signal": [feature.replace("_", " ") for feature in profile_features],
            "Level": [FEATURE_RANGES[feature][2] / FEATURE_RANGES[feature][1] for feature in profile_features],
        }).set_index("Signal")
        st.bar_chart(profile, color="#45d5c7", height=250)
        st.caption("Example values only. This is not a live sensor view.")
        st.markdown('</div>', unsafe_allow_html=True)
    with chart_right:
        st.markdown('<div class="panel"><div class="panel-title">Sensor value ranges</div>', unsafe_allow_html=True)
        range_features = ["Temperature", "Vibration", "Current", "RPM", "Failure_Risk_Score"]
        ranges = pd.DataFrame({
            "Signal": [feature.replace("_", " ") for feature in range_features],
            "Minimum": [FEATURE_RANGES[feature][0] for feature in range_features],
            "Maximum": [FEATURE_RANGES[feature][1] for feature in range_features],
        }).set_index("Signal")
        st.line_chart(ranges, height=250)
        st.caption("Lowest and highest values accepted by the input controls.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab_assess:
    st.markdown('<div class="panel-title">Check one machine</div>', unsafe_allow_html=True)
    with st.expander("How to use this page"):
        st.write("Enter the latest readings from one machine, then select **Check machine**. The model compares the readings with patterns learned from past machine data.")
        st.write("**Confidence** shows how sure the model is. **Risk level** is the risk number you entered. The charts help you see which readings are low or high in their allowed range.")
    input_values = {}
    groups = [feature_columns[:6], feature_columns[6:12], feature_columns[12:]]
    for group_index, group in enumerate(groups):
        st.markdown(f"**{'MAIN READINGS' if group_index == 0 else 'MACHINE READINGS' if group_index == 1 else 'MAINTENANCE HISTORY'}**")
        cols = st.columns(3)
        for i, feature in enumerate(group):
            lo, hi, default, step = FEATURE_RANGES.get(feature, (0.0, 100.0, 0.0, 1.0))
            with cols[i % 3]:
                input_values[feature] = st.number_input(feature.replace("_", " "), min_value=lo, max_value=hi, value=default, step=step, key=f"input_{feature}")
    if st.button("Check machine", type="primary", use_container_width=True):
        X = pd.DataFrame([input_values])[feature_columns]
        pred_label = label_encoder.inverse_transform(model.predict(X))[0]
        probabilities = model.predict_proba(X)[0]
        proba_df = pd.DataFrame({"Status": label_encoder.classes_, "Probability": probabilities}).sort_values("Probability", ascending=False)
        confidence = float(proba_df.iloc[0]["Probability"])
        color = STATUS_COLORS.get(pred_label, "#45d5c7")
        st.markdown(f'<div class="status" style="border-color:{color};"><div class="panel-title">Machine status</div><h2 style="color:{color};">{pred_label.upper()}</h2><p>Model confidence: <span class="mono">{confidence:.1%}</span></p></div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([1, 1.2])
        with result_left:
            st.markdown('<div class="panel"><div class="panel-title">Possible results</div>', unsafe_allow_html=True)
            st.bar_chart(proba_df.set_index("Status"), y="Probability", color="#45d5c7", height=230)
            st.markdown('</div>', unsafe_allow_html=True)
        with result_right:
            st.markdown('<div class="panel"><div class="panel-title">Risk level</div>', unsafe_allow_html=True)
            risk = float(input_values.get("Failure_Risk_Score", 0))
            st.markdown(f"### {risk:.0f} <span class='mono' style='font-size:14px;color:#8fa2ad'>/ 100 entered risk score</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='risk-track'><div class='risk-fill' style='width:{risk}%;'></div></div>", unsafe_allow_html=True)
            st.caption("This score is shown next to the model result for easy review.")
            priority = "Check now" if pred_label in ["Failure", "Critical"] else "Plan a check" if pred_label == "Warning" else "Keep watching"
            st.success(f"**Next step:** {priority}")
            st.info(f"**What this means:** {STATUS_GUIDANCE.get(pred_label, 'Review the readings and follow your maintenance process.')}")
            checkpoint_col, checkpoint_note = st.columns([1, 1.5])
            with checkpoint_col:
                send_checkpoint = st.button("Send checkpoint", use_container_width=True)
            with checkpoint_note:
                st.caption("Sends all machine readings to your saved mobile number.")
            if send_checkpoint:
                saved_number = st.session_state.get("mobile_number")
                if not st.session_state.get("sms_updates") or not saved_number:
                    st.warning("Add a mobile number and turn on updates in Mobile updates on the Overview page first.")
                else:
                    checkpoint = {feature: input_values[feature] for feature in feature_columns}
                    checkpoint["Predicted_Status"] = pred_label
                    checkpoint["Confidence"] = round(confidence, 3)
                    st.session_state["last_checkpoint"] = checkpoint
                    st.success(f"Checkpoint ready for {saved_number}. SMS delivery needs a message service connection.")
                    with st.expander("Checkpoint readings"):
                        st.dataframe(pd.DataFrame([checkpoint]), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown('<div class="panel"><div class="panel-title">Readings in range</div>', unsafe_allow_html=True)
            chart_features = ["Vibration", "Temperature", "Current", "Voltage", "RPM", "Pressure", "Humidity"]
            readings = pd.DataFrame({
                "Signal": [feature.replace("_", " ") for feature in chart_features],
                "Level (%)": [
                    ((float(input_values[feature]) - FEATURE_RANGES[feature][0]) / (FEATURE_RANGES[feature][1] - FEATURE_RANGES[feature][0])) * 100
                    for feature in chart_features
                ],
            }).set_index("Signal").clip(0, 100)
            st.bar_chart(readings, color="#45d5c7", height=260)
            st.caption("Shows where each reading sits between its lowest and highest allowed value. A high bar is not always bad; use it with the machine status.")
            st.markdown('</div>', unsafe_allow_html=True)
        with chart_right:
            st.markdown('<div class="panel"><div class="panel-title">Maintenance readings</div>', unsafe_allow_html=True)
            maintenance_features = ["Operating_Hours", "Days_Since_Maintenance", "Machine_Age", "Maintenance_History", "Load_Percentage"]
            maintenance = pd.DataFrame({
                "Signal": [feature.replace("_", " ") for feature in maintenance_features],
                "Value": [float(input_values[feature]) for feature in maintenance_features],
            }).set_index("Signal")
            st.bar_chart(maintenance, color="#f5b85d", height=260)
            st.caption("These readings help explain when the next service visit may be needed.")
            st.markdown('</div>', unsafe_allow_html=True)

with tab_fleet:
    st.markdown('<div class="panel-title">Check many machines</div>', unsafe_allow_html=True)
    st.write("Upload a CSV with the sensor columns below. The tool adds machine status, confidence, and the next action.")
    with st.expander("Needed CSV columns"):
        st.code(", ".join(feature_columns))
    uploaded = st.file_uploader("Upload your machine CSV here", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = sorted(set(feature_columns) - set(batch_df.columns))
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            X_batch = batch_df[feature_columns]
            preds = model.predict(X_batch)
            batch_df["Predicted_Status"] = label_encoder.inverse_transform(preds)
            probabilities = model.predict_proba(X_batch)
            batch_df["Confidence"] = probabilities.max(axis=1).round(3)
            batch_df["Maintenance_Priority"] = batch_df["Predicted_Status"].map({"Failure": "P0 - Dispatch", "Critical": "P1 - Inspect", "Warning": "P2 - Schedule", "Normal": "P3 - Monitor"})
            counts = batch_df["Predicted_Status"].value_counts().reindex(label_encoder.classes_, fill_value=0)
            st.success(f"Checked {len(batch_df):,} machines and created an ordered work list.")
            metrics = st.columns(4)
            for metric, status in zip(metrics, ["Failure", "Critical", "Warning", "Normal"]):
                metric.metric(status.upper(), int(counts.get(status, 0)))
            chart_col, table_col = st.columns([0.8, 1.2])
            with chart_col:
                st.bar_chart(counts, color="#f5b85d", height=250)
            with table_col:
                priority_order = {"P0 - Dispatch": 0, "P1 - Inspect": 1, "P2 - Schedule": 2, "P3 - Monitor": 3}
                display_df = batch_df.assign(_sort=batch_df["Maintenance_Priority"].map(priority_order)).sort_values(["_sort", "Confidence"], ascending=[True, False]).drop(columns="_sort")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.download_button("Download work list", batch_df.to_csv(index=False).encode("utf-8"), "predictive_maintenance_work_list.csv", "text/csv", type="primary")

with tab_reference:
    st.markdown('<div class="panel-title">Sensor guide</div>', unsafe_allow_html=True)
    reference = pd.DataFrame([{"Signal": feature, "Minimum": values[0], "Maximum": values[1], "Default": values[2]} for feature, values in FEATURE_RANGES.items()])
    st.dataframe(reference, use_container_width=True, hide_index=True)
    st.caption("These are the lowest, highest, and starting values used by the model. Use the same units as your sensors.")
