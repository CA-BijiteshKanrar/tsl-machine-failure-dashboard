"""Streamlit interface for the trained machine failure classifier."""

import hashlib
import io
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from chat_component import suggested_chat_input
from features import RAW_FEATURES, engineer_features
from input_validation import TRAINING_RANGES, out_of_range_reasons
from azure_client import score_records

ARTIFACT = Path(__file__).parent / "artifacts" / "machine_failure_model.joblib"
DEFAULT_AZURE_URI = "https://tsl-mf-bijit-20260927-v2.centralindia.inference.ml.azure.com/score"
PRESET_QUESTIONS = {
    "Risk distribution": [
        "What does the predicted risk distribution tell us about this evaluation?",
        "How many machines meet the inspection threshold, and how should that guide review?",
        "Why do low risk scores not prove that a machine cannot fail?",
    ],
    "Product type alerts": [
        "Which product type has the highest alert rate, and how should I interpret it?",
        "How do alert counts and group sizes affect the product type alert rates?",
        "Does a higher alert rate mean that product type causes machine failure?",
    ],
    "Highest-risk machines": [
        "How should the highest-risk machines table guide inspection priority?",
        "What should be checked before acting on a high predicted failure probability?",
        "What are the limits of ranking machines with this synthetic model?",
    ],
    "Model and project": [
        "How was class imbalance considered when evaluating this model?",
        "How was the inspection alert threshold selected?",
        "What needs validation before using this model in a real steel plant?",
    ],
}
st.set_page_config(page_title="Machine failure risk", layout="wide")
st.title("Machine failure risk")
st.caption("Research demo using synthetic data. A predicted risk is an inspection cue, not a maintenance order.")
st.markdown(
    """<style>
    .evaluation-progress { display: inline-flex; align-items: center; gap: .55rem; font-size: .9rem; }
    .evaluation-progress__icon { width: 1rem; height: 1rem; border: 2px solid #7777;
        border-top-color: #ff4b4b; border-radius: 50%; animation: eval-spin .75s linear infinite; }
    [data-testid="stStatusWidget"] { display: none; }
    @keyframes eval-spin { to { transform: rotate(360deg); } }
    </style>""",
    unsafe_allow_html=True,
)

def setting(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def focus_chart_questions(topic):
    st.session_state["question_topic"] = topic
    st.session_state["chat_focus_token"] = st.session_state.get("chat_focus_token", 0) + 1


def queue_evaluation(signature):
    st.session_state["pending_evaluation"] = signature
    st.session_state.pop("evaluation_error", None)


configured_uri = setting("AZURE_ML_SCORING_URI")
configured_key = setting("AZURE_ML_ENDPOINT_KEY")
gemini_key = setting("GEMINI_API_KEY")
with st.sidebar:
    scoring_options = (["Local model", "Azure ML endpoint"] if ARTIFACT.exists()
                       else ["Azure ML endpoint"])
    scoring_source = st.radio(
        "Scoring source",
        scoring_options,
        index=scoring_options.index("Azure ML endpoint") if configured_uri and configured_key else 0,
    )
    if scoring_source == "Azure ML endpoint":
        scoring_uri = configured_uri or DEFAULT_AZURE_URI
        endpoint_key = configured_key
        st.caption("Azure scoring uses private server configuration.")
    else:
        scoring_uri = None
        endpoint_key = None
use_azure = scoring_source == "Azure ML endpoint"
azure_ready = bool(scoring_uri and endpoint_key) if use_azure else True
with st.sidebar:
    st.caption("Gemini chat uses a private server key." if gemini_key else "Gemini chat is not configured on this server.")
if not use_azure and not ARTIFACT.exists():
    st.error("Set Azure endpoint secrets or place the saved model in artifacts/.")
    st.stop()


@st.cache_resource
def load_bundle():
    # Only load the artifact produced by this project's notebook; joblib is not safe for untrusted files.
    return joblib.load(ARTIFACT)


bundle = load_bundle() if ARTIFACT.exists() else None
model = bundle["model"] if bundle else None
threshold = float(bundle["threshold"]) if bundle else None

if use_azure and not azure_ready:
    st.warning(
        "Azure ML scoring is not configured on this server. Add AZURE_ML_ENDPOINT_KEY "
        "to private Streamlit secrets or the server environment. The app remains available below."
    )


def predict(frame):
    if use_azure:
        raw = frame.loc[:, RAW_FEATURES].astype(object)
        records = raw.where(pd.notna(raw), None).to_dict("records")
        probabilities, alerts = [], []
        remote_threshold = None
        for start in range(0, len(records), 200):
            result = score_records(records[start:start + 200], scoring_uri, endpoint_key)
            if remote_threshold is not None and float(result["threshold"]) != remote_threshold:
                raise RuntimeError("Azure endpoint returned inconsistent thresholds")
            remote_threshold = float(result["threshold"])
            probabilities.extend(result["failure_probability"])
            alerts.extend(result["inspection_alert"])
        return probabilities, alerts, remote_threshold
    probability = model.predict_proba(engineer_features(frame))[:, 1]
    return probability, (probability >= threshold).astype(int), threshold

with st.sidebar:
    st.subheader("Model")
    st.write(f"Azure ML endpoint" if use_azure else bundle["model_name"])
    if not use_azure and threshold is not None:
        st.metric("Alert threshold", f"{threshold:.3f}")
    else:
        st.write("The endpoint returns the selected alert threshold with each score.")
    st.write("The threshold was chosen on labeled validation data for F2 score.")
    st.write("Failure-type flags and product IDs are excluded from predictions.")

tab_single, tab_batch, tab_ask = st.tabs(
    ["Single machine", "Import Evaluation Data", "GenAI Insights"]
)

with tab_single:
    st.caption("Single-machine estimates require values within the ranges observed in this model's training data.")
    with st.form("machine"):
        c1, c2, c3 = st.columns(3)
        with c1:
            machine_type = st.selectbox("Product type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
        with c2:
            process_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)
            rpm = st.number_input("Rotational speed [rpm]", value=1500.0, min_value=0.0, step=1.0)
        with c3:
            torque = st.number_input("Torque [Nm]", value=40.0, min_value=0.0, step=0.1)
            wear = st.number_input("Tool wear [min]", value=100.0, min_value=0.0, step=1.0)
        submitted = st.form_submit_button("Estimate risk", disabled=not azure_ready)
    if submitted:
        row = pd.DataFrame([dict(zip(RAW_FEATURES, [machine_type, air_temp, process_temp, rpm, torque, wear]))])
        out_of_range = [
            f"{column}: {row.at[0, column]:,.1f} (training range {low:,.1f}–{high:,.1f})"
            for column, (low, high) in TRAINING_RANGES.items()
            if not low <= row.at[0, column] <= high
        ]
        if out_of_range:
            st.error(
                "No risk estimate is available for inputs outside the training ranges. "
                "A model score here would be misleading."
            )
            st.markdown("**Out-of-range inputs**\n\n" + "\n".join(f"- {item}" for item in out_of_range))
        else:
            try:
                probability, alerts, _ = predict(row)
                st.metric("Estimated failure probability", f"{float(probability[0]):.1%}")
                st.write("Inspection alert" if alerts[0] else "Below alert threshold")
                st.caption("This score is based on synthetic examples and has not been calibrated for a real plant.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not score this machine: {exc}")

with tab_batch:
    st.write("Upload operational data to score machines and prioritize inspections.")
    uploaded = st.file_uploader("Upload evaluation data", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        contents = uploaded.getvalue()
        upload_digest = hashlib.sha256(contents).hexdigest()
        prior = st.session_state.get("evaluation")
        if prior and prior["signature"][0] != upload_digest:
            st.session_state.pop("evaluation", None)
        is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
        sheet = None
        try:
            if is_excel:
                with pd.ExcelFile(io.BytesIO(contents)) as workbook:
                    sheet = st.selectbox("Excel sheet", workbook.sheet_names)
                excel_frame = pd.read_excel(io.BytesIO(contents), sheet_name=sheet)
                converted_csv = excel_frame.to_csv(index=False).encode("utf-8")
                batch = pd.read_csv(io.BytesIO(converted_csv))
                st.download_button(
                    "Download converted CSV", converted_csv,
                    f"{Path(uploaded.name).stem}.csv", "text/csv",
                )
                st.caption("The selected Excel sheet is converted to CSV in this app before scoring.")
            else:
                batch = pd.read_csv(io.BytesIO(contents))
            missing = sorted(set(RAW_FEATURES) - set(batch.columns))
            if missing:
                st.error(f"Missing required columns: {missing}")
            elif batch.empty:
                st.error("The uploaded sheet or CSV has no rows to score.")
            else:
                validation_reason = out_of_range_reasons(batch)
                eligible = validation_reason.eq("")
                eligible_count = int(eligible.sum())
                excluded_count = len(batch) - eligible_count
                signature = (
                    upload_digest, sheet, scoring_source, scoring_uri
                )
                prior = st.session_state.get("evaluation")
                if prior and prior["signature"] != signature:
                    st.session_state.pop("evaluation", None)
                elif prior and excluded_count and "excluded_rows" not in prior:
                    # Earlier app versions scored every row, including unsupported ones.
                    st.session_state.pop("evaluation", None)
                if st.session_state.get("pending_evaluation") not in (None, signature):
                    st.session_state.pop("pending_evaluation", None)
                st.caption(
                    f"{eligible_count:,} of {len(batch):,} rows within the training ranges "
                    "and ready for evaluation."
                )
                if excluded_count:
                    excluded_label = "row" if excluded_count == 1 else "rows"
                    excluded_verb = "contains" if excluded_count == 1 else "contain"
                    excluded_pronoun = "It" if excluded_count == 1 else "They"
                    st.warning(
                        f"{excluded_count:,} {excluded_label} {excluded_verb} values outside the training data or "
                        f"an unsupported product type. {excluded_pronoun} will not be sent to the model or included "
                        "in the risk charts."
                    )
                    excluded_rows = batch.loc[~eligible].copy()
                    excluded_rows.insert(0, "source_row", excluded_rows.index + 1)
                    excluded_rows["validation_reason"] = validation_reason.loc[~eligible]
                    st.download_button(
                        "Download excluded rows", excluded_rows.to_csv(index=False),
                        "machine_failure_excluded_rows.csv", "text/csv",
                    )
                if not eligible_count:
                    st.error("No rows can be scored. Correct the excluded rows and upload the file again.")
                pending = st.session_state.get("pending_evaluation") == signature
                action_col, progress_col = st.columns([1, 9], gap="small", vertical_alignment="center")
                with action_col:
                    st.button(
                        "Evaluate data", type="primary", key="evaluate_data",
                        disabled=not azure_ready or pending or not eligible_count,
                        on_click=queue_evaluation, args=(signature,),
                    )
                if pending:
                    try:
                        with progress_col:
                            st.markdown(
                                '<div class="evaluation-progress" role="status">'
                                '<span class="evaluation-progress__icon" aria-hidden="true"></span>'
                                'Evaluating data…</div>',
                                unsafe_allow_html=True,
                            )
                        scoreable = batch.loc[eligible]
                        probability, alerts, selected_threshold = predict(scoreable)
                        output = pd.DataFrame({
                            "source_row": range(1, len(batch) + 1),
                            "id": batch["id"] if "id" in batch.columns else range(len(batch)),
                            "failure_probability": float("nan"),
                            "inspection_alert": pd.Series(pd.NA, index=batch.index, dtype="Int64"),
                            "validation_status": "Scored",
                            "validation_reason": validation_reason,
                        })
                        output.loc[eligible, "failure_probability"] = probability
                        output.loc[eligible, "inspection_alert"] = pd.Series(
                            alerts, index=scoreable.index, dtype="Int64"
                        )
                        output.loc[~eligible, "validation_status"] = "Excluded"
                        chart_data = scoreable.loc[:, ["Type", "Tool wear [min]"]].copy()
                        chart_data["id"] = output.loc[eligible, "id"]
                        chart_data["failure_probability"] = output.loc[eligible, "failure_probability"]
                        chart_data["inspection_alert"] = output.loc[eligible, "inspection_alert"]
                        st.session_state["evaluation"] = {
                            "signature": signature,
                            "filename": uploaded.name,
                            "output": output,
                            "chart_data": chart_data,
                            "threshold": selected_threshold,
                            "uploaded_rows": len(batch),
                            "excluded_rows": excluded_count,
                        }
                        st.session_state["chat_messages"] = []
                    except (ValueError, KeyError, RuntimeError) as exc:
                        st.session_state["evaluation_error"] = f"Could not score the file: {exc}"
                    finally:
                        st.session_state.pop("pending_evaluation", None)
                    st.rerun()
                if st.session_state.get("evaluation_error"):
                    st.error(st.session_state.pop("evaluation_error"))
                evaluation = st.session_state.get("evaluation")
                if evaluation and evaluation["signature"] == signature:
                    output = evaluation["output"]
                    st.metric("Inspection alerts", int(output["inspection_alert"].sum()))
                    if evaluation.get("excluded_rows", 0):
                        excluded_total = evaluation["excluded_rows"]
                        excluded_label = "row" if excluded_total == 1 else "rows"
                        excluded_verb = "has" if excluded_total == 1 else "have"
                        st.caption(
                            f"Predictions cover {len(evaluation['chart_data']):,} rows; "
                            f"{excluded_total:,} excluded {excluded_label} {excluded_verb} blank scores "
                            "and appear in the downloadable results."
                        )
                    st.dataframe(
                        output.sort_values("failure_probability", ascending=False).head(100),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download predictions", output.to_csv(index=False),
                        "machine_failure_predictions.csv", "text/csv",
                    )
        except (ValueError, ImportError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Could not read the evaluation file: {exc}")

with tab_ask:
    st.subheader("Evaluation insights")
    evaluation = st.session_state.get("evaluation")
    if evaluation:
        st.markdown(
            """<style>
            .st-key-risk_distribution_card,
            .st-key-type_alert_card,
            .st-key-top_machines_card {
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.20);
            }
            .st-key-ask_risk button,
            .st-key-ask_type button,
            .st-key-ask_top button {
                background: transparent;
                border: 0;
                box-shadow: none;
                color: #79c4ff;
                min-height: 0;
                padding: 0;
                text-decoration: underline;
            }
            .st-key-ask_risk button:hover,
            .st-key-ask_type button:hover,
            .st-key-ask_top button:hover {
                background: transparent;
                color: #b8e0ff;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        scored = evaluation["chart_data"]
        alert_count = int(scored["inspection_alert"].sum())
        alert_rate = alert_count / len(scored)
        c1, c2, c3 = st.columns(3)
        c1.metric("Machines evaluated", f"{len(scored):,}")
        c2.metric("Inspection alerts", f"{alert_count:,}")
        c3.metric("Alert rate", f"{alert_rate:.1%}")
        st.caption(
            f"Latest file: {evaluation['filename']} · "
            f"Alert threshold: {evaluation['threshold']:.3f} · "
            f"Rows excluded before scoring: {evaluation.get('excluded_rows', 0):,}"
        )

        left, right = st.columns(2)
        with left:
            with st.container(border=True, key="risk_distribution_card"):
                st.markdown("#### Predicted risk distribution")
                risk_bins = pd.cut(
                    scored["failure_probability"],
                    bins=[i / 10 for i in range(11)],
                    include_lowest=True,
                )
                risk_counts = risk_bins.value_counts(sort=False)
                risk_counts.index = [f"{i * 10}–{(i + 1) * 10}%" for i in range(10)]
                st.bar_chart(
                    risk_counts.rename("Machines"),
                    x_label="Predicted risk", y_label="Machines",
                )
                most_common_band = risk_counts.idxmax()
                most_common_share = risk_counts.max() / len(scored)
                st.markdown(
                    f"**Insight:** {most_common_share:.1%} of machines fall in the "
                    f"{most_common_band} risk band. {alert_count:,} machines "
                    f"({alert_rate:.1%}) meet the inspection threshold. Higher bands "
                    "are priorities for review, not confirmed failures."
                )
                st.button(
                    "Ask about this", key="ask_risk", type="tertiary",
                    on_click=focus_chart_questions, args=("Risk distribution",),
                    help="Choose a related question in the Gemini chat below",
                    disabled=not bool(gemini_key),
                )
        with right:
            with st.container(border=True, key="type_alert_card"):
                st.markdown("#### Inspection alert rate by product type")
                type_summary = (
                    scored.assign(Type=scored["Type"].fillna("Unknown"))
                    .groupby("Type")["inspection_alert"]
                    .agg(machines="size", alerts="sum")
                )
                type_summary["alert_rate"] = (
                    type_summary["alerts"] / type_summary["machines"] * 100
                )
                st.bar_chart(
                    type_summary["alert_rate"].rename("Alert rate (%)"),
                    x_label="Product type", y_label="Alert rate (%)",
                )
                if alert_count:
                    highest_type = type_summary["alert_rate"].idxmax()
                    highest = type_summary.loc[highest_type]
                    st.markdown(
                        f"**Insight:** Type {highest_type} has the highest alert rate "
                        f"at {highest['alert_rate']:.1f}% "
                        f"({int(highest['alerts']):,} of {int(highest['machines']):,} machines). "
                        "The rate compares how often each type is flagged; it does not show that type causes failure."
                    )
                else:
                    st.markdown(
                        "**Insight:** No machines in this upload meet the alert threshold. "
                        "Each bar compares the share flagged within a product type."
                    )
                st.button(
                    "Ask about this", key="ask_type", type="tertiary",
                    on_click=focus_chart_questions, args=("Product type alerts",),
                    help="Choose a related question in the Gemini chat below",
                    disabled=not bool(gemini_key),
                )

        with st.container(border=True, key="top_machines_card"):
            st.markdown("#### Highest-risk machines")
            st.dataframe(
                scored.sort_values("failure_probability", ascending=False).head(10),
                width="stretch",
                hide_index=True,
            )
            st.markdown(
                "**Insight:** These machines have the highest model-estimated risk "
                "in the uploaded file. Review their operating context first; a score alone "
                "does not confirm a failure or prescribe maintenance."
            )
            st.button(
                "Ask about this", key="ask_top", type="tertiary",
                on_click=focus_chart_questions, args=("Highest-risk machines",),
                help="Choose a related question in the Gemini chat below",
                disabled=not bool(gemini_key),
            )
    else:
        st.info("Import and evaluate a CSV or Excel file in the second tab to see risk charts here.")

    st.subheader("Ask Gemini")
    st.write("Start typing for suggested questions, or ask your own about the model and its results.")
    st.caption("The app sends aggregate evaluation statistics to Gemini, without raw rows or configured API keys.")
    if not gemini_key:
        st.info("Gemini chat is unavailable until GEMINI_API_KEY is set in private server configuration.")
    topic = st.session_state.get("question_topic")
    if topic:
        st.caption(f"Suggested topic: {topic}. Choose a suggestion in the question field to ask Gemini.")
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if st.session_state["chat_messages"] and st.button("Clear conversation"):
        st.session_state["chat_messages"] = []
        st.rerun()
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    suggestions = PRESET_QUESTIONS.get(topic) or [
        item for questions in PRESET_QUESTIONS.values() for item in questions
    ]
    chat_entry = suggested_chat_input(
        data={
            "suggestions": suggestions,
            "focus_token": st.session_state.get("chat_focus_token", 0),
            "disabled": not bool(gemini_key),
        },
        key="gemini_question_entry",
        on_question_change=lambda: None,
    )
    question = chat_entry.question
    if question:
        previous_messages = st.session_state["chat_messages"][-8:]
        st.session_state["chat_messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            context = {
                "model": bundle["model_name"] if bundle else "Registered Azure ML model",
                "threshold": threshold,
                "alert_meaning": "A score at or above the threshold is an inspection cue, not a maintenance order.",
                "data_scope": "Synthetic examples; no real plant calibration or established prediction horizon.",
                "input_features": RAW_FEATURES,
                "threshold_selection": "Chosen on labeled validation data for F2 score.",
                "class_imbalance": (
                    "About 1.57% of training rows are failures. Splits are stratified; "
                    "the selected Random Forest uses balanced_subsample class weights. "
                    "Validation and holdout retain natural prevalence. Average precision, "
                    "F2, precision, recall, and false alert counts are reported."
                ),
                "validation_metrics": bundle["validation_metrics"] if bundle else "Not loaded in this app session",
                "holdout_metrics": bundle["holdout_metrics"] if bundle else "Not loaded in this app session",
            }
            if evaluation:
                scored = evaluation["chart_data"]
                labels = [f"{i * 10}–{(i + 1) * 10}%" for i in range(10)]
                risk_band_counts = pd.cut(
                    scored["failure_probability"],
                    bins=[i / 10 for i in range(11)], labels=labels, include_lowest=True,
                ).value_counts(sort=False)
                type_counts = (
                    scored.assign(Type=scored["Type"].fillna("Unknown"))
                    .groupby("Type")["inspection_alert"]
                    .agg(machines="size", alerts="sum")
                )
                context["latest_evaluation"] = {
                    "uploaded_machines": evaluation.get("uploaded_rows", len(scored)),
                    "machines_scored": len(scored),
                    "rows_excluded_before_scoring": evaluation.get("excluded_rows", 0),
                    "inspection_alerts": int(scored["inspection_alert"].sum()),
                    "mean_predicted_risk": float(scored["failure_probability"].mean()),
                    "risk_band_counts": {band: int(count) for band, count in risk_band_counts.items()},
                    "product_type_counts": {
                        str(machine_type): {
                            "machines": int(row["machines"]), "alerts": int(row["alerts"]),
                        }
                        for machine_type, row in type_counts.iterrows()
                    },
                }
            history = "\n".join(
                f"{message['role']}: {message['content']}" for message in previous_messages
            )
            prompt = (
                "You explain a synthetic machine-failure research model to a maintenance stakeholder. "
                "Use only these results and general definitions. Do not invent plant data, causal findings, "
                "operational savings, or maintenance instructions. Uploaded evaluation rows are unlabeled, "
                "so their scores and alerts are predictions, not observed failures. Explain comparisons "
                "using the supplied counts and say when an answer needs more evidence.\n"
                f"Results: {context}\nRecent conversation:\n{history}\nUser: {question}"
            )
            with st.chat_message("assistant"):
                with st.spinner("Gemini is replying…"):
                    response = client.models.generate_content(
                        model=setting("GEMINI_MODEL") or "gemini-3.5-flash-lite", contents=prompt
                    )
                answer = response.text or "Gemini returned no text."
                st.markdown(answer)
        except Exception:
            answer = "Gemini could not answer right now. Check the private API key and model access, then retry."
            with st.chat_message("assistant"):
                st.error(answer)
        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
