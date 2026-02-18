import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Pump Health", layout="wide", page_icon="🔧")

st.markdown('<h1 style="text-align:center; color:#111; font-size:2.3em; font-weight:900; margin-bottom:0.7em;">Multi Modal Predictive Maintenance Agent</h1>', unsafe_allow_html=True)


# --- Main form and XAI report ---
st.markdown(
    """
    <style>
    .uploaded-file-box {
        background: #22223b !important;
        color: #fff !important;
        border: 1.5px solid #44446b;
        border-radius: 7px;
        padding: 0.45em 1.1em 0.45em 1.1em;
        margin-bottom: 0.3em;
        font-size: 1.01em;
        font-weight: 500;
        display: flex;
        align-items: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.10);
        width: 100%;
    }
    .uploaded-file-cross {
        color: #f87171 !important;
        font-size: 1.2em;
        font-weight: 700;
        margin-left: 0.7em;
        cursor: pointer;
        background: none;
        border: none;
        outline: none;
        padding: 0 0.2em;
        transition: color 0.18s;
    }
    .uploaded-file-cross:hover {
        color: #f43f5e !important;
    }
    .block-container { padding-top: 8.5rem !important; padding-bottom: 2.5rem !important; }
    .stApp { background: #f8fafc; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background: #fff !important;
        font-size: 1.08em;
        padding: 0.7em 1em;
        border-radius: 8px;
        border: 1.5px solid #e2e8f0;
        color: #000 !important;
    }
    .stTextInput>div>div>input::placeholder {
        color: #94a3b8 !important;
    }
    /* Hide default +/- buttons and style native spinners as arrows */
    .stNumberInput button {
        display: none !important;
    }
    /* Style native number input spinners to look like arrows */
    .stNumberInput>div>div>input[type="number"]::-webkit-inner-spin-button,
    .stNumberInput>div>div>input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        appearance: none;
        opacity: 1;
        height: 1.2em;
        width: 1.2em;
        cursor: pointer;
        background: transparent;
        position: relative;
    }
    /* Create custom spinner appearance using pseudo-elements */
    .stNumberInput>div>div>input[type="number"]::-webkit-inner-spin-button::before,
    .stNumberInput>div>div>input[type="number"]::-webkit-outer-spin-button::before {
        content: '▲';
        position: absolute;
        top: -0.3em;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.6em;
        color: #64748b;
        line-height: 1;
    }
    .stNumberInput>div>div>input[type="number"]::-webkit-inner-spin-button::after,
    .stNumberInput>div>div>input[type="number"]::-webkit-outer-spin-button::after {
        content: '▼';
        position: absolute;
        bottom: -0.3em;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.6em;
        color: #64748b;
        line-height: 1;
    }
    /* For Firefox - hide default spinner */
    .stNumberInput>div>div>input[type="number"] {
        -moz-appearance: textfield;
    }
    /* Add padding for arrow area */
    .stNumberInput>div>div>input[type="number"] {
        padding-right: 2em !important;
    }
    /* Add arrow indicators using CSS pseudo-elements on wrapper */
    .stNumberInput>div>div {
        position: relative;
    }
    .stNumberInput>div>div::before {
        content: '▲';
        position: absolute;
        right: 0.6em;
        top: 0.4em;
        font-size: 0.55em;
        color: #94a3b8;
        pointer-events: none;
        z-index: 1;
        line-height: 1;
    }
    .stNumberInput>div>div::after {
        content: '▼';
        position: absolute;
        right: 0.6em;
        bottom: 0.4em;
        font-size: 0.55em;
        color: #94a3b8;
        pointer-events: none;
        z-index: 1;
        line-height: 1;
    }
    .stButton>button {
        font-size: 1.15em;
        padding: 0.7em 2.2em;
        border-radius: 8px;
        background: linear-gradient(90deg, #f1f5f9 0%, #e0e7ef 100%);
        color: #222;
        border: 1.5px solid #cbd5e1;
        margin-top: 1.2em;
    }
    .section-card {
        background: #fff;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        padding: 2.2em 2em 1.5em 2em;
        margin-bottom: 2.2em;
    }
    .section-header {
        font-size: 1.35em;
        font-weight: 700;
        color: #334155;
        margin-bottom: 1.2em;
        letter-spacing: 0.01em;
    }
    label, .stFileUploader label { font-size: 1.08em !important; color: #222 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Add JavaScript to create clickable arrow buttons
st.markdown("""
<script>
if (!window.numberInputArrowsInitialized) {
    window.numberInputArrowsInitialized = true;
    
    function initArrows() {
        const inputs = document.querySelectorAll('.stNumberInput input[type="number"]');
        inputs.forEach(input => {
            if (input.dataset.arrowsAdded === 'true') return;
            input.dataset.arrowsAdded = 'true';
            
            const wrapper = input.parentElement;
            if (!wrapper) return;
            
            wrapper.style.position = 'relative';
            
            // Remove existing arrows if any
            const existing = wrapper.querySelector('.custom-number-arrows');
            if (existing) existing.remove();
            
            const arrows = document.createElement('div');
            arrows.className = 'custom-number-arrows';
            arrows.style.cssText = 'position: absolute; right: 0.5em; top: 50%; transform: translateY(-50%); display: flex; flex-direction: column; gap: 0.05em; z-index: 10; pointer-events: none;';
            
            const up = document.createElement('button');
            up.innerHTML = '▲';
            up.type = 'button';
            up.style.cssText = 'background: transparent; border: none; color: #64748b; cursor: pointer; font-size: 0.6em; padding: 0.05em 0.25em; line-height: 1; pointer-events: auto; transition: color 0.2s;';
            up.onmouseover = function() { this.style.color = '#334155'; };
            up.onmouseout = function() { this.style.color = '#64748b'; };
            up.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                const step = parseFloat(input.getAttribute('step')) || parseFloat(input.step) || 0.01;
                const max = input.max ? parseFloat(input.max) : Infinity;
                const currentValue = parseFloat(input.value) || 0;
                const newValue = Math.min(max, currentValue + step);
                input.value = newValue.toFixed(2);
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            };
            
            const down = document.createElement('button');
            down.innerHTML = '▼';
            down.type = 'button';
            down.style.cssText = 'background: transparent; border: none; color: #64748b; cursor: pointer; font-size: 0.6em; padding: 0.05em 0.25em; line-height: 1; pointer-events: auto; transition: color 0.2s;';
            down.onmouseover = function() { this.style.color = '#334155'; };
            down.onmouseout = function() { this.style.color = '#64748b'; };
            down.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                const step = parseFloat(input.getAttribute('step')) || parseFloat(input.step) || 0.01;
                const min = input.min ? parseFloat(input.min) : -Infinity;
                const currentValue = parseFloat(input.value) || 0;
                const newValue = Math.max(min, currentValue - step);
                input.value = newValue.toFixed(2);
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            };
            
            arrows.appendChild(up);
            arrows.appendChild(down);
            wrapper.appendChild(arrows);
        });
    }
    
    // Initialize immediately
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initArrows);
    } else {
        initArrows();
    }
    
    // Watch for Streamlit updates
    const observer = new MutationObserver(function() {
        setTimeout(initArrows, 100);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Also run periodically to catch late-rendered inputs
    setTimeout(initArrows, 500);
    setTimeout(initArrows, 1500);
    setTimeout(initArrows, 3000);
}
</script>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header section-header-center">Pump Ingestion & Live Telemetry (Full Dashboard)</div>', unsafe_allow_html=True)
with st.form("pump_form", clear_on_submit=False):

    # Top row: left (pump_id), right (sensor fields)
    top_left, top_right = st.columns([1, 2], gap="large")
    with top_left:
        pump_id = st.text_input("Pump ID", value="", help="Required")
    with top_right:
        col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
        with col1:
            temperature = st.number_input("Temperature (°C)", value=None, step=0.01, format="%.2f", placeholder="0.00", help="Required")
        with col2:
            vibration = st.number_input("Vibration Level", value=None, step=0.01, format="%.2f", placeholder="0.00", help="Required")
        with col3:
            pressure = st.number_input("Pressure (bar)", value=None, step=0.01, format="%.2f", placeholder="0.00", help="Required")
        with col4:
            flow_rate = st.number_input("Flow Rate", value=None, step=0.01, format="%.2f", placeholder="0.00", help="Required")
        with col5:
            rpm = st.number_input("RPM", value=None, step=0.01, format="%.2f", placeholder="0.00", help="Required")
    st.markdown('</div>', unsafe_allow_html=True)

    # Transactional & Manuals Uploads
    st.markdown('<div class="section-header section-header-center">Transactional & Manuals Uploads</div>', unsafe_allow_html=True)
    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5, gap="medium")
    with tcol1:
        instruction_manual = st.file_uploader("OEM Manual (PDF)", type=["pdf"], label_visibility="visible", help="Limit 50MB per file", key="instruction_manual")
    with tcol2:
        work_done_logs = st.file_uploader("Work Done Logs (CSV)", type=["csv"], label_visibility="visible", help="Limit 50MB per file", key="work_done_logs")
    with tcol3:
        service_schedules = st.file_uploader("Service Schedules (CSV)", type=["csv"], label_visibility="visible", help="Limit 50MB per file", key="service_schedules")
    with tcol4:
        maintenance_requests = st.file_uploader("Maintenance Requests (CSV)", type=["csv"], label_visibility="visible", help="Limit 50MB per file", key="maintenance_requests")
    with tcol5:
        pump_image = st.file_uploader("Pump Image (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="visible", help="Limit 50MB per file", key="pump_image")
    st.markdown('</div>', unsafe_allow_html=True)

    bcol1, bcol2 = st.columns(2, gap="large")
    with bcol1:
        historical_logs = st.file_uploader("Historic Logs (CSV, max 50MB)", type=["csv"], label_visibility="visible", help="Limit 50MB per file", key="historical_logs")
        if st.session_state.get("historical_logs") is not None and getattr(st.session_state["historical_logs"], 'name', ''):
            if st.session_state["historical_logs"].size > 50 * 1024 * 1024:
                st.error("File too large. Please upload a file smaller than 50MB.")
                st.session_state["historical_logs"] = None
                st.stop()
    with bcol2:
        current_total_hours = st.number_input("Total Operational Hours", min_value=0, value=0, help="Required if no historic logs")

    st.markdown('<div style="height:2.2em;"></div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Analyze", use_container_width=True)

    # --- Manual ingestion button outside the form ---

# --- File removal buttons outside the form ---
remove_cols = st.columns(6, gap="small")
file_keys = [
    ("instruction_manual", "OEM Manual (PDF)"),
    ("work_done_logs", "Work Done Logs (CSV)"),
    ("service_schedules", "Service Schedules (CSV)"),
    ("maintenance_requests", "Maintenance Requests (CSV)"),
    ("pump_image", "Pump Image (JPG/PNG)"),
    ("historical_logs", "Historic Logs (CSV)")
]
for idx, (key, label) in enumerate(file_keys):
    with remove_cols[idx]:
        file_obj = st.session_state.get(key)
        if file_obj is not None and getattr(file_obj, 'name', ''):
            fname = file_obj.name
            st.markdown(f'<div class="uploaded-file-box">{fname}</div>', unsafe_allow_html=True)
            if st.button(f"✖ Remove {label}", key=f"remove_{key}", help=f"Remove {label}", use_container_width=True):
                # Use workaround for Streamlit widget state mutation error
                st.session_state.pop(key, None)
                st.rerun()


if submitted:
    if not pump_id or not str(pump_id).strip():
        st.error("Pump ID is required.")
        st.stop()
    # At least one of historic logs or total hours must be filled
    if (historical_logs is None or getattr(historical_logs, 'name', '') == '') and (current_total_hours is None or current_total_hours == 0):
        st.error("Either Historic Logs or Total Operational Hours is required.")
        st.stop()
    # All sensor fields must be filled and valid numbers
    sensor_fields = [temperature, vibration, rpm, pressure, flow_rate]
    if any(x is None for x in sensor_fields):
        st.error("All sensor fields are required.")
        st.stop()

    import time
    with st.spinner("Running prediction pipeline…"):
        try:
            multipart_data = {
                "asset_id": (None, pump_id),
                "current_total_hours": (None, str(current_total_hours)),
                "sensors": (None, json.dumps({
                    "temperature": temperature,
                    "vibration": vibration,
                    "rpm": rpm,
                    "pressure": pressure,
                    "flow_rate": flow_rate
                }))
            }
            if instruction_manual:
                multipart_data["instruction_manual"] = (f"{pump_id}_manual.pdf", instruction_manual, "application/pdf")
            if historical_logs:
                multipart_data["historical_logs"] = (f"{pump_id}_history.csv", historical_logs, "text/csv")
            if work_done_logs:
                multipart_data["work_done_logs"] = ("work_done_logs.csv", work_done_logs, "text/csv")
            if service_schedules:
                multipart_data["service_schedules"] = ("service_schedules.csv", service_schedules, "text/csv")
            if maintenance_requests:
                multipart_data["maintenance_requests"] = ("maintenance_requests.csv", maintenance_requests, "text/csv")
            if pump_image:
                multipart_data["pump_image"] = (pump_image.name, pump_image, pump_image.type)

            t0 = time.perf_counter()
            response = requests.post(f"{API_BASE}/analyze", files=multipart_data, timeout=120)
            t1 = time.perf_counter()
            inference_ms = int((t1 - t0) * 1000)
            if response.status_code != 200:
                st.error(f"API request failed: {response.text}")
                st.stop()
            data = response.json()
            data["inference_ms"] = inference_ms
        except Exception as exc:
            st.error(f"API request failed: {exc}")
            st.stop()

    st.markdown('<h3 style="color:#000;">XAI Maintenance Report</h3>', unsafe_allow_html=True)
    # Show asset_id (pump_id) first in the report
    # Always show the asset_id as the value the user entered (pump_id from form)
    st.markdown(f'<div style="color:#000;font-weight:600;">Asset ID</div><div style="color:#000;">{pump_id}</div>', unsafe_allow_html=True)
    # Show Fused Score as failure_probability
    st.markdown(f'<div style="color:#000;font-weight:600;">Failure Probability</div><div style="color:#000;">{data.get("fused_score", "—")}</div>', unsafe_allow_html=True)
    # Calculate estimated_time_to_breakdown_hours (ETTB)
    fused_score = data.get("fused_score")
    if fused_score is not None and fused_score > 0:
        ettb = 100.0 / fused_score
        ettb_str = f"{ettb:.1f} hours"
    else:
        ettb_str = "N/A"
    st.markdown(f'<div style="color:#000;font-weight:600;">Estimated Time to Breakdown</div><div style="color:#000;">{ettb_str}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#000;font-weight:600;">Status Label</div><div style="color:#000;">{data.get("status_label", "—").upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#000;font-weight:600;">Inference Time (ms)</div><div style="color:#000;">{data.get("inference_ms", "—")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#222; font-size:1.08em; margin-bottom:1em;">{data.get("explanation", "No explanation available.")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#000;font-weight:600;">Top Signals:</div><div style="color:#000;">{json.dumps(data.get("top_signals", []))}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#000;font-weight:600;">Action Items:</div><div style="color:#000;">{json.dumps(data.get("action_items", []))}</div>', unsafe_allow_html=True)
    # Show manual_context if present
    if data.get("manual_context"):
        st.markdown(f'<div style="color:#000;font-weight:600;">Manual Context:</div><div style="color:#000; white-space:pre-wrap;">{json.dumps(data["manual_context"], ensure_ascii=False, indent=2)}</div>', unsafe_allow_html=True)
    # Show anomaly_query if present
    if data.get("anomaly_query"):
        st.markdown(f'<div style="color:#000;font-weight:600;">Anomaly Query:</div><div style="color:#000; white-space:pre-wrap;">{data["anomaly_query"]}</div>', unsafe_allow_html=True)
    # Removed raw JSON output and divider
else:
    st.info("Fill in all required fields and click Analyze to generate a report.")



# --- Common Pump Failures Section (always visible, centered heading) ---

# --- Redesigned Common Pump Failures Section ---
st.markdown('''
.section-header-center {
    display: block;
    width: 100%;
    text-align: center !important;
    font-size: 2.1em !important;
    font-weight: 800 !important;
    margin-top: 5.2em !important;
    margin-bottom: 1.2em !important;
    color: #1a202c !important;
    letter-spacing: 0.01em;
}
.site-bottom-space { height: 3.5em; width: 100%; display: block; }
<style>
.failures-section {
    margin-top: 2.5em;
    display: flex;
    flex-wrap: wrap;
    gap: 2.2em 2.2em;
    justify-content: center;
}
.failure-card-modern {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1.5px solid #e2e8f0;
    padding: 1.3em 1.1em 1em 1.1em;
    margin-bottom: 0.5em;
    width: 32%;
    min-width: 230px;
    max-width: 340px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    transition: box-shadow 0.2s;
}
.failure-card-modern:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    border-color: #cbd5e1;
    transform: translateY(-1px) scale(1);
    transition: box-shadow 0.32s cubic-bezier(.4,0,.2,1), transform 0.32s cubic-bezier(.4,0,.2,1);
}
.failure-title-modern {
    font-size: 1em;
    font-weight: 700;
    color: #22223b;
    margin-bottom: 0.13em;
    margin-top: 0.15em;
}
.failure-desc-modern {
    color: #444;
    font-size: 0.92em;
    margin-bottom: 0.5em;
}
.failure-list-modern {
    margin: 0 0 0 0.2em;
    padding: 0;
    font-size: 0.89em;
    color: #222;
    list-style: none;
}
.failure-list-modern li {
    margin-bottom: 0.28em;
    display: flex;
    align-items: flex-start;
}
.failure-list-modern li:before {
    content: "✔";
    color: #22c55e;
    font-size: 1em;
    margin-right: 0.5em;
    margin-top: 0.08em;
}
.failures-heading {
    display: block;
    width: 100%;
    text-align: center;
    font-size: 2.1em;
    font-weight: 800;
    color: #1a202c;
    margin-bottom: 2.2em;
    margin-top: 0.7em;
    letter-spacing: 0.01em;
}
@media (max-width: 1100px) {
    .failure-card-modern { width: 100%; min-width:unset; max-width:unset; }
    .failures-section { flex-direction: column; gap: 1.2em 0; }
}
</style>
<div class="failures-heading">Common Pump Failures</div>
<div class="failures-section">
    <div class="failure-card-modern">
        <div class="failure-title-modern">Impeller Wear & Cavitation</div>
        <div class="failure-desc-modern">Cavitation damage and wear reduce pump efficiency and can cause catastrophic failure.</div>
        <ul class="failure-list-modern">
            <li>Cavitation erosion and pitting damage</li>
            <li>Impeller blade wear and corrosion</li>
            <li>Flow reduction and efficiency loss</li>
        </ul>
    </div>
    <div class="failure-card-modern">
        <div class="failure-title-modern">Bearing Failures</div>
        <div class="failure-desc-modern">Pump bearing deterioration causes vibration, noise, and eventual shaft seizure.</div>
        <ul class="failure-list-modern">
            <li>Lubrication breakdown and contamination</li>
            <li>Race and rolling element wear</li>
            <li>Thermal damage from overheating</li>
        </ul>
    </div>
    <div class="failure-card-modern">
        <div class="failure-title-modern">Mechanical Seal Leakage</div>
        <div class="failure-desc-modern">Seal failures cause leakage, contamination, and potential safety hazards.</div>
        <ul class="failure-list-modern">
            <li>Seal face wear and thermal cracking</li>
            <li>O-ring deterioration and hardening</li>
            <li>Spring and secondary seal failures</li>
        </ul>
    </div>
    <div class="failure-card-modern">
        <div class="failure-title-modern">Shaft Misalignment</div>
        <div class="failure-desc-modern">Misalignment causes premature bearing and coupling wear, vibration, and efficiency loss.</div>
        <ul class="failure-list-modern">
            <li>Angular and parallel misalignment</li>
            <li>Coupling wear and deterioration</li>
            <li>Increased vibration and noise</li>
        </ul>
    </div>
    <div class="failure-card-modern">
        <div class="failure-title-modern">Suction & Discharge Issues</div>
        <div class="failure-desc-modern">System pressure problems affect pump performance and can cause damage.</div>
        <ul class="failure-list-modern">
            <li>Net positive suction head problems</li>
            <li>Discharge pressure fluctuations</li>
            <li>Flow rate variations and instability</li>
        </ul>
    </div>
    <div class="failure-card-modern">
        <div class="failure-title-modern">Casing Wear & Corrosion</div>
        <div class="failure-desc-modern">Pump casing deterioration affects performance and can lead to catastrophic failure.</div>
        <ul class="failure-list-modern">
            <li>Internal erosion and wear patterns</li>
            <li>Corrosion from aggressive fluids</li>
            <li>Clearance increases and efficiency loss</li>
        </ul>
    </div>
 </div>
<div class="site-bottom-space"></div>
<div style="height:12em; width:100%; display:block;"></div>
''', unsafe_allow_html=True)
