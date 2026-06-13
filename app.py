import streamlit as st
import pandas as pd
import json
import os
import re
import subprocess

# Set page configuration
st.set_page_config(
    page_title="Microclimate AI DSS",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .status-safe { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-stress { color: #dc3545; }
    .section-header {
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
REPORT_PATH = "outputs/final_execution_report.txt"
METRICS_PATH = "outputs/final_metrics.json"
PLOTS_DIR = "outputs/plots"

def parse_report(filepath):
    """Parses the text execution report to extract key information."""
    data = {
        "prediction": "N/A",
        "irrigation": "N/A",
        "irrigation_reason": "N/A",
        "crop_class": "N/A",
        "crop_reason": "N/A",
        "input_data": None,
        "research_summary": ""
    }
    
    if not os.path.exists(filepath):
        return data
        
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Core prediction
        match = re.search(r"Predicted Soil Moisture for Next Day:\s*([\d\.]+)\s*%", content)
        if match:
            data['prediction'] = f"{match.group(1)} %"
            
        # Irrigation decision
        match = re.search(r"Recommendation:\s*(.*?)\nReason:\s*(.*)", content)
        if match:
            data['irrigation'] = match.group(1).strip()
            data['irrigation_reason'] = match.group(2).strip()
            
        # Crop condition
        match = re.search(r"Class:\s*(.*?)\nExplanation:\s*(.*)", content)
        if match:
            data['crop_class'] = match.group(1).strip()
            data['crop_reason'] = match.group(2).strip()
            
        # Extract table block
        match = re.search(r"5\. Input Data Used for Final Inference\n(.*?)\n\n6\.", content, re.DOTALL)
        if match:
            table_text = match.group(1).strip().split('\n')
            if len(table_text) > 1:
                # Parse header
                headers = [h for h in table_text[0].split() if h]
                # If first column is empty (index), handling it
                if "date" not in headers[0].lower() and len(headers) < len(table_text[1].split()):
                     headers = ["index"] + headers
                
                rows = []
                for line in table_text[1:]:
                    parts = line.split()
                    if parts:
                        rows.append(parts)
                try:
                    df = pd.DataFrame(rows, columns=headers)
                    if "index" in df.columns:
                        df = df.drop(columns=["index"])
                    data['input_data'] = df
                except Exception as e:
                    # Fallback simple string format if dataframe fails
                    data['input_data'] = "Error framing table data"
                    
        # Extract research summary
        match = re.search(r"7\. Research Summary\n(.*)", content, re.DOTALL)
        if match:
            data['research_summary'] = match.group(1).strip()
            
    except Exception as e:
         st.error(f"Error reading report file: {e}")
         
    return data

def run_inference():
    """Executes the main pipeline."""
    try:
        # Use python from the environment
        result = subprocess.run(
            ["python", "main.py"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        if result.returncode == 0:
            st.success("Pipeline executed successfully!")
        else:
            st.error(f"Pipeline execution failed. Error:\n{result.stderr}")
    except Exception as e:
        st.error(f"Failed to run subprocess: {e}")


def main():
    # -------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933092.png", width=100) # Placeholder icon
        st.title("Control Panel")
        st.markdown("Agentic AI-Based Irrigation Advisory for Smallholder Farmers.")
        
        st.markdown("---")
        if st.button("▶ Run New Inference", use_container_width=True, type="primary"):
            with st.spinner("Executing prediction pipeline..."):
                 run_inference()
                 st.rerun()
                 
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("[Core Predictions](#core-prediction-panel)")
        st.markdown("[Input Data](#input-data-table)")
        st.markdown("[Model Performance](#model-performance-comparison)")
        st.markdown("[Visualizations](#visualization-section)")
        st.markdown("[Research Summary](#research-summary)")
        st.markdown("[AI Farm Assistant](#ai-farm-assistant)")

    # -------------------------------------------------------------
    # Main Content
    # -------------------------------------------------------------
    
    # ⭐ A. Project Header
    st.title("🌱 Microclimate AI Decision Support System")
    st.markdown("*Agentic AI-Based Irrigation Advisory for Smallholder Farmers*")
    st.markdown("This dashboard presents the model predictions and agentic decision outputs in a farmer-friendly interface, supporting academic demonstration and actionable insights.")
    
    # Load data
    report_data = parse_report(REPORT_PATH)
    
    # ⭐ B. Core Prediction Panel
    st.markdown("<h2 class='section-header' id='core-prediction-panel'>🎯 Core Prediction Panel</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Formatting styles based on class
    crop_class = report_data.get('crop_class', 'UNKNOWN')
    crop_color_class = "status-safe" if crop_class == "SAFE" else "status-warning" if crop_class == "WARNING" else "status-stress" if crop_class == "STRESS" else ""
    
    irr_class = report_data.get('irrigation', 'UNKNOWN')
    irr_color_class = "status-stress" if "REQUIRED" in irr_class and "NOT" not in irr_class else "status-safe"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Soil Moisture (Next Day)</h4>
            <div class="metric-value">{report_data.get('prediction', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Crop Condition</h4>
            <div class="metric-value {crop_color_class}">{crop_class}</div>
            <p style="font-size: 0.9em; color: #6c757d;">{report_data.get('crop_reason', '')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Irrigation Recommendation</h4>
            <div class="metric-value {irr_color_class}">{irr_class.replace('IRRIGATION ', '')}</div>
            <p style="font-size: 0.9em; color: #6c757d;">{report_data.get('irrigation_reason', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
    # ⭐ C. Input Data Table
    st.markdown("<h2 class='section-header' id='input-data-table'>📊 Input Data (Last 15 Days)</h2>", unsafe_allow_html=True)
    if isinstance(report_data['input_data'], pd.DataFrame):
        st.dataframe(report_data['input_data'], use_container_width=True)
    else:
        st.info("Input data sequence not found in the current report execution.")

    # ⭐ D. Model Performance Comparison
    st.markdown("<h2 class='section-header' id='model-performance-comparison'>📈 Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
                
            metrics_df = pd.DataFrame(metrics_data).T
            
            # Display Table and Chart side-by-side
            m_col1, m_col2 = st.columns([1, 2])
            
            with m_col1:
                st.markdown("#### Evaluation Metrics Table")
                # Format to 4 decimal places
                st.table(metrics_df.style.format("{:.4f}"))
                
            with m_col2:
                st.markdown("#### Metrics Comparison Chart")
                st.bar_chart(metrics_df)
        else:
            st.warning("Metrics file not found. Run inference to generate.")
    except Exception as e:
         st.error(f"Error loading metrics: {e}")

    # ⭐ E. Visualization Section
    st.markdown("<h2 class='section-header' id='visualization-section'>🖼️ Visualizations & Model Evaluation</h2>", unsafe_allow_html=True)
    
    if os.path.exists(PLOTS_DIR):
        plot_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
        
        # Mapping nice titles
        title_mapping = {
            "loss_curve.png": "Standard LSTM Training Loss",
            "tffn_loss_curve.png": "Temporal–Feature Fusion Network (TFFN) Training Loss",
            "actual_vs_predicted.png": "Actual vs Predicted Soil Moisture",
            "exp1_comparison.png": "Standard vs Temporal–Feature Fusion Network (TFFN) Comparison",
            "exp2_seq_length.png": "Sequence Length Impact Experiment",
            "exp3_scaling.png": "Feature Scaling Impact Experiment",
            "classifier_accuracy_curve.png": "Classification Accuracy Curve"
        }
        
        if plot_files:
            # Create a 2-column grid for plots
            cols = st.columns(2)
            
            for index, plot_file in enumerate(plot_files):
                img_path = os.path.join(PLOTS_DIR, plot_file)
                plot_title = title_mapping.get(plot_file, plot_file)
                
                with cols[index % 2]:
                    st.image(img_path, caption=plot_title, use_container_width=True)
        else:
            st.info("No plots found in the outputs directory.")
    else:
        st.warning("Plots directory not found.")
        
    # ⭐ F. Research Summary Panel
    st.markdown("<h2 class='section-header' id='research-summary'>🔬 Research Summary</h2>", unsafe_allow_html=True)
    
    if report_data['research_summary']:
         with st.expander("View Full Execution Report Conclusions", expanded=True):
             # Format the summary to be bulleted or preserved nicely
             summary_lines = report_data['research_summary'].split('\n')
             for line in summary_lines:
                 if line.strip():
                     st.write(line)
    else:
         st.info("Research summary not fully populated in recent pipeline run.")

    # ⭐ G. AI Farm Assistant
    st.markdown("<h2 class='section-header' id='ai-farm-assistant'>🤖 AI Farm Assistant</h2>", unsafe_allow_html=True)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("LLM service unavailable. Please set the GEMINI_API_KEY environment variable to enable the AI Farm Assistant.")
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # React to user input
            if prompt := st.chat_input("Ask a question about your farm (e.g., 'Do I need to irrigate today?')..."):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Context for Gemini
                context = f"""
                You are an expert Agricultural AI Assistant. Your purpose is to provide conversational decision support for farmers.
                You must ONLY answer questions related to agriculture, irrigation guidance, crop condition, soil moisture, weather impact, and farming recommendations.
                If the user asks a question outside this domain, respond exactly with: "I can assist with irrigation, crop condition, and field management questions."
                
                Be clear, farmer-friendly, reference the system outputs below, and avoid technical jargon. Include reasoning for your answers based on the context.
                
                SYSTEM OUTPUTS:
                - Predicted Soil Moisture: {report_data.get('prediction', 'N/A')}
                - Crop Condition: {report_data.get('crop_class', 'N/A')} ({report_data.get('crop_reason', 'N/A')})
                - Irrigation Recommendation: {report_data.get('irrigation', 'N/A')} ({report_data.get('irrigation_reason', 'N/A')})
                """
                
                try:
                    # Configure the model and start a new chat session to generate a response
                    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=context)
                    
                    # Convert Streamlit history to Gemini format
                    # 'user' and 'model' are the supported roles
                    history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in st.session_state.messages[:-1]]
                    
                    chat = model.start_chat(history=history)
                    
                    with st.spinner("AI Assistant is thinking..."):
                        response = chat.send_message(prompt)
                        
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")
                    
        except ImportError:
            st.error("The `google-generativeai` package is missing. Please run `pip install google-generativeai` to enable the chatbot.")
        except Exception as e:
            st.error(f"Could not initialize the LLM service: {e}")

if __name__ == "__main__":
    main()
