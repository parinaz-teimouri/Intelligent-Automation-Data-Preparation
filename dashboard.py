import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from data_preparation import AutoDataPreparer
import time

st.set_page_config(page_title="Data Preparation Dashboard", layout="wide")

def load_report(report_path):
    with open(report_path, 'r') as f:
        return json.load(f)

def get_all_reports():
    reports = []
    # Search in current directory and output directory
    search_dirs = ['.', 'output']
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith('report_') and f.endswith('.json'):
                    reports.append(os.path.join(d, f))
    return sorted(reports, reverse=True)

st.title("🤖 Intelligent Data Preparation Dashboard")

# --- SIDEBAR: NAVIGATION & CONTROLS ---
st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode", ["Live Run", "Historical Reports"])

if mode == "Live Run":
    st.header("⚡ Live Pipeline Execution")
    
    uploaded_file = st.file_uploader("Upload your raw CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Optional: Target Column
        df_preview = pd.read_csv(file_path, nrows=5)
        st.write("Preview:")
        st.dataframe(df_preview)
        
        target_col = st.selectbox("Select Target Column (Optional - for Supervised Learning)", ["None"] + list(df_preview.columns))
        target_col = None if target_col == "None" else target_col
        
        if st.button("🚀 Run Auto-Preparation"):
            with st.spinner("Initializing Pipeline..."):
                try:
                    # Initialize Preparer
                    preparer = AutoDataPreparer(file_path, target_col=target_col)
                    
                    # Define Output Paths
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    processed_path = os.path.join(output_dir, f"processed_{base_name}.csv")
                    
                    # Run Pipeline
                    preparer.run_pipeline(processed_path)
                    
                    st.success("Pipeline Finished Successfully!")
                    
                    # Auto-select the new report for visualization
                    report_filename = f"report_processed_{base_name}.json"
                    report_path = os.path.join(output_dir, report_filename)
                    
                    if os.path.exists(report_path):
                        st.session_state['current_report'] = report_path
                        st.rerun() # Refresh to show results
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.exception(e)

# --- REPORT VISUALIZATION LOGIC ---
report_path_to_load = None

if mode == "Historical Reports":
    reports = get_all_reports()
    if not reports:
        st.warning("No historical reports found.")
    else:
        report_path_to_load = st.sidebar.selectbox("Choose a report:", reports)

elif mode == "Live Run" and 'current_report' in st.session_state:
    report_path_to_load = st.session_state['current_report']

# Only render dashboard if a report is selected/available
if report_path_to_load and os.path.exists(report_path_to_load):
    report = load_report(report_path_to_load)
    
    # Extract Data
    metadata = report.get('metadata', {})
    execution = metadata.get('execution', {})
    evaluation = metadata.get('evaluation', {})
    before = evaluation.get('before', {})
    after = evaluation.get('after', {})
    changes = report.get('changes_summary', [])
    decisions = report.get('decisions', [])

    # --- HEADER METRICS ---
    st.markdown("---")
    st.subheader(f"📊 Results for: {os.path.basename(report_path_to_load)}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows Processed", f"{before.get('rows', 0):,}")
        
    with col2:
        duration = execution.get('duration_seconds', 0)
        st.metric("Execution Time", f"{duration:.2f} s")

    with col3:
        speed = execution.get('rows_per_second', 0)
        st.metric("Processing Speed", f"{speed:.0f} rows/s")

    with col4:
        mem_before = before.get('memory_usage_kb', 0) / 1024
        mem_after = after.get('memory_usage_kb', 0) / 1024
        st.metric("Memory Usage (MB)", f"{mem_after:.2f} MB", delta=f"{mem_after - mem_before:.2f} MB", delta_color="inverse")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stress Test & Scalability", "🔄 What Changed?", "🧠 Intelligent Decisions", "📂 Data Preview"])

    # --- TAB 1: STRESS TEST ---
    with tab1:
        st.subheader("System Capabilities")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.info(f"**Dataset Size:** {before.get('rows', 0)} rows x {before.get('columns', 0)} columns")
            st.info(f"**Processed Size:** {after.get('rows', 0)} rows x {after.get('columns', 0)} columns")
            
            # Simple Chart for Column Explosion (One-Hot Encoding impact)
            fig_cols = go.Figure(data=[
                go.Bar(name='Before', x=['Columns'], y=[before.get('columns', 0)]),
                go.Bar(name='After', x=['Columns'], y=[after.get('columns', 0)])
            ])
            fig_cols.update_layout(title_text='Dimensionality Change (Column Explosion)')
            st.plotly_chart(fig_cols, use_container_width=True)

        with col_b:
            # Time vs Data Size (Theoretical based on current run)
            st.markdown("### ⚡ Performance Analysis")
            if duration > 0:
                st.success(f"The pipeline processed **{speed:.0f} rows per second**.")
                st.write(f"Estimated time for 100k rows: **{100000/speed:.1f} seconds**")
                st.write(f"Estimated time for 1M rows: **{1000000/speed/60:.1f} minutes**")
            else:
                st.warning("Duration too short to measure speed accurately.")

    # --- TAB 2: WHAT CHANGED ---
    with tab2:
        st.subheader("🔍 Detailed Transformation Log")
        
        if changes:
            df_changes = pd.DataFrame(changes)
            
            # Filters
            categories = st.multiselect("Filter by Category", df_changes['category'].unique(), default=df_changes['category'].unique())
            if categories:
                df_changes = df_changes[df_changes['category'].isin(categories)]
                
            st.dataframe(df_changes, use_container_width=True)
            
            # Visual breakdown of changes
            st.subheader("Impact by Category")
            change_counts = df_changes.groupby('category')['affected_rows'].sum().reset_index()
            fig_changes = px.bar(change_counts, x='category', y='affected_rows', title="Rows Affected by Transformation Type")
            st.plotly_chart(fig_changes, use_container_width=True)
        else:
            st.info("No detailed change logs available in this report.")

    # --- TAB 3: DECISIONS ---
    with tab3:
        st.subheader("🧠 Why did the system do that?")
        
        st.markdown("The **Meta-Learner** and **Heuristics** made the following decisions based on your data properties:")
        
        # Filter for interesting decisions
        important_decisions = [d for d in decisions if d['decision'] not in ['Success', 'Skipped']]
        
        for log in important_decisions:
            with st.expander(f"{log['step']}: {log['decision']}"):
                st.write(f"**Reason:** {log['reason']}")
                if log.get('details'):
                    st.json(log['details'])

    # --- TAB 4: DATA PREVIEW ---
    with tab4:
        st.subheader("📂 Output Data Preview")
        
        # Try to find the processed CSV file associated with this report
        # Assumption: Report name is report_processed_FILENAME.json -> processed_FILENAME.csv
        report_filename = os.path.basename(report_path_to_load)
        data_filename = report_filename.replace('report_', '').replace('.json', '.csv')
        data_path = os.path.join(os.path.dirname(report_path_to_load), data_filename)
        
        if os.path.exists(data_path):
            st.write(f"Loading: `{data_filename}`")
            # Load only first 100 rows for performance
            df_processed = pd.read_csv(data_path, nrows=100)
            st.dataframe(df_processed)
            
            with open(data_path, "rb") as file:
                st.download_button(
                    label="Download Processed Data",
                    data=file,
                    file_name=data_filename,
                    mime="text/csv"
                )
        else:
            st.warning(f"Could not find processed data file at: {data_path}")

elif mode == "Live Run" and 'current_report' not in st.session_state:
    st.info("⬆️ Upload a CSV file above to start!")


