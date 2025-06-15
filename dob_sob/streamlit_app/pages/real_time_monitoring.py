"""
🚨 Real-Time Fraud Monitoring Dashboard

Live fraud detection monitoring with real-time alerts, network visualization,
and investigation workflows powered by the Cross-Entity Correlation Engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import threading

# Page configuration
st.set_page_config(
    page_title="🚨 Real-Time Fraud Monitoring",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for real-time monitoring
st.markdown("""
<style>
    .alert-critical {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #721c24;
        animation: pulse 2s infinite;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #fd7e14 0%, #e55100 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #bf360c;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ffc107 0%, #ff8f00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #e65100;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .monitoring-status {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .network-alert {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #495057;
    }
    
    .metric-live {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .investigation-panel {
        background: linear-gradient(135deg, #343a40 0%, #495057 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust as needed

def check_api_connection():
    """Check if the fraud monitoring API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_monitoring_status():
    """Get current monitoring system status."""
    try:
        response = requests.get(f"{API_BASE_URL}/monitoring/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_active_alerts(severity=None, entity_type=None, limit=50):
    """Get active fraud alerts."""
    try:
        params = {"limit": limit}
        if severity:
            params["severity"] = severity
        if entity_type:
            params["entity_type"] = entity_type
            
        response = requests.get(f"{API_BASE_URL}/alerts", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def dismiss_alert(alert_id):
    """Dismiss a fraud alert."""
    try:
        response = requests.delete(f"{API_BASE_URL}/alerts/{alert_id}", timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_entities(entities_data, correlation_threshold=0.3, min_network_size=3):
    """Submit entities for fraud analysis."""
    try:
        payload = {
            "entities": entities_data,
            "correlation_threshold": correlation_threshold,
            "min_network_size": min_network_size,
            "include_networks": True,
            "include_correlations": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_analysis_history(limit=20):
    """Get analysis history."""
    try:
        response = requests.get(f"{API_BASE_URL}/monitoring/history", params={"limit": limit}, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_real_time_metrics_display(status_data):
    """Create real-time metrics display."""
    if not status_data:
        st.error("❌ Unable to connect to monitoring API")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-live">
            <h3>🟢 System Status</h3>
            <h2>{status_data.get('status', 'Unknown').upper()}</h2>
            <p>Uptime: {status_data.get('uptime_seconds', 0)//3600}h {(status_data.get('uptime_seconds', 0)%3600)//60}m</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-live">
            <h3>📊 Entities Monitored</h3>
            <h2>{status_data.get('entities_monitored', 0):,}</h2>
            <p>Total analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-live">
            <h3>🚨 Alerts Generated</h3>
            <h2>{status_data.get('alerts_generated', 0):,}</h2>
            <p>All time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        last_analysis = status_data.get('last_analysis')
        if last_analysis:
            last_time = datetime.fromisoformat(last_analysis.replace('Z', '+00:00'))
            time_ago = datetime.now() - last_time.replace(tzinfo=None)
            time_str = f"{time_ago.seconds//60}m ago"
        else:
            time_str = "Never"
            
        st.markdown(f"""
        <div class="metric-live">
            <h3>⏱️ Last Analysis</h3>
            <h2>{time_str}</h2>
            <p>Most recent</p>
        </div>
        """, unsafe_allow_html=True)

def create_alerts_dashboard(alerts_data):
    """Create alerts dashboard."""
    if not alerts_data:
        st.info("✅ No active alerts")
        return
    
    # Group alerts by severity
    critical_alerts = [a for a in alerts_data if a.get('severity') == 'critical']
    high_alerts = [a for a in alerts_data if a.get('severity') == 'high']
    medium_alerts = [a for a in alerts_data if a.get('severity') == 'medium']
    
    # Display critical alerts first
    if critical_alerts:
        st.markdown("### 🔴 CRITICAL ALERTS")
        for alert in critical_alerts:
            display_alert(alert, 'critical')
    
    if high_alerts:
        st.markdown("### 🟠 HIGH PRIORITY ALERTS")
        for alert in high_alerts:
            display_alert(alert, 'high')
    
    if medium_alerts:
        st.markdown("### 🟡 MEDIUM PRIORITY ALERTS")
        for alert in medium_alerts:
            display_alert(alert, 'medium')

def display_alert(alert, severity):
    """Display individual alert."""
    alert_class = f"alert-{severity}"
    
    # Format timestamp
    timestamp = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
    time_ago = datetime.now() - timestamp.replace(tzinfo=None)
    time_str = f"{time_ago.seconds//60}m ago"
    
    # Risk factors as bullet points
    risk_factors_html = "<br>".join([f"• {factor}" for factor in alert.get('risk_factors', [])])
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown(f"""
        <div class="{alert_class}">
            <h4>🚨 {alert['alert_type'].replace('_', ' ').title()}</h4>
            <p><strong>Entity:</strong> {alert['entity_id']} ({alert['entity_type']})</p>
            <p><strong>Fraud Score:</strong> {alert['fraud_score']:.1f}/100</p>
            <p><strong>Time:</strong> {time_str}</p>
            <p><strong>Risk Factors:</strong></p>
            <p>{risk_factors_html}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button(f"Dismiss", key=f"dismiss_{alert['alert_id']}"):
            if dismiss_alert(alert['alert_id']):
                st.success("Alert dismissed")
                st.experimental_rerun()
            else:
                st.error("Failed to dismiss alert")
        
        if st.button(f"Investigate", key=f"investigate_{alert['alert_id']}"):
            st.session_state.investigation_entity = alert['entity_id']
            st.session_state.investigation_type = alert['entity_type']
            st.experimental_rerun()

def create_live_analysis_interface():
    """Create interface for live fraud analysis."""
    st.markdown("### 🔍 Live Fraud Analysis")
    
    with st.expander("Submit Entities for Analysis", expanded=False):
        # Entity input form
        entity_type = st.selectbox(
            "Entity Type",
            ["PERSON_PROFESSIONAL", "PROPERTY", "JOB_PROJECT", 
             "VIOLATION_ENFORCEMENT", "REGULATORY_INSPECTION", "FINANCIAL_COMPLIANCE"]
        )
        
        entity_id = st.text_input("Entity ID", placeholder="e.g., prof_001")
        
        # Dynamic form based on entity type
        entity_data = {}
        
        if entity_type == "PERSON_PROFESSIONAL":
            col1, col2 = st.columns(2)
            with col1:
                entity_data['license_filing_count'] = st.number_input("License Filing Count", min_value=0, value=100)
                entity_data['license_types_count'] = st.number_input("License Types Count", min_value=1, value=1)
            with col2:
                entity_data['active_boroughs'] = st.number_input("Active Boroughs", min_value=1, max_value=5, value=1)
                entity_data['license_type'] = st.text_input("License Type", value="General Contractor")
        
        elif entity_type == "PROPERTY":
            col1, col2 = st.columns(2)
            with col1:
                entity_data['block_filing_count'] = st.number_input("Block Filing Count", min_value=0, value=50)
                entity_data['ownership_changes_count'] = st.number_input("Ownership Changes", min_value=0, value=2)
            with col2:
                entity_data['violations_count'] = st.number_input("Violations Count", min_value=0, value=5)
                entity_data['assessed_value'] = st.number_input("Assessed Value", min_value=0, value=500000)
                entity_data['market_value'] = st.number_input("Market Value", min_value=0, value=600000)
        
        elif entity_type == "JOB_PROJECT":
            col1, col2 = st.columns(2)
            with col1:
                entity_data['filing_attempts_count'] = st.number_input("Filing Attempts", min_value=1, value=1)
                entity_data['applicant_changes_count'] = st.number_input("Applicant Changes", min_value=0, value=0)
            with col2:
                entity_data['amendments_count'] = st.number_input("Amendments Count", min_value=0, value=1)
                status_changes = st.text_area("Status Changes (comma-separated)", value="Pending,Approved")
                entity_data['status_changes'] = [s.strip() for s in status_changes.split(',')]
                work_types = st.text_area("Work Types (comma-separated)", value="Electrical")
                entity_data['work_types'] = [w.strip() for w in work_types.split(',')]
        
        # Analysis parameters
        st.markdown("**Analysis Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            correlation_threshold = st.slider("Correlation Threshold", 0.1, 1.0, 0.3, 0.1)
        with col2:
            min_network_size = st.slider("Min Network Size", 2, 10, 3)
        
        if st.button("🔍 Analyze Entity", type="primary"):
            if entity_id:
                with st.spinner("Analyzing entity for fraud patterns..."):
                    entities_data = [{
                        "id": entity_id,
                        "entity_type": entity_type,
                        "data": entity_data
                    }]
                    
                    results = analyze_entities(
                        entities_data,
                        correlation_threshold,
                        min_network_size
                    )
                    
                    if results:
                        st.success(f"✅ Analysis completed! Analysis ID: {results['analysis_id']}")
                        display_analysis_results(results)
                    else:
                        st.error("❌ Analysis failed. Please check API connection.")
            else:
                st.error("Please enter an Entity ID")

def display_analysis_results(results):
    """Display fraud analysis results."""
    st.markdown("### 📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entities Analyzed", results['total_entities_analyzed'])
    with col2:
        st.metric("High Risk Entities", results['high_risk_entities'])
    with col3:
        st.metric("Total Correlations", results['total_correlations'])
    with col4:
        st.metric("Fraud Networks", results['fraud_networks_detected'])
    
    # Entity scores
    if results.get('entity_scores'):
        st.markdown("#### Entity Fraud Scores")
        scores_df = pd.DataFrame(results['entity_scores'])
        
        # Create score visualization
        fig = px.bar(
            scores_df,
            x='entity_id',
            y='fraud_score',
            color='risk_level',
            title="Entity Fraud Scores",
            color_discrete_map={
                'CRITICAL': '#dc3545',
                'HIGH': '#fd7e14',
                'MEDIUM': '#ffc107',
                'LOW': '#20c997',
                'MINIMAL': '#6c757d'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scores table
        st.dataframe(scores_df, use_container_width=True)
    
    # Correlations
    if results.get('top_correlations'):
        st.markdown("#### Cross-Entity Correlations")
        correlations_df = pd.DataFrame(results['top_correlations'])
        st.dataframe(correlations_df, use_container_width=True)
    
    # Fraud networks
    if results.get('fraud_networks'):
        st.markdown("#### Detected Fraud Networks")
        networks_df = pd.DataFrame(results['fraud_networks'])
        st.dataframe(networks_df, use_container_width=True)

def create_investigation_panel():
    """Create investigation panel for detailed entity analysis."""
    if 'investigation_entity' not in st.session_state:
        return
    
    st.markdown("### 🔍 Investigation Panel")
    
    entity_id = st.session_state.investigation_entity
    entity_type = st.session_state.investigation_type
    
    st.markdown(f"""
    <div class="investigation-panel">
        <h4>🎯 Investigating: {entity_id}</h4>
        <p><strong>Type:</strong> {entity_type}</p>
        <p><strong>Status:</strong> Under Investigation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Investigation actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Generate Report"):
            st.info("Investigation report generation would be implemented here")
    
    with col2:
        if st.button("🔗 Find Connections"):
            st.info("Network analysis would be implemented here")
    
    with col3:
        if st.button("✅ Close Investigation"):
            del st.session_state.investigation_entity
            del st.session_state.investigation_type
            st.experimental_rerun()

def create_analysis_history_chart(history_data):
    """Create analysis history visualization."""
    if not history_data or not history_data.get('recent_analyses'):
        return
    
    analyses = history_data['recent_analyses']
    df = pd.DataFrame(analyses)
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Analysis Frequency', 'High-Risk Entities Found'),
            vertical_spacing=0.1
        )
        
        # Analysis frequency
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['entity_count'],
                mode='lines+markers',
                name='Entities Analyzed',
                line=dict(color='#17a2b8')
            ),
            row=1, col=1
        )
        
        # High-risk entities
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['high_risk_count'],
                mode='lines+markers',
                name='High-Risk Found',
                line=dict(color='#dc3545')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            title_text="Analysis History",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard function."""
    st.title("🚨 Real-Time Fraud Monitoring Dashboard")
    st.markdown("Live fraud detection monitoring with real-time alerts and investigation workflows")
    
    # Check API connection
    api_connected = check_api_connection()
    
    if not api_connected:
        st.error("❌ Cannot connect to Fraud Monitoring API. Please ensure the API is running on http://localhost:8000")
        st.info("To start the API, run: `uvicorn dob_sob.api.fraud_monitoring:app --reload`")
        return
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=True)
    
    if auto_refresh:
        # Auto-refresh every 30 seconds
        time.sleep(0.1)  # Small delay to prevent too frequent updates
        st.experimental_rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.experimental_rerun()
    
    # Get current status
    status_data = get_monitoring_status()
    
    # Real-time metrics
    create_real_time_metrics_display(status_data)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Active Alerts", "🔍 Live Analysis", "📊 History", "🎯 Investigation"])
    
    with tab1:
        st.markdown("## Active Fraud Alerts")
        
        # Alert filters
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.selectbox("Filter by Severity", ["All", "critical", "high", "medium"])
        with col2:
            entity_filter = st.selectbox("Filter by Entity Type", ["All", "PERSON_PROFESSIONAL", "PROPERTY", "JOB_PROJECT"])
        with col3:
            limit = st.number_input("Max Alerts", min_value=10, max_value=100, value=50)
        
        # Get and display alerts
        alerts = get_active_alerts(
            severity=None if severity_filter == "All" else severity_filter,
            entity_type=None if entity_filter == "All" else entity_filter,
            limit=limit
        )
        
        create_alerts_dashboard(alerts)
    
    with tab2:
        create_live_analysis_interface()
    
    with tab3:
        st.markdown("## Analysis History")
        history_data = get_analysis_history()
        
        if history_data:
            # Summary metrics
            summary = history_data.get('summary', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Analyses", history_data.get('total_analyses', 0))
            with col2:
                st.metric("Total Entities", summary.get('total_entities_analyzed', 0))
            with col3:
                st.metric("High-Risk Found", summary.get('total_high_risk_found', 0))
            
            # History chart
            create_analysis_history_chart(history_data)
            
            # Recent analyses table
            if history_data.get('recent_analyses'):
                st.markdown("### Recent Analyses")
                recent_df = pd.DataFrame(history_data['recent_analyses'])
                recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No analysis history available")
    
    with tab4:
        create_investigation_panel()
    
    # Footer with system info
    st.markdown("---")
    if status_data:
        st.markdown(f"""
        <div class="monitoring-status">
            <strong>System Health:</strong> {status_data.get('system_health', 'Unknown').upper()} | 
            <strong>API Version:</strong> 2.1.0 | 
            <strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 