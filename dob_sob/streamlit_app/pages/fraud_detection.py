"""
🔍 Advanced Fraud Detection Dashboard

Comprehensive fraud detection interface leveraging our 6-entity correlation engine
for real-time fraud scoring, network analysis, and investigation workflows.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional

# Import our fraud detection components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fraud_detection.algorithms.correlation import (
    CrossEntityCorrelationEngine, 
    EntityType, 
    FraudRiskLevel
)

# Page configuration
st.set_page_config(
    page_title="🔍 Fraud Detection Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .risk-critical { border-left: 5px solid #dc3545; }
    .risk-high { border-left: 5px solid #fd7e14; }
    .risk-medium { border-left: 5px solid #ffc107; }
    .risk-low { border-left: 5px solid #20c997; }
    .risk-minimal { border-left: 5px solid #6c757d; }
    
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #c0392b;
    }
    
    .network-stats {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_correlation_engine():
    """Initialize the correlation engine."""
    if 'correlation_engine' not in st.session_state:
        st.session_state.correlation_engine = CrossEntityCorrelationEngine()
    return st.session_state.correlation_engine

def generate_sample_fraud_data():
    """Generate sample fraud data for demonstration."""
    np.random.seed(42)
    
    # Sample data for each entity type
    sample_data = {
        EntityType.PERSON_PROFESSIONAL: [
            {
                'id': 'prof_001',
                'license_filing_count': 12500,  # High concentration
                'license_type': 'General Contractor',
                'license_types_count': 7,
                'active_boroughs': 5
            },
            {
                'id': 'prof_002', 
                'license_filing_count': 8900,
                'license_type': 'Electrician',
                'license_types_count': 3,
                'active_boroughs': 3
            },
            {
                'id': 'prof_003',
                'license_filing_count': 150,
                'license_type': 'Plumber',
                'license_types_count': 1,
                'active_boroughs': 1
            }
        ],
        EntityType.PROPERTY: [
            {
                'id': 'prop_001',
                'block_filing_count': 2800,  # High clustering
                'ownership_changes_count': 15,
                'violations_count': 45,
                'assessed_value': 500000,
                'market_value': 2000000
            },
            {
                'id': 'prop_002',
                'block_filing_count': 1200,
                'ownership_changes_count': 8,
                'violations_count': 25,
                'assessed_value': 800000,
                'market_value': 1200000
            }
        ],
        EntityType.JOB_PROJECT: [
            {
                'id': 'job_001',
                'filing_attempts_count': 8,  # Serial re-filing
                'status_changes': ['Objections', 'Filing Withdrawn'],
                'applicant_changes_count': 4,
                'amendments_count': 15,
                'work_types': ['Electrical', 'Plumbing', 'HVAC', 'Structural', 'Roofing', 'Windows']
            },
            {
                'id': 'job_002',
                'filing_attempts_count': 3,
                'status_changes': ['Pending', 'Approved'],
                'applicant_changes_count': 1,
                'amendments_count': 2,
                'work_types': ['Electrical']
            }
        ],
        EntityType.VIOLATION_ENFORCEMENT: [
            {
                'id': 'viol_001',
                'total_violations': 100,
                'dismissed_violations': 85,  # 85% dismissal rate
                'work_without_permit_count': 25,
                'outstanding_penalties': 75000,
                'violation_types_count': 18
            },
            {
                'id': 'viol_002',
                'total_violations': 20,
                'dismissed_violations': 5,
                'work_without_permit_count': 2,
                'outstanding_penalties': 5000,
                'violation_types_count': 4
            }
        ],
        EntityType.REGULATORY_INSPECTION: [
            {
                'id': 'insp_001',
                'required_inspections': 50,
                'completed_inspections': 12,  # 24% completion rate
                'failed_inspections': 25,
                'overdue_inspection_days': 220,
                'inspector_changes': 8
            }
        ],
        EntityType.FINANCIAL_COMPLIANCE: [
            {
                'id': 'fin_001',
                'total_penalties_imposed': 100000,
                'amount_paid': 15000,  # 15% payment rate
                'declared_work_value': 0,
                'permit_type': 'major',
                'entity_age_days': 60,
                'collection_attempts': 15
            }
        ]
    }
    
    return sample_data

def create_fraud_score_gauge(score: float, title: str):
    """Create a gauge chart for fraud scores."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 40], 'color': "yellow"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 90], 'color': "red"},
                {'range': [90, 100], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_network_visualization(fraud_networks: List[Dict[str, Any]]):
    """Create network visualization of fraud connections."""
    if not fraud_networks:
        return None
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes and edges from the first (highest risk) network
    network = fraud_networks[0]
    
    # Add nodes
    for node in network.get('nodes', []):
        entity_id = node.get('entity_id', '')
        entity_type = node.get('entity_type', '')
        fraud_score = node.get('fraud_score', 0)
        
        G.add_node(entity_id, 
                  entity_type=entity_type,
                  fraud_score=fraud_score,
                  size=max(10, fraud_score/5))
    
    # Add edges
    for edge in network.get('edges', []):
        G.add_edge(edge.get('entity_1', ''), edge.get('entity_2', ''),
                  weight=edge.get('correlation_strength', 0))
    
    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Extract node and edge information
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_info = G.nodes[node]
        entity_type = node_info.get('entity_type', 'unknown')
        fraud_score = node_info.get('fraud_score', 0)
        
        node_text.append(f"{node}<br>Type: {entity_type}<br>Score: {fraud_score:.1f}")
        node_color.append(fraud_score)
        node_size.append(max(20, fraud_score/3))
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node.split('_')[0] for node in G.nodes()],
        textposition="middle center",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Fraud Score"),
            line=dict(width=2, color='black')
        )
    ))
    
    fig.update_layout(
        title="🕸️ Fraud Network Visualization",
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Node size and color represent fraud risk scores",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    return fig

def display_fraud_alerts(analysis_results: Dict[str, Any]):
    """Display fraud alerts and warnings."""
    critical_networks = analysis_results.get('critical_networks', 0)
    high_risk_entities = analysis_results.get('high_risk_entities', 0)
    
    if critical_networks > 0 or high_risk_entities > 10:
        st.markdown(f"""
        <div class="fraud-alert">
            <h3>🚨 CRITICAL FRAUD ALERTS</h3>
            <p><strong>{critical_networks}</strong> critical fraud networks detected</p>
            <p><strong>{high_risk_entities}</strong> high-risk entities identified</p>
            <p>Immediate investigation recommended</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main fraud detection dashboard."""
    st.title("🔍 Advanced Fraud Detection Engine")
    st.markdown("**Real-time cross-entity fraud analysis powered by our 6-category correlation engine**")
    
    # Initialize correlation engine
    engine = initialize_correlation_engine()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    correlation_threshold = st.sidebar.slider(
        "Correlation Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        help="Minimum correlation strength to detect relationships"
    )
    
    min_network_size = st.sidebar.slider(
        "Minimum Network Size", 
        min_value=2, 
        max_value=10, 
        value=3,
        help="Minimum number of entities to form a fraud network"
    )
    
    # Analysis trigger
    if st.sidebar.button("🔍 Run Fraud Analysis", type="primary"):
        with st.spinner("🔄 Analyzing fraud patterns across all entity categories..."):
            # Generate sample data
            sample_data = generate_sample_fraud_data()
            
            # Run analysis
            try:
                analysis_results = asyncio.run(
                    engine.analyze_fraud_patterns(
                        sample_data,
                        correlation_threshold=correlation_threshold,
                        min_network_size=min_network_size
                    )
                )
                
                st.session_state.analysis_results = analysis_results
                st.success("✅ Fraud analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return
    
    # Display results if available
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        # Display fraud alerts
        display_fraud_alerts(results)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Total Entities</h3>
                <h2>{}</h2>
                <p>Analyzed across 6 categories</p>
            </div>
            """.format(results.get('total_entities_analyzed', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>⚠️ High Risk</h3>
                <h2>{}</h2>
                <p>Entities requiring attention</p>
            </div>
            """.format(results.get('high_risk_entities', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🔗 Correlations</h3>
                <h2>{}</h2>
                <p>Cross-entity relationships</p>
            </div>
            """.format(results.get('total_correlations', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🕸️ Networks</h3>
                <h2>{}</h2>
                <p>Fraud networks detected</p>
            </div>
            """.format(results.get('fraud_networks_detected', 0)), unsafe_allow_html=True)
        
        # Entity breakdown
        st.subheader("📈 Entity Category Breakdown")
        
        entity_breakdown = results.get('entity_breakdown', {})
        if entity_breakdown:
            breakdown_df = pd.DataFrame([
                {'Category': k.replace('_', ' ').title(), 'Count': v} 
                for k, v in entity_breakdown.items()
            ])
            
            fig = px.bar(
                breakdown_df, 
                x='Category', 
                y='Count',
                title="Entities Analyzed by Category",
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top fraud scores
        st.subheader("🎯 Highest Risk Entities")
        
        entity_scores = results.get('entity_scores', [])
        if entity_scores:
            # Create fraud score gauges for top entities
            top_entities = entity_scores[:3]
            
            cols = st.columns(len(top_entities))
            for i, entity in enumerate(top_entities):
                with cols[i]:
                    fig = create_fraud_score_gauge(
                        entity['fraud_score'], 
                        f"{entity['entity_id']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk factors
                    risk_class = f"risk-{entity.get('risk_level', 'minimal')}"
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h4>Risk Factors:</h4>
                        <ul>
                    """, unsafe_allow_html=True)
                    
                    for factor in entity.get('risk_factors', [])[:3]:
                        st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Detailed entity table
            st.subheader("📋 Detailed Entity Analysis")
            
            entity_df = pd.DataFrame(entity_scores)
            entity_df = entity_df.rename(columns={
                'entity_id': 'Entity ID',
                'entity_type': 'Category',
                'fraud_score': 'Fraud Score',
                'risk_level': 'Risk Level'
            })
            
            # Color-code by risk level
            def color_risk_level(val):
                colors = {
                    'critical': 'background-color: #dc3545; color: white',
                    'high': 'background-color: #fd7e14; color: white',
                    'medium': 'background-color: #ffc107; color: black',
                    'low': 'background-color: #20c997; color: white',
                    'minimal': 'background-color: #6c757d; color: white'
                }
                return colors.get(val, '')
            
            styled_df = entity_df.style.applymap(
                color_risk_level, 
                subset=['Risk Level']
            ).format({'Fraud Score': '{:.1f}'})
            
            st.dataframe(styled_df, use_container_width=True)
        
        # Network visualization
        fraud_networks = results.get('fraud_networks', [])
        if fraud_networks:
            st.subheader("🕸️ Fraud Network Analysis")
            
            # Network statistics
            st.markdown("""
            <div class="network-stats">
                <h4>🔍 Network Intelligence</h4>
                <p>Sophisticated fraud networks detected using graph analysis and correlation patterns</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Network visualization
            network_fig = create_network_visualization(fraud_networks)
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)
            
            # Network details
            for i, network in enumerate(fraud_networks[:3]):
                with st.expander(f"🕸️ Network {i+1}: {network.get('network_type', 'Unknown')} (Risk: {network.get('risk_level', 'Unknown')})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Risk Score", f"{network.get('risk_score', 0):.1f}")
                        st.metric("Entity Count", network.get('entity_count', 0))
                        st.metric("Connections", network.get('connection_count', 0))
                    
                    with col2:
                        evidence = network.get('evidence_summary', {})
                        st.write("**Entity Types:**")
                        for entity_type in evidence.get('entity_types', []):
                            st.write(f"• {entity_type.replace('_', ' ').title()}")
                        
                        st.write("**Risk Factors:**")
                        for factor in evidence.get('risk_factors', [])[:5]:
                            st.write(f"• {factor}")
        
        # Correlation analysis
        top_correlations = results.get('top_correlations', [])
        if top_correlations:
            st.subheader("🔗 Cross-Entity Correlations")
            
            corr_df = pd.DataFrame(top_correlations)
            corr_df = corr_df.rename(columns={
                'entity_1': 'Entity 1',
                'entity_2': 'Entity 2',
                'entity_1_type': 'Type 1',
                'entity_2_type': 'Type 2',
                'correlation_strength': 'Strength',
                'correlation_type': 'Pattern Type'
            })
            
            # Format correlation strength as percentage
            corr_df['Strength'] = (corr_df['Strength'] * 100).round(1).astype(str) + '%'
            
            st.dataframe(corr_df, use_container_width=True)
    
    else:
        # Welcome message
        st.info("👆 Click 'Run Fraud Analysis' in the sidebar to begin comprehensive fraud detection analysis")
        
        # Feature overview
        st.subheader("🎯 Fraud Detection Capabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Entity Analysis:**
            - Person/Professional fraud patterns
            - Property manipulation detection
            - Job/Project abuse identification
            - Violation/Enforcement evasion
            - Regulatory/Inspection avoidance
            - Financial/Compliance violations
            """)
        
        with col2:
            st.markdown("""
            **🕸️ Network Detection:**
            - Cross-entity correlation analysis
            - Fraud network visualization
            - Risk scoring algorithms
            - Real-time pattern recognition
            - Investigation workflow support
            - Evidence aggregation
            """)
        
        # Sample fraud patterns
        st.subheader("📊 Detected Fraud Patterns")
        
        patterns_data = {
            'Pattern Type': [
                'License Concentration Abuse',
                'Geographic Clustering',
                'Serial Re-filing',
                'Violation Dismissal Abuse',
                'Inspection Avoidance',
                'Payment Evasion'
            ],
            'Risk Level': ['Critical', 'High', 'High', 'Critical', 'Medium', 'High'],
            'Frequency': [15, 23, 18, 31, 12, 27],
            'Entity Categories': [
                'Professional',
                'Property',
                'Job/Project',
                'Violation',
                'Inspection',
                'Financial'
            ]
        }
        
        patterns_df = pd.DataFrame(patterns_data)
        
        fig = px.scatter(
            patterns_df,
            x='Frequency',
            y='Pattern Type',
            color='Risk Level',
            size='Frequency',
            color_discrete_map={
                'Critical': '#dc3545',
                'High': '#fd7e14',
                'Medium': '#ffc107'
            },
            title="Fraud Pattern Detection Overview"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 