#!/usr/bin/env python3
"""
🗽 NYC DOB Data Explorer Dashboard
Interactive Streamlit dashboard for exploring raw CSV datasets from NYC Department of Buildings
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🗽 NYC DOB Data Explorer",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = Path("data/raw")
METADATA_DIR = Path("data/metadata")

@st.cache_data
def load_available_datasets():
    """Load all available datasets and their metadata"""
    datasets = {}
    
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir() and list(dataset_dir.glob("*.csv")):
            # Get latest CSV file
            csv_files = list(dataset_dir.glob("*.csv"))
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            
            # Load metadata if available
            metadata_file = METADATA_DIR / f"{dataset_dir.name}_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            datasets[dataset_dir.name] = {
                'csv_path': latest_csv,
                'metadata': metadata,
                'size_mb': round(latest_csv.stat().st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(latest_csv.stat().st_mtime)
            }
    
    return datasets

@st.cache_data
def load_dataset_sample(csv_path, sample_size=10000):
    """Load a sample of the dataset for quick exploration"""
    try:
        # Try to load a sample first
        df = pd.read_csv(csv_path, nrows=sample_size, low_memory=False)
        return df, False  # False means not full dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, False

@st.cache_data  
def load_full_dataset(csv_path):
    """Load the complete dataset"""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        return df, True  # True means full dataset
    except Exception as e:
        st.error(f"Error loading full dataset: {e}")
        return None, False

def format_dataset_name(name):
    """Convert dataset name to human-readable format"""
    return name.replace('_', ' ').title()

def main():
    st.title("🗽 NYC DOB Data Explorer")
    st.markdown("**Interactive dashboard for exploring 25GB of NYC Department of Buildings data**")
    
    # Load available datasets
    datasets = load_available_datasets()
    
    if not datasets:
        st.error("No datasets found in data/raw/ directory")
        return
    
    # Sidebar - Dataset Selection
    st.sidebar.header("📊 Dataset Selection")
    
    # Sort datasets by size (largest first) for interesting exploration
    sorted_datasets = sorted(datasets.items(), key=lambda x: x[1]['size_mb'], reverse=True)
    dataset_options = [f"{format_dataset_name(name)} ({info['size_mb']} MB)" for name, info in sorted_datasets]
    
    selected_display = st.sidebar.selectbox(
        "Choose dataset:",
        dataset_options,
        help="Datasets sorted by size (largest first)"
    )
    
    # Extract actual dataset name
    selected_dataset = sorted_datasets[dataset_options.index(selected_display)][0]
    dataset_info = datasets[selected_dataset]
    
    # Dataset overview
    st.sidebar.markdown("### 📋 Dataset Info")
    st.sidebar.write(f"**Size:** {dataset_info['size_mb']} MB")
    st.sidebar.write(f"**Updated:** {dataset_info['modified'].strftime('%Y-%m-%d %H:%M')}")
    
    # Load options
    st.sidebar.markdown("### ⚙️ Load Options")
    load_full = st.sidebar.checkbox(
        "Load full dataset", 
        value=False,
        help="⚠️ May be slow for large datasets"
    )
    
    sample_size = st.sidebar.slider(
        "Sample size",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        disabled=load_full,
        help="Number of rows to load for quick exploration"
    )
    
    # Load the dataset
    with st.spinner(f"Loading {'full dataset' if load_full else f'{sample_size:,} rows'}..."):
        if load_full:
            df, is_full = load_full_dataset(dataset_info['csv_path'])
        else:
            df, is_full = load_dataset_sample(dataset_info['csv_path'], sample_size)
    
    if df is None:
        return
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        if not is_full:
            st.metric("Sample", f"{len(df):,} of full dataset")
        else:
            st.metric("Status", "Full dataset loaded")
    
    # Warning for sample data
    if not is_full:
        st.warning(f"⚠️ Showing sample of {len(df):,} rows. Enable 'Load full dataset' for complete analysis.")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Overview", "🔍 Data Browser", "📊 Analytics", "🔗 Relationships", "📝 Metadata", "🚨 Fraud Detection"])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Column information
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            col_info.append({
                'Column': col,
                'Data Type': dtype,
                'Null Count': null_count,
                'Null %': f"{null_pct:.1f}%",
                'Unique Values': unique_count,
                'Sample Values': ', '.join(str(x) for x in df[col].dropna().head(3))
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True)
        
        # Data quality summary
        st.subheader("Data Quality Summary")
        quality_col1, quality_col2 = st.columns(2)
        
        with quality_col1:
            # Null values chart
            null_data = df.isnull().sum().sort_values(ascending=False)
            if null_data.sum() > 0:
                fig_null = px.bar(
                    x=null_data.values,
                    y=null_data.index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Missing Count', 'y': 'Column'}
                )
                fig_null.update_layout(height=400)
                st.plotly_chart(fig_null, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with quality_col2:
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            # Convert dtype names to strings for JSON serialization
            dtype_names = [str(dtype) for dtype in dtype_counts.index]
            fig_dtype = px.pie(
                values=dtype_counts.values,
                names=dtype_names,
                title="Data Types Distribution"
            )
            st.plotly_chart(fig_dtype, use_container_width=True)
    
    with tab2:
        st.header("Data Browser")
        
        # Search and filter
        search_col, filter_col = st.columns(2)
        
        with search_col:
            search_term = st.text_input("🔍 Search across all columns:", help="Case-insensitive search")
        
        with filter_col:
            selected_columns = st.multiselect(
                "📋 Select columns to display:",
                df.columns.tolist(),
                default=df.columns[:10].tolist(),
                help="Choose which columns to show"
            )
        
        # Apply search filter
        display_df = df[selected_columns] if selected_columns else df
        
        if search_term:
            # Search across all text columns
            text_columns = display_df.select_dtypes(include=['object']).columns
            mask = False
            for col in text_columns:
                mask |= display_df[col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = display_df[mask]
            st.info(f"Found {len(display_df):,} rows matching '{search_term}'")
        
        # Display data
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download filtered data
        if len(display_df) > 0:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="💾 Download filtered data as CSV",
                data=csv,
                file_name=f"{selected_dataset}_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("Analytics & Visualizations")
        
        # Numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            st.subheader("📈 Numeric Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                selected_numeric = st.selectbox("Choose numeric column:", numeric_cols)
            
            with analysis_col2:
                chart_type = st.selectbox("Chart type:", ["Histogram", "Box Plot", "Time Series"])
            
            if selected_numeric:
                if chart_type == "Histogram":
                    fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                elif chart_type == "Box Plot":
                    fig = px.box(df, y=selected_numeric, title=f"Box Plot of {selected_numeric}")
                elif chart_type == "Time Series":
                    # Try to find date columns for time series
                    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if date_cols:
                        date_col = st.selectbox("Choose date column:", date_cols)
                        try:
                            df_time = df.copy()
                            df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
                            df_time = df_time.dropna(subset=[date_col])
                            if len(df_time) > 0:
                                fig = px.line(df_time, x=date_col, y=selected_numeric, 
                                            title=f"{selected_numeric} over time")
                            else:
                                st.warning("No valid dates found for time series")
                                fig = None
                        except:
                            st.warning("Could not create time series - invalid date format")
                            fig = None
                    else:
                        st.warning("No date columns found for time series")
                        fig = None
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        if text_cols:
            st.subheader("📝 Categorical Analysis")
            
            cat_col1, cat_col2 = st.columns(2)
            
            with cat_col1:
                selected_categorical = st.selectbox("Choose categorical column:", text_cols)
            
            with cat_col2:
                top_n = st.slider("Show top N values:", 5, 50, 20)
            
            if selected_categorical:
                value_counts = df[selected_categorical].value_counts().head(top_n)
                
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f"Top {top_n} values in {selected_categorical}",
                    labels={'x': 'Count', 'y': selected_categorical}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔗 Data Relationships")
        
        if len(numeric_cols) >= 2:
            st.subheader("Correlation Analysis")
            
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Scatter plot
            st.subheader("Scatter Plot Analysis")
            scatter_col1, scatter_col2 = st.columns(2)
            
            with scatter_col1:
                x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
            
            with scatter_col2:
                y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
            
            if x_col != y_col:
                # Option to color by categorical column
                color_by = st.selectbox("Color by (optional):", ["None"] + text_cols)
                
                if color_by == "None":
                    fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                else:
                    # Limit categories for better visualization
                    df_scatter = df.copy()
                    top_categories = df[color_by].value_counts().head(10).index
                    df_scatter.loc[~df_scatter[color_by].isin(top_categories), color_by] = 'Other'
                    
                    fig_scatter = px.scatter(
                        df_scatter, x=x_col, y=y_col, color=color_by,
                        title=f"{x_col} vs {y_col} (colored by {color_by})"
                    )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for relationship analysis")
    
    with tab5:
        st.header("📝 Dataset Metadata")
        
        if dataset_info['metadata']:
            metadata = dataset_info['metadata']
            
            # Display key metadata fields
            if 'description' in metadata:
                st.subheader("Dataset Description")
                st.write(metadata['description'])
            
            if 'columns' in metadata:
                st.subheader("Column Definitions")
                
                # Create a nice table of column metadata
                col_metadata = []
                for col_info in metadata['columns']:
                    col_metadata.append({
                        'Field Name': col_info.get('fieldName', 'N/A'),
                        'Data Type': col_info.get('dataTypeName', 'N/A'),
                        'Description': col_info.get('description', 'No description available')
                    })
                
                if col_metadata:
                    metadata_df = pd.DataFrame(col_metadata)
                    st.dataframe(metadata_df, use_container_width=True)
            
            # Raw metadata
            with st.expander("🔧 Raw Metadata (JSON)"):
                st.json(metadata)
        else:
            st.info("No metadata available for this dataset")
    
    with tab6:
        st.header("🚨 Fraud Detection Analysis")
        st.info("🔧 **Cross-Entity Correlation Engine** - Advanced fraud detection across 6 entity categories")
        
        # Quick fraud indicators for current dataset
        st.subheader("Quick Fraud Indicators")
        
        # Look for common fraud indicator columns
        fraud_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in ['violation', 'fine', 'penalty', 'suspend', 'revok', 'complaint', 'emergency']):
                fraud_columns.append(col)
        
        if fraud_columns:
            st.write("**Potential fraud indicator columns found:**")
            for col in fraud_columns[:5]:  # Show first 5
                unique_vals = df[col].nunique()
                st.write(f"• `{col}` ({unique_vals:,} unique values)")
            
            # Quick analysis of first fraud column
            if len(fraud_columns) > 0:
                analysis_col = fraud_columns[0]
                st.subheader(f"Analysis: {analysis_col}")
                
                if df[analysis_col].dtype == 'object':
                    value_counts = df[analysis_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.values, y=value_counts.index, orientation='h',
                               title=f"Top values in {analysis_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(df, x=analysis_col, title=f"Distribution of {analysis_col}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No obvious fraud indicator columns detected in this dataset.")
        
        # Link to dedicated fraud detection pages
        st.markdown("---")
        st.markdown("### 🔗 Advanced Fraud Detection")
        st.markdown("For comprehensive fraud analysis, use the dedicated fraud detection modules:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 [Fraud Detection Dashboard](fraud_detection)** - Comprehensive analysis across all entity types")
        with col2:
            st.markdown("**⚡ [Real-time Monitoring](real_time_monitoring)** - Live fraud detection and alerts")
    
    # Footer
    st.markdown("---")
    st.markdown("*🗽 NYC DOB Data Explorer - Built with Streamlit*")

if __name__ == "__main__":
    main()