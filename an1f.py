#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, f
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set up the app
st.set_page_config(page_title="ANOVA Analyzer", layout="wide")
st.title("One-Way ANOVA with Tukey HSD Test")
st.markdown("Upload your CSV file with groups as columns and measurements as rows")

# File upload section
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="Each column should represent a different group/condition"
)

# Sample data download
st.sidebar.markdown("### Need sample data?")
sample_data = """Control,TreatmentA,TreatmentB
12.5,15.3,18.2
13.2,14.7,17.5
11.8,16.1,19.0
10.9,15.8,18.6
12.1,14.2,17.9
11.5,15.6,18.3
12.8,14.9,17.7
13.0,16.4,18.9"""
st.sidebar.download_button(
    label="Download Sample CSV",
    data=sample_data,
    file_name="anova_sample_data.csv",
    mime="text/csv"
)

# Main analysis function
def run_analysis(df):
    # Convert to long format
    df_long = df.melt(var_name='Group', value_name='Value').dropna()
    
    # Calculate ANOVA
    grouped_data = [group[1]['Value'].values for group in df_long.groupby('Group')]
    anova_result = f_oneway(*grouped_data)
    
    if len(grouped_data) < 2:
        st.error("Need at least 2 groups for ANOVA")
        st.stop()
    
    if any(len(g) < 3 for g in grouped_data):
        st.warning("Some groups have very small sample sizes (<3)")

    # Create ANOVA table
    groups = df_long.groupby('Group')['Value']
    n_groups = len(groups)
    n_total = len(df_long)
    grand_mean = df_long['Value'].mean()
    
    # Between-groups calculations
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
    df_between = n_groups - 1
    ms_between = ss_between / df_between
    
    # Within-groups calculations
    ss_within = sum(sum((x - g.mean())**2 for x in g) for _, g in groups)
    df_within = n_total - n_groups
    ms_within = ss_within / df_within
    
    # F-statistic
    f_value = ms_between / ms_within
    p_value = anova_result.pvalue
    f_crit = f.ppf(0.95, df_between, df_within)
    
    # Create ANOVA table
    anova_table = pd.DataFrame({
        'Source': ['Between Groups', 'Within Groups', 'Total'],
        'SS': [ss_between, ss_within, ss_between + ss_within],
        'df': [df_between, df_within, n_total - 1],
        'MS': [ms_between, ms_within, np.nan],
        'F': [f_value, np.nan, np.nan],
        'p-value': [p_value, np.nan, np.nan],
        'F critical (α=0.05)': [f_crit, np.nan, np.nan]
    }).set_index('Source')
    
    return df_long, anova_table.round(4)

# When file is uploaded
if uploaded_file is not None:
    try:
        # Read the file
        df = pd.read_csv(uploaded_file)
        
        # Show raw data
        st.subheader("Uploaded Data")
        st.dataframe(df)
        
        # Check if data is numeric
        if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            st.warning("Non-numeric data detected. Converting to numeric...")
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            if df.empty:
                st.error("No numeric data remaining after conversion")
                st.stop()
        
        # Run analysis
        df_long, anova_table = run_analysis(df)
        
        # Show ANOVA results
        st.subheader("ANOVA Results")
        st.dataframe(anova_table)
        
        # Interpretation
        p_value = anova_table.loc['Between Groups', 'p-value']
        if p_value < 0.05:
            st.success(f"✅ Significant difference found between groups (p = {p_value:.4f})")
            
            # Post-hoc Tukey test
            st.subheader("Tukey HSD Post-Hoc Test")
            tukey = pairwise_tukeyhsd(df_long['Value'], df_long['Group'])
            st.text(tukey.summary())
            
            # Plot results
            st.subheader("Visualizations")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Boxplot
            sns.boxplot(data=df_long, x='Group', y='Value', ax=ax1)
            ax1.set_title("Distribution by Group")
            
            # Tukey plot
            tukey.plot_simultaneous(ax=ax2)
            ax2.set_title("Tukey HSD Comparisons")
            
            st.pyplot(fig)
            
            # Show significant pairs
            results = pd.DataFrame(tukey._results_table.data[1:], 
                                 columns=tukey._results_table.data[0])
            sig_pairs = results[results['reject']]
            if not sig_pairs.empty:
                st.write("Significant differences:")
                for _, row in sig_pairs.iterrows():
                    st.write(f"- {row['group1']} vs {row['group2']}: p = {row['p-adj']:.4f}")
        else:
            st.warning(f"❌ No significant difference found (p = {p_value:.4f})")
        
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df_long.groupby('Group')['Value'].describe().round(4))
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis")

