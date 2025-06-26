#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, f
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set up the app
st.set_page_config(page_title="ANOVA Analyzer", layout="wide")
st.title("General ANOVA Analysis Tool")
st.markdown("Analyze your data with complete ANOVA table and post-hoc tests")

# File upload section
uploaded_file = st.file_uploader(
    "Upload your data (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Upload data with measurements in columns (each column represents a different condition/group)"
)

# Check if file is uploaded
if uploaded_file is None:
    st.warning("Please upload a file to analyze your data")
    st.stop()

# Read the file
try:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {str(e)}")
    st.stop()

# Ensure numeric data
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

# Show raw data
st.subheader("Your Data")
st.dataframe(df)

# Get column names (conditions/groups)
conditions = df.columns.tolist()

# Convert to long format
df_long = df.melt(var_name='Condition', value_name='Value')

# Calculate ANOVA table manually
def calculate_anova_table(data_long):
    groups = data_long.groupby('Condition')['Value']
    n_groups = len(groups)
    n_total = len(data_long)
    grand_mean = data_long['Value'].mean()
    
    # Between-groups
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
    df_between = n_groups - 1
    ms_between = ss_between / df_between
    
    # Within-groups
    ss_within = sum(sum((x - g.mean())**2 for x in g) for _, g in groups)
    df_within = n_total - n_groups
    ms_within = ss_within / df_within
    
    # F-statistic
    f_value = ms_between / ms_within
    p_value = f.sf(f_value, df_between, df_within)
    f_crit = f.ppf(0.95, df_between, df_within)
    
    # Create table
    anova_table = pd.DataFrame({
        'Source': ['Between Conditions', 'Within Conditions', 'Total'],
        'SS': [ss_between, ss_within, ss_between + ss_within],
        'df': [df_between, df_within, n_total - 1],
        'MS': [ms_between, ms_within, np.nan],
        'F': [f_value, np.nan, np.nan],
        'p-value': [p_value, np.nan, np.nan],
        'F crit (α=0.05)': [f_crit, np.nan, np.nan]
    }).set_index('Source')
    
    return anova_table.round(4)

# Perform analysis
st.subheader("ANOVA Results")
anova_table = calculate_anova_table(df_long)
st.dataframe(anova_table)

# Visualization
st.subheader("Data Visualization")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot
sns.boxplot(data=df_long, x='Condition', y='Value', ax=ax1)
ax1.set_title("Value Distribution by Condition")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# Individual points
sns.swarmplot(data=df_long, x='Condition', y='Value', color='black', alpha=0.7, ax=ax2)
ax2.set_title("Individual Measurements")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

st.pyplot(fig)

# Interpretation
p_value = anova_table.loc['Between Conditions', 'p-value']
if p_value < 0.05:
    st.success("✅ Significant difference found between conditions (p < 0.05)")
    
    # Post-hoc tests
    st.subheader("Post-Hoc Analysis (Tukey HSD)")
    tukey = pairwise_tukeyhsd(df_long['Value'], df_long['Condition'])
    st.text(tukey.summary())
    
    # Plot results
    fig2, ax = plt.subplots(figsize=(10, 4))
    tukey.plot_simultaneous(ax=ax)
    ax.set_title("Tukey HSD Group Comparisons")
    st.pyplot(fig2)
    
    # Show significant pairs
    results = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    sig_pairs = results[results['reject']]
    if not sig_pairs.empty:
        st.write("Significant differences:")
        for _, row in sig_pairs.iterrows():
            st.write(f"- {row['group1']} vs {row['group2']}: p = {row['p-adj']:.4f}")
else:
    st.warning("❌ No significant difference found (p ≥ 0.05)")

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.dataframe(df_long.groupby('Condition')['Value'].describe().round(4))

