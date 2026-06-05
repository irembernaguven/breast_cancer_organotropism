import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import networkx as nx

# ---------------------------------------------------------
# PAGE CONFIGURATION & PREMIUM STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Metastasis Dashboard", layout="wide", page_icon="🧬")

# Custom premium CSS injection for clean scientific branding
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #3498db;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #7f8c8d;
        font-weight: bold;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 24px;
        color: #2c3e50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Metastatic Breast Cancer Organotropism Dashboard")
st.markdown("""
Welcome to the interactive exploration dashboard. This platform analyzes the clinical and genomic drivers of site-specific breast cancer metastasis (organotropism)
using the cBioPortal dataset **"Metastatic Breast Cancer (MSK, Cancer Discovery 2022)"**.
Select a target metastatic site from the sidebar to partition the cohort and run statistical and comparative analyses.
""")

# ---------------------------------------------------------
# DATA LOADING (CACHED FOR PERFORMANCE)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    clinical_file = "/Users/irembernaguven/Downloads/cancer project/breast_ink4_msk_2021_clinical_data.tsv"
    mutation_file = "/Users/irembernaguven/Downloads/cancer project/breast_ink4_msk_2021/data_mutations.txt"
    cna_file = "/Users/irembernaguven/Downloads/cancer project/breast_ink4_msk_2021/data_cna.txt"
    
    # Load clinical and mutation data using comment='#' to ignore metadata headers
    clin_data = pd.read_csv(clinical_file, sep='\t', comment='#')
    mut_data = pd.read_csv(mutation_file, sep='\t', comment='#', low_memory=False)
    
    # Load copy number alteration data (CNA has no comments in line 1)
    cna_data = pd.read_csv(cna_file, sep='\t')
    
    # Pre-clean Age and TMB to numeric types
    clin_data['Patient Current Age'] = pd.to_numeric(clin_data['Patient Current Age'], errors='coerce')
    clin_data['TMB (nonsynonymous)'] = pd.to_numeric(clin_data['TMB (nonsynonymous)'], errors='coerce')
    
    return clin_data, mut_data, cna_data

with st.spinner('Loading clinical and genomic datasets...'):
    clinical_df, mutation_df, cna_df = load_data()

# ---------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
st.sidebar.header("Filter Settings")

# Find valid metastatic sites (only keep sites with > 10 patients for statistical significance)
site_counts = clinical_df['Metastatic Site'].value_counts()
valid_sites = site_counts[site_counts > 10].index.tolist()

# Dropdown for user selection
selected_site = st.sidebar.selectbox(
    "Select Target Metastasis Site:", 
    valid_sites, 
    index=valid_sites.index('Lung') if 'Lung' in valid_sites else 0
)

# Create binary target column based on user selection
clinical_df['Target'] = clinical_df['Metastatic Site'].apply(
    lambda x: 1 if pd.notnull(x) and selected_site in str(x) else 0
)

# Define Target and Control cohorts
target_sample_ids = clinical_df[clinical_df['Target'] == 1]['Sample ID'].dropna().tolist()
control_sample_ids = clinical_df[clinical_df['Target'] == 0]['Sample ID'].dropna().tolist()

target_count = len(target_sample_ids)
control_count = len(control_sample_ids)
total_count = len(clinical_df)

st.sidebar.success(f"**Cohort Selected:**\n- {selected_site}: {target_count} patients\n- Control (Others): {control_count} patients")

st.markdown("---")

# ---------------------------------------------------------
# DASHBOARD TABS LAYOUT
# ---------------------------------------------------------
tab_demographics, tab_mutations, tab_enrichment, tab_cna, tab_tmb, tab_ml, tab_network = st.tabs([
    "📊 Cohort Demographics",
    "🧬 Mutation Frequencies",
    "🧪 Genomic Enrichment (Fisher's)",
    "📈 Copy Number Alterations (CNA)",
    "🔬 Tumor Mutational Burden (TMB)",
    "🤖 Machine Learning Classifier",
    "🕸️ Network & Co-occurrence"
])

# =========================================================
# TAB 1: COHORT OVERVIEW & DEMOGRAPHICS
# =========================================================
with tab_demographics:
    st.header(f"📊 Demographics: {selected_site} vs Control")
    
    # Metrics display row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e74c3c;">
            <div class="metric-title">Target Cohort ({selected_site})</div>
            <div class="metric-value">{target_count} Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #95a5a6;">
            <div class="metric-title">Control Cohort (Others)</div>
            <div class="metric-value">{control_count} Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2ecc71;">
            <div class="metric-title">Study Representation</div>
            <div class="metric-value">{(target_count / total_count) * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.subheader("Age Distribution")
        fig_age, ax_age = plt.subplots(figsize=(8, 5))
        sns.histplot(data=clinical_df, x='Patient Current Age', hue='Target', kde=True, 
                     palette=['#7f8c8d', '#e74c3c'], multiple='dodge', bins=15, ax=ax_age)
        ax_age.set_xlabel("Patient Current Age")
        ax_age.set_ylabel("Count")
        legend = ax_age.get_legend()
        if legend:
            legend.set_title("Cohort")
            for t, l in zip(legend.texts, ['Control', selected_site]):
                t.set_text(l)
        st.pyplot(fig_age)
        
    with d_col2:
        st.subheader("Sex Distribution")
        sex_counts = clinical_df.groupby(['Sex', 'Target']).size().unstack(fill_value=0)
        sex_counts.columns = ['Control', selected_site]
        
        fig_sex, ax_sex = plt.subplots(figsize=(8, 5))
        sex_counts.plot(kind='bar', stacked=True, color=['#7f8c8d', '#e74c3c'], ax=ax_sex)
        ax_sex.set_ylabel("Number of Patients")
        plt.xticks(rotation=0)
        st.pyplot(fig_sex)

    st.subheader("Top Detailed Cancer Types in Target Cohort")
    target_cancer_types = clinical_df[clinical_df['Target'] == 1]['Cancer Type Detailed'].value_counts()
    st.dataframe(target_cancer_types, column_config={"Cancer Type Detailed": "Counts"})

# =========================================================
# TAB 2: MUTATION FREQUENCIES
# =========================================================
with tab_mutations:
    st.header(f"🧬 Top Mutated Genes")
    
    target_mutations = mutation_df[mutation_df['Tumor_Sample_Barcode'].isin(target_sample_ids)]
    control_mutations = mutation_df[mutation_df['Tumor_Sample_Barcode'].isin(control_sample_ids)]
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.subheader(f"Top mutated in {selected_site} Cohort")
        if not target_mutations.empty:
            target_top_genes = target_mutations['Hugo_Symbol'].value_counts().head(10)
            target_gene_percentages = (target_top_genes / target_count) * 100
            
            fig_t_genes, ax_t_genes = plt.subplots(figsize=(8, 6))
            sns.barplot(x=target_gene_percentages.values, y=target_gene_percentages.index, 
                        palette='Reds_r', hue=target_gene_percentages.index, legend=False, ax=ax_t_genes)
            ax_t_genes.set_xlabel('Mutation Frequency (%)')
            ax_t_genes.set_ylabel('Gene Symbol')
            st.pyplot(fig_t_genes)
        else:
            st.warning("No mutation data found for the Target cohort.")
            
    with m_col2:
        st.subheader("Top mutated in Control Cohort")
        if not control_mutations.empty:
            control_top_genes = control_mutations['Hugo_Symbol'].value_counts().head(10)
            control_gene_percentages = (control_top_genes / control_count) * 100
            
            fig_c_genes, ax_c_genes = plt.subplots(figsize=(8, 6))
            sns.barplot(x=control_gene_percentages.values, y=control_gene_percentages.index, 
                        palette='Greys_r', hue=control_gene_percentages.index, legend=False, ax=ax_c_genes)
            ax_c_genes.set_xlabel('Mutation Frequency (%)')
            ax_c_genes.set_ylabel('Gene Symbol')
            st.pyplot(fig_c_genes)
        else:
            st.warning("No mutation data found for the Control cohort.")

# =========================================================
# TAB 3: GENOMIC ENRICHMENT (FISHER'S EXACT TEST)
# =========================================================
with tab_enrichment:
    st.header("🧪 Mutational Enrichment Analysis")
    st.markdown(f"""
    This analysis identifies genes that are **statistically enriched or depleted** in the selected metastasis site ({selected_site}) 
    compared to all other sites using **Fisher's Exact Test** for each gene.
    """)
    
    @st.cache_data
    def run_enrichment(target_samples, control_samples, mut_df):
        n_t = len(target_samples)
        n_c = len(control_samples)
        if n_t == 0 or n_c == 0:
            return pd.DataFrame()
            
        # Group mutated counts per gene
        target_counts = mut_df[mut_df['Tumor_Sample_Barcode'].isin(target_samples)].groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique()
        control_counts = mut_df[mut_df['Tumor_Sample_Barcode'].isin(control_samples)].groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique()
        
        all_genes = set(target_counts.index).union(set(control_counts.index))
        enrichment_results = []
        
        for gene in all_genes:
            a = target_counts.get(gene, 0)
            c = control_counts.get(gene, 0)
            b = n_t - a
            d = n_c - c
            
            # Filter to avoid testing low frequency mutations (must have at least 2 target mutations)
            if a >= 2:
                odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
                enrichment_results.append({
                    'Gene': gene,
                    'Target Mutated': a,
                    f'Target %': (a / n_t) * 100,
                    'Control Mutated': c,
                    f'Control %': (c / n_c) * 100,
                    'Odds Ratio': odds_ratio,
                    'p-value': p_value
                })
                
        df_res = pd.DataFrame(enrichment_results)
        if not df_res.empty:
            df_res = df_res.sort_values('p-value')
        return df_res

    # Run and display enrichment results
    with st.spinner('Calculating Fisher\'s Exact Test for all genes...'):
        enrichment_df = run_enrichment(target_sample_ids, control_sample_ids, mutation_df)
        
    if not enrichment_df.empty:
        p_threshold = st.slider("Select p-value threshold:", min_value=0.01, max_value=1.00, value=0.05, step=0.01)
        
        # Display significant genes
        filtered_df = enrichment_df[enrichment_df['p-value'] <= p_threshold].copy()
        filtered_df['Significant'] = filtered_df['p-value'].apply(lambda p: '✅ Yes' if p < 0.05 else '❌ No')
        
        # Style formatting
        formatted_df = filtered_df.style.format({
            'Target %': '{:.2f}%',
            'Control %': '{:.2f}%',
            'Odds Ratio': '{:.2f}',
            'p-value': '{:.4e}'
        })
        
        st.subheader(f"Enrichment Results (p-value <= {p_threshold})")
        st.dataframe(formatted_df, use_container_width=True)
        
        # Export as CSV option
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Enrichment Table as CSV",
            data=csv_data,
            file_name=f"{selected_site}_genomic_enrichment.csv",
            mime="text/csv"
        )
    else:
        st.info("No genes met the minimum mutation frequency requirements for statistical enrichment calculation.")

# =========================================================
# TAB 4: COPY NUMBER ALTERATIONS (CNA)
# =========================================================
with tab_cna:
    st.header("📈 Copy Number Alterations (CNA)")
    st.markdown("""
    Compare copy number alteration profiles between your Target metastasis cohort and the Control cohort.
    CNA values map to cBioPortal categories: **-2: Deep Deletion, -1: Shallow Loss, 0: Diploid, 1: Gain, 2: Amplification**.
    """)
    
    # Autocomplete gene selection
    cna_genes = sorted(cna_df['Hugo_Symbol'].unique().tolist())
    selected_cna_gene = st.selectbox("Search and Select Gene of Interest:", cna_genes, index=cna_genes.index('CDKN2A') if 'CDKN2A' in cna_genes else 0)
    
    # Extract copy number values
    gene_cna = cna_df[cna_df['Hugo_Symbol'] == selected_cna_gene].iloc[0]
    
    # Match samples
    cna_samples = [c for c in cna_df.columns if c != 'Hugo_Symbol']
    matched_target = [c for c in target_sample_ids if c in cna_samples]
    matched_control = [c for c in control_sample_ids if c in cna_samples]
    
    if len(matched_target) > 0 and len(matched_control) > 0:
        target_cna_vals = gene_cna[matched_target].values
        control_cna_vals = gene_cna[matched_control].values
        
        label_map = {-2: 'Deep Deletion (-2)', -1: 'Shallow Loss (-1)', 0: 'Diploid (0)', 1: 'Gain (1)', 2: 'Amplification (2)'}
        
        # Calculate percentages
        t_counts = pd.Series(target_cna_vals).value_counts().reindex([-2, -1, 0, 1, 2], fill_value=0)
        c_counts = pd.Series(control_cna_vals).value_counts().reindex([-2, -1, 0, 1, 2], fill_value=0)
        
        t_pct = (t_counts / len(matched_target)) * 100
        c_pct = (c_counts / len(matched_control)) * 100
        
        # Plot stacked bar chart
        plot_df = pd.DataFrame({'Target': t_pct, 'Control': c_pct}).T
        plot_df.columns = [label_map[x] for x in plot_df.columns]
        
        fig_cna, ax_cna = plt.subplots(figsize=(10, 5))
        plot_df.plot(kind='barh', stacked=True, color=['#2980b9', '#3498db', '#bdc3c7', '#e74c3c', '#c0392b'], ax=ax_cna)
        ax_cna.set_xlabel('Percentage of Patients (%)')
        ax_cna.set_title(f'Copy Number Status of {selected_cna_gene}')
        ax_cna.set_yticklabels([selected_site, 'Control'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        st.pyplot(fig_cna)
    else:
        st.warning("Insufficient samples with CNA data for comparison.")

# =========================================================
# TAB 5: TUMOR MUTATIONAL BURDEN (TMB)
# =========================================================
with tab_tmb:
    st.header("🔬 Tumor Mutational Burden (TMB)")
    st.markdown("""
    Tumor Mutational Burden (TMB) indicates the count of mutations per megabase. We check if the target metastatic group 
    displays a statistically distinct mutational burden compared to all other sites using the **Mann-Whitney U Test**.
    """)
    
    target_tmb = clinical_df[clinical_df['Target'] == 1]['TMB (nonsynonymous)'].dropna()
    other_tmb = clinical_df[clinical_df['Target'] == 0]['TMB (nonsynonymous)'].dropna()
    
    if len(target_tmb) > 0 and len(other_tmb) > 0:
        stat, p_value = mannwhitneyu(target_tmb, other_tmb, alternative='two-sided')
        
        fig_tmb, ax_tmb = plt.subplots(figsize=(8, 5))
        sns.violinplot(x='Target', y='TMB (nonsynonymous)', data=clinical_df, 
                       palette=['#95a5a6', '#e74c3c'], hue='Target', legend=False, inner='quartile', ax=ax_tmb)
        
        ax_tmb.set_xticks([0, 1])
        ax_tmb.set_xticklabels(['Control (Others)', f'{selected_site}'])
        ax_tmb.set_xlabel('')
        ax_tmb.set_ylabel('TMB (Mutations per MB)')
        
        ax_tmb.annotate(f'Mann-Whitney p-value: {p_value:.4f}', 
                     xy=(0.5, 0.90), xycoords='axes fraction', 
                     ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
        
        st.pyplot(fig_tmb)
        
        # Interpret result
        if p_value < 0.05:
            st.success(f"🎉 **Result is Statistically Significant (p < 0.05):** There is a significant difference in mutational burden (TMB) between the {selected_site} metastasis group and the control group.")
        else:
            st.info(f"ℹ️ **Result is Not Statistically Significant (p >= 0.05):** The mutational burden (TMB) profiles between {selected_site} metastasis and other cohorts are statistically similar.")
    else:
        st.warning("Not enough TMB data available for this site.")

# =========================================================
# TAB 6: MACHINE LEARNING CLASSIFIER
# =========================================================
with tab_ml:
    st.header("🤖 Machine Learning Classifier")
    st.markdown(f"""
    Predict the probability of a patient developing metastasis in the **{selected_site}** based on their clinical features 
    (Age, TMB) and the mutational profiles of the top mutated genes.
    """)
    
    # Selection of classifier
    classifier_name = st.selectbox("Select Machine Learning Model:", ["Random Forest", "Logistic Regression"])
    
    # Prepare data for Machine Learning
    # Find the top 20 mutated genes across the entire study
    top_study_genes = mutation_df['Hugo_Symbol'].value_counts().head(20).index.tolist()
    
    # Build a binary mutation matrix for all samples
    mut_matrix = mutation_df[mutation_df['Hugo_Symbol'].isin(top_study_genes)].pivot_table(
        index='Tumor_Sample_Barcode', columns='Hugo_Symbol', aggfunc='size', fill_value=0
    )
    mut_matrix = (mut_matrix > 0).astype(int)
    
    # Merge clinical features and target with mutation matrix
    ml_base = clinical_df[['Sample ID', 'Patient Current Age', 'TMB (nonsynonymous)', 'Target']].dropna()
    ml_df = pd.merge(ml_base, mut_matrix, left_on='Sample ID', right_index=True, how='left').fillna(0)
    
    # Define features (X) and target (y)
    feature_cols = ['Patient Current Age', 'TMB (nonsynonymous)'] + top_study_genes
    X = ml_df[feature_cols]
    y = ml_df['Target']
    
    if len(ml_df) > 50 and y.sum() > 5:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Train model
        if classifier_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            
        model.fit(X_train, y_train)
        
        # Predict probability
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Compute ROC-AUC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Metrics presentation
        ml_m1, ml_m2, ml_m3, ml_m4 = st.columns(4)
        with ml_m1:
            st.metric("Test Accuracy", f"{acc:.2%}")
        with ml_m2:
            st.metric("Precision", f"{prec:.2%}")
        with ml_m3:
            st.metric("Recall / Sensitivity", f"{rec:.2%}")
        with ml_m4:
            st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
            
        ml_c1, ml_c2 = st.columns(2)
        
        with ml_c1:
            st.subheader("📈 ROC (Receiver Operating Characteristic) Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='#34495e', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Model ROC Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            
        with ml_c2:
            st.subheader("📊 Feature Importance Plot")
            fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
            if classifier_name == "Random Forest":
                importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(10)
                importances.plot(kind='barh', color='#e74c3c', ax=ax_imp)
                ax_imp.set_xlabel('Feature Importance')
            else:
                coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=True)
                # Keep top 10 strongest features (absolute weight)
                coefs_sorted = coefs.reindex(coefs.abs().sort_values().tail(10).index)
                coefs_sorted.plot(kind='barh', color='#3498db', ax=ax_imp)
                ax_imp.set_xlabel('Model Coefficient')
            ax_imp.set_ylabel('Features')
            st.pyplot(fig_imp)
    else:
        st.warning("Insufficient samples or positive metastasis classes to perform Machine Learning training.")
        
    st.markdown("---")
    
    # Option to view static pre-calculated figure
    show_static_ml = st.checkbox("Show Static Multi-omics Feature Importance Figure (Previously Generated)")
    if show_static_ml:
        static_img_path = "/Users/irembernaguven/Downloads/cancer project/multiomics_feature_importance.png"
        if os.path.exists(static_img_path):
            st.image(static_img_path, caption="Pre-calculated Multi-omics Feature Importance Heatmap / Chart", use_container_width=True)
        else:
            st.error("Static image 'multiomics_feature_importance.png' was not found in the project folder.")

# =========================================================
# TAB 7: NETWORK & CO-OCCURRENCE ANALYSIS
# =========================================================
with tab_network:
    st.header("🕸️ Gene Mutation Co-occurrence & Network Analysis")
    st.markdown("""
    Explore relationships of co-occurrence or mutual exclusivity between gene mutations. 
    Genes that frequently mutate together indicate synergistic pathway activation, while mutually exclusive mutations point to redundant mechanisms.
    """)
    
    # Calculate Jaccard / correlation for top 15 mutated genes
    top_net_genes = mutation_df['Hugo_Symbol'].value_counts().head(15).index.tolist()
    
    # Binary mutation matrix
    net_matrix = mutation_df[mutation_df['Hugo_Symbol'].isin(top_net_genes)].pivot_table(
        index='Tumor_Sample_Barcode', columns='Hugo_Symbol', aggfunc='size', fill_value=0
    )
    net_matrix = (net_matrix > 0).astype(int)
    
    # Calculate Pearson Correlation
    corr_matrix = net_matrix.corr()
    
    net_c1, net_c2 = st.columns(2)
    
    with net_c1:
        st.subheader("🔥 Mutation Co-occurrence Correlation Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1.0, vmax=1.0, square=True, cbar=True, ax=ax_heat)
        st.pyplot(fig_heat)
        
    with net_c2:
        st.subheader("🕸️ Dynamic Interaction Network")
        st.markdown("Specify the correlation threshold to draw edges in the network graph:")
        
        # Edge correlation threshold slider
        edge_threshold = st.slider("Correlation Threshold (Absolute value):", min_value=0.01, max_value=0.30, value=0.05, step=0.01)
        
        # Build NetworkX graph
        G = nx.Graph()
        for gene in top_net_genes:
            G.add_node(gene)
            
        for i in range(len(top_net_genes)):
            for j in range(i+1, len(top_net_genes)):
                gene_a = top_net_genes[i]
                gene_b = top_net_genes[j]
                val = corr_matrix.loc[gene_a, gene_b]
                if abs(val) >= edge_threshold:
                    G.add_edge(gene_a, gene_b, weight=val)
                    
        # Draw NetworkX graph
        fig_net, ax_net = plt.subplots(figsize=(8, 8))
        pos = nx.spring_layout(G, k=0.6, seed=42)
        
        edges = G.edges()
        if len(edges) > 0:
            weights = [G[u][v]['weight'] for u,v in edges]
            edge_colors = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights]
            edge_widths = [abs(w) * 15 for w in weights]
            
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='#34495e', ax=ax_net)
            nx.draw_networkx_labels(G, pos, font_color='white', font_size=9, font_weight='bold', ax=ax_net)
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors, ax=ax_net)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='#34495e', ax=ax_net)
            nx.draw_networkx_labels(G, pos, font_color='white', font_size=9, font_weight='bold', ax=ax_net)
            st.info("No connections found at this threshold. Try lowering the threshold.")
            
        ax_net.axis('off')
        st.pyplot(fig_net)
        st.markdown("**Legend:** Green edges indicate **positive co-occurrence** (mutating together). Red edges indicate **negative correlation / mutual exclusivity**.")

    st.markdown("---")
    
    # Option to view static pre-calculated figure
    show_static_net = st.checkbox("Show Static Mutation Co-occurrence Heatmap Figure (Previously Generated)")
    if show_static_net:
        static_net_path = "/Users/irembernaguven/Downloads/cancer project/mutation_cooccurrence_heatmap.png"
        if os.path.exists(static_net_path):
            st.image(static_net_path, caption="Pre-calculated Mutation Co-occurrence Correlation Heatmap", use_container_width=True)
        else:
            st.error("Static image 'mutation_cooccurrence_heatmap.png' was not found in the project folder.")

# ---------------------------------------------------------
# FOOTER / INSIGHTS
# ---------------------------------------------------------
st.markdown("---")
st.info(f"**Data Scientist Note:** This dashboard dynamically queries the MSK-IMPACT 2022 dataset to discover genomic signatures specific to {selected_site} metastasis. Try selecting different organs (like Liver or Bone) from the sidebar to see how the genomic landscape completely shifts based on the tumor's destination.")