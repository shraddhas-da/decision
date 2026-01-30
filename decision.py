import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize, StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Heart Disease Analytics Pro", layout="wide", page_icon="🩺")

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef; }
    h3 { color: #2c3e50; padding-top: 20px; }
    </style>
    """, unsafe_allow_html=True)


# --- DATA LOADING ---

import os

@st.cache_data
def load_data():
    file_path = "r"C:\Users\shraddha sharma\Machine Learning\heart.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path).drop_duplicates()
    else:
        # This will show you exactly where the app is looking on the server
        st.error(f"File not found! Current directory is: {os.getcwd()}")
        st.stop()
@st.cache_data
def load_data():
    try:
        # Changed to relative path for environment compatibility
        df = pd.read_csv(r"C:\Users\shraddha sharma\Machine Learning\heart.csv")
        return df.drop_duplicates()  # Removing duplicates for better accuracy
    except FileNotFoundError:
        st.error("Please ensure 'heart.csv' is in the same folder as this script.")
        st.stop()


df = load_data()

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.title("🌲 Model Control Center")
problem_mode = st.sidebar.radio("Objective", ["Binary: Heart Disease", "Multiclass: Chest Pain Type"])

if problem_mode == "Binary: Heart Disease":
    target_col = 'target'
    class_names = ["Healthy", "Disease"]
else:
    target_col = 'cp'
    class_names = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]

X = df.drop(target_col, axis=1)
y = df[target_col]

# --- AUTO-OPTIMIZATION LOGIC ---
if st.sidebar.button("🚀 Auto-Optimize for Max Accuracy"):
    with st.spinner("Finding best parameters..."):
        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'ccp_alpha': [0.0, 0.001, 0.005, 0.01]
        }
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
        grid.fit(X, y)
        st.session_state['best_params'] = grid.best_params_
        st.sidebar.success(f"Best Accuracy Found: {grid.best_score_:.2%}")

# Use optimized values if they exist, else use defaults
best = st.session_state.get('best_params', {})

# --- HYPERPARAMETERS (All parameters included) ---
with st.sidebar.expander("🛠️ All Hyperparameters", expanded=True):
    criterion = st.selectbox("Criterion", ("gini", "entropy", "log_loss"),
                             index=["gini", "entropy", "log_loss"].index(best.get('criterion', 'entropy')))
    splitter = st.selectbox("Splitter", ("best", "random"))
    max_depth = st.slider("Max Depth", 1, 30, best.get('max_depth', 10))
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.slider("Min Samples at Leaf", 1, 50, best.get('min_samples_leaf', 1))
    max_features = st.selectbox("Max Features", (None, "sqrt", "log2"))
    ccp_alpha = st.number_input("Pruning Alpha (ccp_alpha)", value=best.get('ccp_alpha', 0.0), format="%.3f")
    class_weight = st.selectbox("Class Weight", (None, "balanced"))

# --- MODEL ENGINE ---
# Standardizing features for better PCA and model stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    ccp_alpha=ccp_alpha,
    class_weight=class_weight,
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)

# --- MAIN INTERFACE ---
st.title("🫀 Heart Disease Diagnostic Lab")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Insights", "🧠 Model Training", "📉 Diagnostics", "🔮 Live Predictor"])

# --- TAB 1: DATA INSIGHTS ---
with tab1:
    st.subheader("Dataset Explorer")
    st.write("**Raw Dataset Preview**")
    st.dataframe(df, height=300)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.write("**Statistical Summary**")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'))
    with col_b:
        st.write("**Feature Correlation Matrix**")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

# --- TAB 2: MODEL TRAINING (Vertical Layout) ---
with tab2:
    st.subheader("Training Results & Model Logic")

    # Global Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("Nodes Created", clf.tree_.node_count)
    m3.metric("Depth Reached", clf.get_depth())
    m4.metric("Features Used", X.shape[1])

    st.markdown("---")

    # Diagram 1: PCA Decision Boundary (Large, Top)
    st.write("### 🌐 Decision Surface (PCA Projection)")
    st.info("Visualizing how the model separates classes in a 2D space.")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    clf_vis = DecisionTreeClassifier(max_depth=5).fit(X_pca, y_train)

    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig_pca, ax_pca = plt.subplots(figsize=(14, 7))
    ax_pca.contourf(xx, yy, Z, alpha=0.3, cmap='Spectral')
    ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolors='k', cmap='Spectral', s=50)
    ax_pca.set_xlabel("PC 1")
    ax_pca.set_ylabel("PC 2")
    st.pyplot(fig_pca)

    st.markdown("---")

    # Diagram 2: Feature Importance (Large, Bottom)
    st.write("### 🎯 Importance Hierarchy")
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig_imp, ax_imp = plt.subplots(figsize=(14, 8))
    feat_imp.plot(kind='barh', color='#3498db', ax=ax_imp)
    ax_imp.set_title("Relative Importance of All Features")
    st.pyplot(fig_imp)

# --- TAB 3: DIAGNOSTICS ---
with tab3:
    st.subheader("Advanced Evaluation")
    col_e, col_f = st.columns(2)
    with col_e:
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names)
        st.pyplot(fig_cm)
    with col_f:
        st.write("**ROC Curve Analysis**")
        fig_roc, ax_roc = plt.subplots()
        if problem_mode.startswith("Binary"):
            fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
            ax_roc.plot(fpr, tpr, label=f'AUC: {auc(fpr, tpr):.2f}', color='orange', lw=2)
        else:
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
            for i in range(4):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
                ax_roc.plot(fpr, tpr, label=f'Class {i} AUC: {auc(fpr, tpr):.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.legend()
        st.pyplot(fig_roc)

# --- TAB 4: LIVE PREDICTOR ---
with tab4:
    st.subheader("Patient Risk Assessment")
    st.write("Input patient data to simulate a live diagnostic prediction.")

    input_dict = {}
    input_cols = st.columns(4)  # Using 4 columns to fit all features neatly
    for i, col_name in enumerate(X.columns):
        with input_cols[i % 4]:
            val = df[col_name].median()
            if df[col_name].nunique() <= 5:
                unique_vals = sorted(df[col_name].unique())
                input_dict[col_name] = st.selectbox(f"{col_name}", unique_vals,
                                                    index=unique_vals.index(val) if val in unique_vals else 0)
            else:
                input_dict[col_name] = st.number_input(f"{col_name}", value=float(val))

    if st.button("Generate Diagnostic Report"):
        input_df = pd.DataFrame([input_dict])[X.columns]
        input_scaled = scaler.transform(input_df)  # Crucial: Scaling the user input

        prediction = clf.predict(input_scaled)[0]
        confidence = np.max(clf.predict_proba(input_scaled))

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.success(f"### Predicted Result: {class_names[prediction]}")
        c2.info(f"### Prediction Confidence: {confidence:.2%}")

        if problem_mode.startswith("Binary") and prediction == 1:
            st.warning("⚠️ ALERT: High risk for heart disease. Clinical consultation recommended.")
        else:
            st.success("✅ Patient profiles as low risk under current parameters.")
