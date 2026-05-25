import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

st.set_page_config(page_title="CE Numerical Methods Lab", layout="wide")

# Force Streamlit to hide the top-right toolbar and menus for a clean interface
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.title("🧮 CE Numerical Methods Lab App")
st.markdown("Use this web calculator to experiment with iterative linear solvers and eigenvalue problems.")

# Sidebar Controls
st.sidebar.header("Select Analysis Module")
module = st.sidebar.radio("Choose Method:", ["Jacobi / GS / SOR Comparison", "Power Method (Eigenvalues)"])

# ==========================================
# MODULE 1: JACOBI VS GS VS SOR COMPARISON
# ==========================================
if module == "Jacobi / GS / SOR Comparison":
    st.header("Linear Systems: Variable Trajectory Race")
    st.markdown("Compare convergence paths for: **Jacobi**, **Gauss-Seidel (GS)**, and **Successive Over-Relaxation (SOR)**.")
    
    st.sidebar.subheader("System Parameters")
    lam_val = st.sidebar.slider("Lambda (λ) for SOR:", 0.1, 2.0, 1.25, step=0.05)
    max_iter = st.sidebar.number_input("Max Iterations:", min_value=5, max_value=100, value=25)
    
    # 3x3 Grid Inputs layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matrix $[A]$")
        # Creating manual row columns for smooth grid entry
        r1 = st.columns(3)
        a11 = r1[0].number_input("A[1,1]", value=10.0, key="a11")
        a12 = r1[1].number_input("A[1,2]", value=2.0, key="a12")
        a13 = r1[2].number_input("A[1,3]", value=2.0, key="a13")
        
        r2 = st.columns(3)
        a21 = r2[0].number_input("A[2,1]", value=2.0, key="a21")
        a22 = r2[1].number_input("A[2,2]", value=10.0, key="a22")
        a23 = r2[2].number_input("A[2,3]", value=2.0, key="a23")
        
        r3 = st.columns(3)
        a31 = r3[0].number_input("A[3,1]", value=2.0, key="a31")
        a32 = r3[1].number_input("A[3,2]", value=2.0, key="a32")
        a33 = r3[2].number_input("A[3,3]", value=10.0, key="a33")
        
    with col2:
        st.subheader("Vector $[b]$")
        b1 = st.number_input("b[1]", value=10.0)
        b2 = st.number_input("b[2]", value=25.0)
        b3 = st.number_input("b[3]", value=40.0)

    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    b = np.array([b1, b2, b3])

    def run_solver(A, b, method, lam=1.0, max_iter=25):
        n = len(b)
        x = np.zeros(n)
        trajectories = [x.copy()]
        for k in range(max_iter):
            x_old = x.copy()
            if method == "Jacobi":
                x_new = np.zeros(n)
                for i in range(n):
                    s = sum(A[i][j] * x_old[j] for j in range(n) if i != j)
                    x_new[i] = (b[i] - s) / A[i][i]
                x = x_new
            else:
                for i in range(n):
                    s1 = sum(A[i][j] * x[j] for j in range(i))
                    s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
                    x_gs = (b[i] - s1 - s2) / A[i][i]
                    if method == "SOR":
                        x[i] = (1 - lam) * x_old[i] + lam * x_gs
                    else:
                        x[i] = x_gs
            trajectories.append(x.copy())
            if np.linalg.norm(x - x_old, ord=np.inf) < 1e-7: break
        return np.array(trajectories)

    if st.button("Compare Solution Paths", type="primary"):
        j_traj = run_solver(A, b, "Jacobi", max_iter=max_iter)
        gs_traj = run_solver(A, b, "Gauss-Seidel", max_iter=max_iter)
        sor_traj = run_solver(A, b, "SOR", lam=lam_val, max_iter=max_iter)

        st.markdown("---")
        st.subheader("📊 Execution Summary Results")
        
        # Display final converged state nicely
        st.success(f"**Final Converged Solution Vector $[x]$:** {np.round(sor_traj[-1], 5)}")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Jacobi Iterations", len(j_traj))
        metric_col2.metric("Gauss-Seidel Iterations", len(gs_traj))
        metric_col3.metric("SOR Iterations", len(sor_traj))

        # Re-generating your Matplotlib plot inside Streamlit
        fig, ax = plt.subplots(figsize=(10, 5))
        markers = ['o', 's', '^']
        
        for i in range(3):
            ax.plot(j_traj[:, i], color='red', marker=markers[i], alpha=0.3, label=f'x{i+1} (Jacobi)' if i==0 else "")
            ax.plot(gs_traj[:, i], color='blue', marker=markers[i], alpha=0.4, label=f'x{i+1} (GS)' if i==0 else "")
            ax.plot(sor_traj[:, i], color='green', marker=markers[i], linewidth=2, label=f'x{i+1} (SOR)' if i==0 else "")

        ax.set_title(f"Convergence Comparison (SOR Lambda = {lam_val})")
        ax.set_xlabel("Iteration Number")
        ax.set_ylabel("Value of Variable x_i")
        
        custom_lines = [
            Line2D([0], [0], color='red', lw=2, label='Jacobi'),
            Line2D([0], [0], color='blue', lw=2, label='Gauss-Seidel'),
            Line2D([0], [0], color='green', lw=2, label='SOR'),
            Line2D([0], [0], color='black', marker='o', lw=0, label='Variable x1'),
            Line2D([0], [0], color='black', marker='s', lw=0, label='Variable x2'),
            Line2D([0], [0], color='black', marker='^', lw=0, label='Variable x3')
        ]
        ax.legend(handles=custom_lines, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# ==========================================
# MODULE 2: POWER METHOD
# ==========================================
else:
    st.header("Eigenvalue Problems: Dominant Power Method")
    st.markdown("Track how an arbitrary vector shifts iterations to find the dominant eigenvalue using the Rayleigh Quotient.")
    
    n = st.sidebar.number_input("Matrix Size (n):", min_value=2, max_value=4, value=2)
    
    st.subheader(f"Input elements for {n}x{n} Matrix $[A]$")
    
    matrix_inputs = []
    for i in range(n):
        cols = st.columns(n)
        row_inputs = []
        for j in range(n):
            default_val = 2.0 if i==0 and j==0 else (-12.0 if i==0 and j==1 else (1.0 if i==1 and j==0 else -5.0)) if n==2 else (1.0 if i==j else 0.1)
            val = cols[j].number_input(f"A[{i+1},{j+1}]", value=float(default_val), key=f"p_{i}_{j}")
            row_inputs.append(val)
        matrix_inputs.append(row_inputs)

    if st.button("Run Power Method", type="primary"):
        A = np.array(matrix_inputs)
        
        # Consistent static seed initial vector initialization for student demonstration
        v = np.array([1.0] * n)
        v = v / np.linalg.norm(v)
        history = []
        iter_log = []
        
        for i in range(31):
            w = np.dot(A, v)
            rayleigh_lambda = np.dot(v.T, w) / np.dot(v.T, v)
            history.append(rayleigh_lambda)
            v = w / np.linalg.norm(w)
            
            if i % 5 == 0 or i == 30:
                iter_log.append({"Iteration": i, "Dominant Rayleigh λ Estimate": round(rayleigh_lambda, 6)})

        st.markdown("---")
        st.subheader("🎯 Computed Output Results")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Dominant Eigenvalue (λ)", f"{rayleigh_lambda:.6f}")
            st.write("**Dominant Eigenvector:**", np.round(v, 6))
            st.dataframe(pd.DataFrame(iter_log), use_container_width=True)
            
        with col_res2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history, marker='o', color='tab:red', markersize=4)
            ax.set_title("Convergence to Dominant Eigenvalue")
            ax.set_xlabel("Iteration Step")
            ax.set_ylabel("Estimated λ Value")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
