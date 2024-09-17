import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import pyswarms as ps
import pandas as pd

import warnings
st.set_page_config(layout = "wide", initial_sidebar_state = "expanded")
# Suppress all warnings
warnings.filterwarnings("ignore")

def compute_P_ppt(p):
    """
    Compute the matrix P_ppt.

    Parameters:
    p (np.ndarray): kx1 vector of multinomial distribution probabilities

    Returns:
    np.ndarray: Diagonal matrix P_ppt - outer product of p with itself
    """
    return np.diag(p) - np.outer(p, p)

def f1(x):
    """
    Compute the vector f1.

    Parameters:
    x (float): Dose value

    Returns:
    np.ndarray: Vector [1, x]
    """
    return np.array([1, x])

def f2(x):
    """
    Compute the vector f2.

    Parameters:
    x (float): Dose value

    Returns:
    np.ndarray: Vector [1, x]
    """
    return np.array([1, x])

def compute_u1(cache1, cache2, rho):
    """
    Compute u1 based on given parameters.

    Parameters:
    cache1 (float): Precomputed value theta1.dot(f1(x))
    cache2 (float): Precomputed value theta2.dot(f2(x))
    rho (float): Correlation coefficient

    Returns:
    float: Computed value of u1
    """
    return (cache2 - rho * cache1) / np.sqrt(1 - rho ** 2)

def compute_u2(cache1, cache2, rho):
    """
    Compute u2 based on given parameters.

    Parameters:
    cache1 (float): Precomputed value theta1.dot(f1(x))
    cache2 (float): Precomputed value theta2.dot(f2(x))
    rho (float): Correlation coefficient

    Returns:
    float: Computed value of u2
    """
    return (cache1 - rho * cache2) / np.sqrt(1 - rho ** 2)

def compute_C1(x, cache1, cache2):
    """
    Compute the matrix C1.

    Parameters:
    x (float): Dose value
    cache1 (float): Precomputed value theta1.dot(f1(x))
    cache2 (float): Precomputed value theta2.dot(f2(x))

    Returns:
    np.ndarray: Matrix C1
    """
    vec1 = norm.pdf(cache1) * f1(x)
    vec2 = norm.pdf(cache2) * f2(x)

    # Create columns of zeros
    zeros_vec1 = np.zeros_like(vec1)
    zeros_vec2 = np.zeros_like(vec2)

    # Concatenate vec1 with zeros and vec2 with zeros
    vec1_with_zeros = np.vstack((vec1, zeros_vec1)).T
    vec2_with_zeros = np.vstack((zeros_vec2, vec2)).T

    # Concatenate vertically to get the final matrix
    return np.vstack((vec1_with_zeros, vec2_with_zeros))

def compute_C2(u1, u2):
    """
    Compute the matrix C2.

    Parameters:
    u1 (float): Computed value u1
    u2 (float): Computed value u2

    Returns:
    np.ndarray: Matrix C2
    """
    F1 = norm.cdf(u1)
    row1 = [F1, 1 - F1, - F1]

    F2 = norm.cdf(u2)
    row2 = [F2, - F2, 1 - F2]

    return np.vstack([row1, row2])

def penalty_function(x, theta1, theta2, CE=1, CT=1):
    theta1, theta2 = np.array(theta1), np.array(theta2)
    cache1, cache2 = theta1.dot(f1(x)), theta2.dot(f2(x))

    p1_, p_1 = norm.cdf(cache1), norm.cdf(cache2)

    return p1_ ** (- CE) * p_1 ** (- CT)

def elemental_bivariate_probit(x, theta1, theta2, rho):
    """
    Compute the elemental information matrix for the bivariate probit model.

    Parameters:
    x (float): Dose value
    theta1 (np.ndarray): Parameter vector for efficacy
    theta2 (np.ndarray): Parameter vector for toxicity
    rho (float): Correlation coefficient

    Returns:
    np.ndarray: Fisher information matrix
    """
    theta1, theta2 = np.array(theta1), np.array(theta2)
    cache1, cache2 = theta1.dot(f1(x)), theta2.dot(f2(x))
    u1, u2 = compute_u1(cache1, cache2, rho), compute_u2(cache1, cache2, rho)

    p1_, p_1 = norm.cdf(cache1), norm.cdf(cache2)
    p11 = multivariate_normal.cdf([cache1, cache2], mean=[0, 0], cov=[[1, rho], [rho, 1]])
    p10 = p1_ - p11
    p01 = p_1 - p11
    p = [p11, p10, p01]
    P_ppt = compute_P_ppt(p)

    C1 = compute_C1(x, cache1, cache2)
    C2 = compute_C2(u1, u2)

    result = C1.dot(C2).dot(np.linalg.inv(P_ppt)).dot(C2.T).dot(C1.T)

    return result

def fisher_info_bivariate_probit(design, theta1, theta2, rho):
    """
    Compute the Fisher information matrix for the bivariate probit model.

    Parameters:
    design (np.ndarray): Given design
    theta1 (np.ndarray): Parameter vector for efficacy
    theta2 (np.ndarray): Parameter vector for toxicity
    rho (float): Correlation coefficient

    Returns:
    np.ndarray: Fisher information matrix
    """
    m = np.zeros((len(theta1) + len(theta2), len(theta1) + len(theta2)))
    d = len(design)
    doses = design[:(d // 2)]
    probs = design[(d // 2):]
    Phi = 0

    for j in range((d // 2)):
        E = elemental_bivariate_probit(doses[j], theta1, theta2, rho)
        phi = penalty_function(doses[j], theta1, theta2)
        m += probs[j] * E
        Phi += probs[j] * phi

    return m, Phi

def D_optim(b, **kwargs):
    """
    Computes the D-Optimal design cost for given sets of inputs.

    Parameters
    ----------
    b : numpy.ndarray
        Sets of inputs, shape: (n_particles, dimensions).

    Returns
    -------
    numpy.ndarray
        Computed cost of size (n_particles,).
    """
    theta1, theta2, rho = kwargs.values()

    n, d = b.shape
    loss = np.zeros(n)

    for i in range(n):
        m = 1e-7 * np.eye(4)
        x = b[i, :(d // 2)]
        p = b[i, (d // 2):]
        p = p / np.sum(p)
        Phi = 0

        for j in range((d // 2)):
            E = elemental_bivariate_probit(x[j], theta1, theta2, rho)
            phi = penalty_function(x[j], theta1, theta2)
            m += p[j] * E
            Phi += p[j] * phi

        D_eff = np.linalg.det(m / Phi)
        loss[i] = np.log(D_eff)
        if p[-1] < 0:
            loss[i] -= 1e200

    return - loss

def D_optim_sensitivity(x, design, params):
    theta1, theta2, rho = params
    M = np.zeros((4, 4))
    d = len(design)

    doses = design[:(d // 2)]
    probs = design[(d // 2):]
    probs = probs / np.sum(probs)

    phi_x = penalty_function(x, theta1, theta2)
    Phi = 0

    for j in range((d // 2)):
        E = elemental_bivariate_probit(doses[j], theta1, theta2, rho)
        Phi += probs[j] * penalty_function(doses[j], theta1, theta2)
        M += probs[j] * E

    mu_x = elemental_bivariate_probit(x, theta1, theta2, rho)

    return 1 / phi_x * np.trace(mu_x.dot(np.linalg.inv(M))) - 4 / Phi

def D_optim_two_stage(b, **kwargs):
    """
    Computes the D-Optimal design cost for given sets of inputs.

    Parameters
    ----------
    b : numpy.ndarray
        Sets of inputs, shape: (n_particles, dimensions).

    Returns
    -------
    numpy.ndarray
        Computed cost of size (n_particles,).
    """
    theta1, theta2, rho, m0, alpha = kwargs.values()
    # m0: Fisher information matrix of the first-stage
    # alpha:  proportion of added second-stage design

    n, d = b.shape
    loss = np.zeros(n)

    for i in range(n):
        m = 1e-7 * np.eye(4)
        x = b[i, :(d // 2)]
        p = b[i, (d // 2):]
        p = p / np.sum(p)
        Phi = 0

        for j in range((d // 2)):
            E = elemental_bivariate_probit(x[j], theta1, theta2, rho)
            phi = penalty_function(x[j], theta1, theta2)
            m += p[j] * E
            Phi += p[j] * phi

        m_two_stage = alpha * m + (1 - alpha) * m0
        D_eff = np.linalg.det(m_two_stage / Phi)
        loss[i] = np.log(D_eff)
        if p[-1] < 0:
            loss[i] -= 1e200

    return - loss

# Streamlit App

# Left column - Introduction
with st.sidebar:
    st.title("Bivariate Probit Model Introduction")
    st.markdown("""
    ### 1. Model Overview
    Let $Y \in \\{0,1\\}$ denote efficacy response and $Z \in \\{0,1\\}$ denote toxicity response with 1 meaning occurrence and 0 meaning no occurrence. The efficacy response corresponds to 'no VTE event' and the toxicity response to 'bleeding'. A possible dose is denoted by $x$. The probabilities of different responses can be defined as:

    $$ p_{yz}(x) = Pr(Y=y, Z=z|x), \quad y,z = 0, 1 $$

    The relationship among these probabilities can be clearly shown as:
    """)

    st.latex(r"""
    p_{11}(x, \theta) = F(\theta_1^T f_1(x), \theta_2^T f_2(x), \rho) = 
    \int_{-\infty}^{\theta_1^T f_1(x)} \int_{-\infty}^{\theta_2^T f_2(x)} 
    \frac{1}{2\pi |\Sigma|^{1/2}} 
    \exp\left( -\frac{1}{2} \mathbf{v}^T \Sigma^{-1} \mathbf{v} \right) dv_1 dv_2
    """)

    st.markdown("""
    where $\\theta = (\\theta_1^T, \\theta_2^T, \\rho)$, $\\theta_1$ and $\\theta_2$ are unknown parameters. The variance-covariance matrix $\\Sigma$ is defined as:

    $$
    \\Sigma = \\begin{pmatrix} 1 & \\rho \\\\ \\rho & 1 \\end{pmatrix}
    $$

    In this model, $\\Sigma$ is treated as known, though similar derivations can be obtained when $\\rho$ is unknown.
    """)

    st.markdown("""
    ### 2. Probabilities and Functions
    The probabilities $p_{1.}(x, \\theta)$ and $p_{.1}(x, \\theta)$ are expressed as the marginal distributions of the bivariate normal distribution:

    $$
    p_{1.}(x, \\theta) = F(\\theta_1^T f_1(x)) \quad and \quad p_{.1}(x, \\theta) = F(\\theta_2^T f_2(x))
    $$

    where

    $$
    F(v) = \\int_{-\infty}^{v} \\frac{1}{\\sqrt{2\\pi}} \\exp\\left(-\\frac{u^2}{2}\\right) du
    $$

    The other probabilities are defined as:

    $$
    p_{10}(x, \\theta) = p_{1.}(x, \\theta) - p_{11}(x, \\theta)
    $$

    $$
    p_{01}(x, \\theta) = p_{.1}(x, \\theta) - p_{11}(x, \\theta)
    $$

    $$
    p_{00}(x, \\theta) = 1 - p_{1.}(x, \\theta) - p_{.1}(x, \\theta) + p_{11}(x, \\theta)
    $$
    """)

    st.markdown("""
    ### 3. Conclusion
    This introduction outlines the core components of a bivariate probit model, focusing on the joint distribution of efficacy and toxicity outcomes. The model uses a probit link function, assuming a bivariate normal distribution with a known covariance structure.
    """)

# Right column - Implementation
# Panel selection

def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

panel_selection = st.radio("Choose a Panel", ["One-stage D-optimal Design", "Two-stage D-optimal Design", "Brief Introduction to Fihser Info Calclation"])
ChangeWidgetFontSize('Choose a Panel', '32px')

# Panel 1: D-optimal Design for Dose-finding
if panel_selection == "One-stage D-optimal Design":
    st.title("One-stage D-optimal Design for Dose-finding")
    st.subheader("that accounts for both efficacy and safety")

    st.text("")

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        theta1_input = st.text_input("Theta1 values (comma-separated)", "-0.9, 1.9")
    with col2:
        theta2_input = st.text_input("Theta2 values (comma-separated)", "-3.98, 3")
    with col3:
        rho_input = st.number_input("Rho value (-1 to 1)", value=0.5)
    col1, col2, col3 = st.columns(3)
    with col1:
        n_iterations = st.number_input("Number of PSO iterations", min_value=20, max_value=1000, step=5, value=90)
    with col2:
        low = st.number_input("Lower bound of design", value=0.2)
    with col3:
        upp = st.number_input("Upper bound of design", value=1.4)

    # Parse inputs
    theta1 = list(map(float, theta1_input.split(',')))
    theta2 = list(map(float, theta2_input.split(',')))
    rho = float(rho_input)

    # Generate random design matrix b for demonstration purposes
    n = 20  # number of particles
    d = 6  # dimension of the problem
    b = np.random.random((n, d))

    # Calculate D-optimal design when the button is pressed
    if st.button("Calculate D-optimal Design"):
        # low, upp = 0.2, 1.4
        bounds = [tuple(np.concatenate([[low] * (d // 2), [0] * (d // 2)])),
                  tuple(np.concatenate([[upp] * (d // 2), [1] * (d // 2)]))]

        options = {'c1': 0.9, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=d, options=options, bounds=bounds)

        # Perform optimization using PSO
        st.text("Performing optimization using PSO...")
        progress_bar = st.progress(0)  # Initialize progress bar
        # n_iterations = 100
        for i in range(n_iterations):
            best_cost, best_pos = optimizer.optimize(D_optim, iters=1, theta1=theta1, theta2=theta2, rho=rho)
            progress_bar.progress((i + 1) / n_iterations)
        best_pos[(d // 2):] /= sum(best_pos[(d // 2):])

        st.text("Optimization finished!")
        best_cost = np.round(best_cost, 3)
        st.write(f"D-optimal Design Cost: {best_cost}")

        designs = np.round(best_pos[:(d//2)], 3)
        weights = np.round(best_pos[(d//2):], 3)
        result = pd.DataFrame(np.transpose([designs, weights]), columns=["Design points", "Design weights"])
        st.write(result)

        st.text("The Sensitivity Plot:")
        x_s = np.linspace(low, upp, 100)
        sens = []
        for x in x_s:
            sens.append(D_optim_sensitivity(x, best_pos, (theta1, theta2, rho)))

        fig, ax = plt.subplots()
        plt.plot(x_s, sens, c="orange")
        plt.hlines(0, xmin=low - 0.05, xmax=upp + 0.05, color="red")
        plt.scatter(best_pos[:(d // 2)], [0] * (d // 2), c="orange")
        # Use st.pyplot to display the figure in Streamlit
        st.pyplot(fig)

elif panel_selection == "Two-stage D-optimal Design":
    st.title("Two-stage D-optimal Design for Dose-finding")
    st.subheader("that accounts for both efficacy and safety")

    st.text("")

    # Inputs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        theta1_input = st.text_input("Theta1 values (comma-separated)", "-0.9, 1.9")
    with col2:
        theta2_input = st.text_input("Theta2 values (comma-separated)", "-3.98, 3")
    with col3:
        rho_input = st.number_input("Rho value (-1 to 1)", value=0.5)
    with col4:
        alpha_input = st.slider("Alpha value (0 to 1)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

    # User input for design points
    design_points_input = st.text_input(f"Enter stage-one design points (comma-separated)", "0.2, 0.5, 0.8, 1.1, 1.4")

    # User input for corresponding design weights
    design_weights_input = st.text_input(f"Enter stage-one design weights (comma-separated)", "0.2, 0.2, 0.2, 0.2, 0.2")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_iterations = st.number_input("Number of PSO iterations", min_value=20, max_value=1000, step=5, value=90)
    with col2:
        low = st.number_input("Lower bound of design", value=0.2)
    with col3:
        upp = st.number_input("Upper bound of design", value=1.4)

    # Parse inputs
    theta1 = list(map(float, theta1_input.split(',')))
    theta2 = list(map(float, theta2_input.split(',')))
    rho = float(rho_input)
    alpha_input = float(alpha_input)
    design_points = list(map(float, design_points_input.split(',')))
    design_weights = list(map(float, design_weights_input.split(',')))
    initial_design = np.append(design_points, design_weights)

    m0, Phi = fisher_info_bivariate_probit(initial_design, theta1, theta2, rho)

    # Generate random design matrix b for demonstration purposes
    n = 20  # number of particles
    d = 6  # dimension of the problem
    b = np.random.random((n, d))

    # Calculate D-optimal design when the button is pressed
    if st.button("Calculate D-optimal Design"):
        # low, upp = 0.2, 1.4
        bounds = [tuple(np.concatenate([[low] * (d // 2), [0] * (d // 2)])),
                  tuple(np.concatenate([[upp] * (d // 2), [1] * (d // 2)]))]

        options = {'c1': 0.9, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=d, options=options, bounds=bounds)

        # Perform optimization using PSO
        st.text("Performing optimization using PSO...")
        progress_bar = st.progress(0)  # Initialize progress bar
        # n_iterations = 100
        for i in range(n_iterations):
            best_cost, best_pos = optimizer.optimize(D_optim_two_stage, iters=1, theta1=theta1, theta2=theta2, rho=rho,
                                                     m0=m0, Phi=Phi)
            progress_bar.progress((i + 1) / n_iterations)
        best_pos[(d // 2):] /= sum(best_pos[(d // 2):])

        st.text("Optimization finished!")
        best_cost = np.round(best_cost, 3)
        st.write(f"D-optimal Design Cost: {best_cost}")

        designs = np.round(best_pos[:(d // 2)], 3)
        weights = np.round(best_pos[(d // 2):], 3)
        result = pd.DataFrame(np.transpose([designs, weights]), columns=["Design points", "Design weights"])
        st.write(result)

        # st.text("The Sensitivity Plot:")
        # x_s = np.linspace(low, upp, 100)
        # sens = []
        # for x in x_s:
        #     sens.append(D_optim_sensitivity(x, best_pos, (theta1, theta2, rho)))
        #
        # fig, ax = plt.subplots()
        # plt.plot(x_s, sens, c="orange")
        # plt.hlines(0, xmin=low - 0.05, xmax=upp + 0.05, color="red")
        # plt.scatter(best_pos[:(d // 2)], [0] * (d // 2), c="orange")
        # # Use st.pyplot to display the figure in Streamlit
        # st.pyplot(fig)

elif panel_selection == "Brief Introduction to Fihser Info Calclation":

    st.title("Brief Introduction to Fisher Info Calculation")

    # Add the reference
    st.markdown("""
    ### References
    - Dragalin, Vladimir, Valerii V. Fedorov, and Yuehui Wu. "Two‐stage design for dose‐finding that accounts for both efficacy and safety." *Statistics in Medicine* 27.25 (2008): 5156-5176.
    """)

    st.text("")

    st.markdown("### Appendix: The Fisher Information Matrix for Bivariate Probit Model")
    st.markdown(r"""
    Let $\theta = (\theta_1, \theta_2) \in \mathbb{R}^m$, where $m = m_1 + m_2$, $m_i = \text{dim}(\theta_i)$, $i = 1, 2$. 
    The log-likelihood function of a single observation $(Y, Z)$ at dose $x$ is
    """)

    st.latex(r"""
    \ell(\theta | Y, Z; x) = YZ \ln p_{11}(x, \theta) + Y(1 - Z) \ln p_{10}(x, \theta) 
    + (1 - Y)Z \ln p_{01}(x, \theta) + (1 - Y)(1 - Z) \ln p_{00}(x, \theta)
    """)

    st.markdown(r"""
    If $\{Y, Z\} = \{Y_i, Z_i\}_{i=1}^N$ are $N$ independent observations at doses $X = \{x_i\}_{i=1}^N$, then the corresponding log-likelihood function is
    """)

    st.latex(r"""
    \ell_N(\theta | Y, Z; X) = \sum_{i=1}^{N} n_i \ell(\theta | Y_i, Z_i; x_i)
    """)

    st.markdown(r"""
    The maximum likelihood estimate (MLE) of $\theta$ is $\hat{\theta}_N = \text{arg max}_\theta \ell_N(\theta | Y, Z; X)$. 
    Asymptotically, $\text{cov}(\hat{\theta}_N) \simeq M_N^{-1}(X, \hat{\theta}_N)$ and the information matrix $M_N(X, \theta)$ is defined as
    """)

    st.latex(r"""
    M_N(X, \theta) = \sum_{i=1}^{N} \mu(x_i, \theta)
    """)

    st.markdown(r"""
    where $\mu(x, \theta) = \text{var}\left[\frac{\partial \ell_N(\theta | Y, Z; X)}{\partial \theta}\right]$ is the information matrix for a single observation at dose $x$:
    """)

    st.latex(r"""
    \mu(x, \theta) = C_1 C_2 (P - pp^T)^{-1} C_2^T C_1^T
    """)

    st.markdown(r"""
    with
    """)

    st.latex(r"""
    C_1 = \begin{pmatrix}
    \psi(\theta_1^T f_1) f_1 & 0 \\
    0 & \psi(\theta_2^T f_2) f_2 
    \end{pmatrix},C_2 = \begin{pmatrix}
    F(u_1) & 1 - F(u_1) & -F(u_1) \\
    F(u_2) & -F(u_2) & 1 - F(u_2)
    \end{pmatrix}
    """)

    st.latex(r"""
    u_1 = \frac{\theta_2^T f_2 - \rho \theta_1^T f_1}{\sqrt{1 - \rho^2}}, 
    \quad u_2 = \frac{\theta_1^T f_1 - \rho \theta_2^T f_2}{\sqrt{1 - \rho^2}}, \quad P = \begin{pmatrix}
    p_{11} & 0 & 0 \\
    0 & p_{10} & 0 \\
    0 & 0 & p_{01}
    \end{pmatrix}
    """)

    st.markdown("and $p = (p_{11}, p_{10}, p_{01})^T$.")

