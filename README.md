
# Adaptive Optimal Designs for Bivariate Probit Models

## Bivariate Probit Model D-Optimal Design

### Introduction
This repository implements a **D-optimal design** strategy for dose-finding that accounts for both efficacy and safety in a **bivariate probit model**. The project uses **Particle Swarm Optimization (PSO)** to find the optimal design points and weights. The approach covers both **one-stage** and **two-stage** designs, with a focus on the Fisher information matrix for optimizing the design.

#### Key Features
- **Bivariate Probit Model** for efficacy and toxicity.
- **D-optimal design**: Efficiently selects dose levels to maximize the Fisher information.
- **Two-stage design**: Optimizes the design considering an additional stage for refinement.
- **User input**: Accepts user-defined initial designs for greater flexibility.

### Prerequisites
To run this project, you need to have the following libraries installed:
- `streamlit`
- `numpy`
- `scipy`
- `matplotlib`
- `pyswarms`
- `pandas`

Install the necessary packages by running:
```bash
pip install streamlit numpy scipy matplotlib pyswarms pandas
```

## How to Run

```bash
git clone https://github.com/yourusername/optimal-bivariate-design.git

cd optimal-bivariate-design

streamlit run app.py
```

Finally, Open your browser to `localhost:8501` to access the app.
