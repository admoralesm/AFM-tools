import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Cubic polyelectrolyte compression model function


def cubic_polyelectrolyte_model(x, E, nu, h, R):
    return ((2 * np.pi * E * h * R) / (1 - nu**2)) * x**3

# Function to calculate R-squared


def calculate_r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Function to calculate chi-square


def calculate_chi_square(y_true, y_pred):
    residuals = y_true - y_pred
    return np.sum((residuals**2) / y_pred)


# Parameters
params = {
    'file_path': r'C:\Users\dammd\OneDrive\UC Davis\Research\Cardiac Project\2024\20240227_WT_007.csv',
    'nu': 0.45,
    'h': 4E-9,
    'R': 6.455E-6,
    'initial_guess_E': [1e9],
    'fit_range': (0, 0.55),  # Range for fitting E
    'x_display_range': (-0.01, 0.75),
    'y_display_range': (-0.1e-6, 3.5e-6),  # Original range for Newtons
    'xlabel': r'Relative deformation, $\epsilon$',  # Use epsilon without boldsymbol
    'ylabel': r'Force ($\mu$N)',  # Use micro without boldsymbol
    'title': '',
}

# Read CSV file
try:
    data = pd.read_csv(params['file_path'])
except FileNotFoundError:
    print(f"File not found: {params['file_path']}")
    raise

x_data = data['RelDef'].values  # Extract relative deformation values
y_data = data['Force'].values  # Extract force values in Newtons

# Fit for E using the cubic polyelectrolyte compression model
fit_mask = (x_data >= params['fit_range'][0]) & (
    x_data <= params['fit_range'][1])
params_fit_E, _ = curve_fit(
    lambda x, E: cubic_polyelectrolyte_model(
        x, E, params['nu'], params['h'], params['R']),
    x_data[fit_mask], y_data[fit_mask], p0=params['initial_guess_E']
)
y_fitted_E = cubic_polyelectrolyte_model(
    x_data, params_fit_E[0], params['nu'], params['h'], params['R'])
r_squared_E = calculate_r_squared(y_data[fit_mask], y_fitted_E[fit_mask])
chi_square_E = calculate_chi_square(y_data[fit_mask], y_fitted_E[fit_mask])

# Plotting
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 6))

# Set white background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot all data
ax.scatter(x_data, y_data, color='#89CFF0', label='Data',
           alpha=0.5)  # Plot data points in pastel blue

# Plot the fitted curve within the fitting range
plot_mask = (x_data >= params['fit_range'][0]) & (
    x_data <= params['fit_range'][1])
ax.plot(x_data[plot_mask], y_fitted_E[plot_mask], 'b--',
        label='Fitting', linewidth=3)  # Plot fitted curve

# Customize the graph and axes
ax.set_xlabel(params['xlabel'], fontsize=16,
              fontweight='bold')  # Set x-axis label
ax.set_ylabel(params['ylabel'], fontsize=16,
              fontweight='bold')  # Set y-axis label
ax.set_xlim(params['x_display_range'])  # Set x-axis range
ax.set_ylim(params['y_display_range'])  # Set y-axis range
ax.legend(loc='upper left', fontsize=14)

# Customize ticks
ax.tick_params(axis='x', direction='out', length=8, width=2.5, colors='black')
ax.tick_params(axis='y', direction='out', length=8, width=2.5, colors='black')

# Set bold font for tick labels
ax.set_xticklabels(
    [f'{xtick:.2f}' for xtick in ax.get_xticks()], fontsize=12, fontweight='bold')
ax.set_yticklabels([f'{ytick/1e-6:.1f}' for ytick in ax.get_yticks()],
                   fontsize=12, fontweight='bold')  # y in micro-Newtons

# Bold axis lines and remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)

# Save the plot
plt.savefig('cubic_polyelectrolyte_fit.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print fitted parameters and additional statistics
math_expr = r'$F(x) = \frac{2 \pi E h R}{1 - \nu^2} x^3$'
print(f"Fitted Model: {math_expr}")
print(f"R² (E): {r_squared_E:.4f}")
print(f"Chi-Square (E): {chi_square_E:.4f}")
print(f"E (Elastic Modulus): {params_fit_E[0]/1e6:.2f} MPa")
print(f" - Poisson's ratio (ν): {params['nu']}")
print(f" - Lipid bilayer (h): {params['h']} m")
print(f" - Cell Radius (R): {params['R']} m")
