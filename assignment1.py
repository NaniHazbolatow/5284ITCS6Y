import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def RK4(f, current_state: tuple, step_size: float = 0.01) -> float:
    """Performs Runga-Kunta for a single step

    Args:
        f (function): The differential equation as function , either dSdt or dIdt.
        current_state (tuple): A tuple (S, I) representing the current state of the system.
        step_size (float, optional): Step size for Runga-Kunta. Defaults to 0.01.

    Returns:
        float : Updated value of the differential equation.
    """
    k1 = step_size * f(*current_state)

    k2 = step_size * f(
        *[current_state[i] + k1 * 0.5 for i in range(len(current_state))]
    )
    k3 = step_size * f(
        *[current_state[i] + k2 * 0.5 for i in range(len(current_state))]
    )
    k4 = step_size * f(*[current_state[i] + k3 for i in range(len(current_state))])

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


class baseSIR:
    def __init__(self, beta: float, gamma: float, I0: float):
        """
        Initialize base SIR model with model parameters and initial conditions
        """
        self.beta = beta
        self.gamma = gamma

        self.S = 1 - I0
        self.I = I0
        self.R = 0

    def dSdt(self, S, I) -> float:
        """
        Differential equation for susceptible population.
        """
        return -self.beta * S * I

    def dIdt(self, S, I) -> float:
        """
        Differential equation for infected population.
        """
        return self.beta * S * I - self.gamma * I

    def update_step(self, dt=0.01):
        """
        Perform step update using the Runge–Kutta method.
        """
        self.S += RK4(self.dSdt, (self.S, self.I), step_size=dt)
        self.I += RK4(self.dIdt, (self.S, self.I), step_size=dt)
        self.R = 1 - self.S - self.I

    def numerical_integration(self, t: int, dt: float = 0.01):
        """Numerical Integration of the SIR model over time t using RK4 with step size dt.

        Args:
            t (int): Total time.
            dt (float, optional): Step size for RK4. Defaults to 0.01.

        Returns:
            Matrix of t, S, I, R

        """
        times = np.arange(0, t + dt, dt)
        S_values = [self.S]
        I_values = [self.I]
        R_values = [self.R]

        for _ in times[1:]:
            self.update_step(dt)
            S_values.append(self.S)
            I_values.append(self.I)
            R_values.append(self.R)

        return np.column_stack((times, S_values, I_values, R_values))


class demographySIR(baseSIR):
    def __init__(self, beta: float, gamma: float, I0: float, birth_rate: float):
        """
        Initialize SIR model with demography parameters
        """
        super().__init__(beta, gamma, I0)
        self.birth_rate = birth_rate

    def dSdt(self, S, I):
        """
        Differential equation for susceptible population including demography.
        """
        return self.birth_rate - self.beta * S * I - self.birth_rate * S

    def dIdt(self, S, I):
        """
        Differential equation for infected population including demography.
        """
        return self.beta * S * I - I * (self.gamma - self.birth_rate)


class mortalitySIR(demographySIR, baseSIR):
    def __init__(
        self,
        beta: float,
        gamma: float,
        I0: float,
        birth_rate: float,
        mortality_probability: float,
    ):
        super().__init__(beta, gamma, I0, birth_rate)
        self.mortality_probability = mortality_probability

    def dSdt(self, S, I):
        """
        Differential equation for susceptible population including demography.
        """
        return self.birth_rate - self.beta * S * I - self.birth_rate * S

    def dIdt(self, S, I):
        """
        Differential equation for infected population including demography.
        """
        return self.beta * S * I - I * (
            (self.gamma + self.birth_rate) + I * (1 - self.mortality_probability)
        )

    def dRdt(self, I, R):
        return self.gamma * I - self.birth_rate * R

    def update_step(self, dt=0.01):
        """
        Perform step update using the Runge–Kutta method, including infection-induced mortality.
        """
        self.S += RK4(self.dSdt, (self.S, self.I), step_size=dt)
        self.I += RK4(self.dIdt, (self.S, self.I), step_size=dt)
        self.R += RK4(self.dRdt, (self.I, self.R), step_size=dt)


class vaccinationSIR(baseSIR):
    def __init__(self, beta: float, gamma: float, I0: float, vaccination_rate: float):
        """
        Initialize SIR model with vaccination parameters
        """
        super().__init__(beta, gamma, I0)
        self.vaccination_rate = vaccination_rate

    def dSdt(self, S, I) -> float:
        """
        Differential equation for susceptible population.
        """
        return -self.beta * S * I - self.vaccination_rate * S


class flatVaccinationSIR(baseSIR):
    def __init__(self, beta: float, gamma: float, I0: float, vaccination_rate: float):
        """
        Initialize SIR model with vaccination parameters
        """
        super().__init__(beta, gamma, I0)
        self.vaccination_rate = vaccination_rate

    def dSdt(self, S, I) -> float:
        """
        Differential equation for susceptible population.
        """
        return -self.beta * S * I - self.vaccination_rate * S   

class thresholdVaccinationSIR(baseSIR):
    def __init__(self, beta: float, gamma: float, I0: float, vaccination_rate: float):
        super().__init__(beta, gamma, I0)
        self.vaccination_rate = vaccination_rate

    def effective_reproduction_number(self, S):
        """
        Calculate the effective reproduction number Rt at the current time.
        """
        R0 = self.beta / self.gamma
        Rt = R0 * (S / 1.0)
        return Rt
    
    def dSdt(self, S, I) -> float:
        """
        Differential equation for susceptible population with threshold-based vaccination.
        Vaccination is applied only when Rt exceeds the threshold.
        """
        Rt = self.effective_reproduction_number(S)
        
        # Apply vaccination only if Rt >= threshold
        if Rt >= 1:
            vaccination = self.vaccination_rate * S
        else:
            vaccination = 0
        
        return -self.beta * S * I - vaccination


# TO-DO: Write Solver








sir_model = thresholdVaccinationSIR(beta = 1/3, gamma = 1/6, I0 = 0.01, vaccination_rate = 0.01)
data = sir_model.numerical_integration(t=300, dt=0.01)
plt.plot(data[:, 0], data[:, 1], label="Susceptible")
plt.plot(data[:, 0], data[:, 2], label="Infected")
plt.plot(data[:, 0], data[:, 3], label="Recovered")
plt.legend()
plt.show()






################################################ In progress
def plot_single_phase_diagram(
    model_class, beta, gamma, I0_values, title=None, subplot_index=None, **model_kwargs
):
    if subplot_index is not None:
        plt.subplot(2, 3, subplot_index)
    else:
        plt.figure(figsize=(10, 8))  # Adjust size as needed

    # Plot a light red diagonal line from (1, 0) to (0, 1)
    plt.plot([1, 0], [0, 1], "r-", linewidth=1, alpha=0.7, label="Diagonal")

    # Loop through the different I0 values for the same beta, gamma pair
    for I0 in I0_values:
        model_instance = model_class(beta=beta, gamma=gamma, I0=I0, **model_kwargs)
        data = model_instance.numerical_integration(t=120, dt=0.01)
        plt.plot(
            data[:, 1], data[:, 2], color="lightblue", linewidth=1.5, alpha=0.7
        )  # Use light blue for all lines

    # Set axis limits, labels, and grid for each subplot or single plot
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Susceptible (S)")
    plt.ylabel("Infected (I)")
    plt.title(title, fontsize=10)  # Adjust font size of the subplot title
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)  # Softer grid lines
    plt.gca().set_facecolor("whitesmoke")  # Softer background color

    if subplot_index is None:
        plt.show()  # Only call show() if it's a single plot


def plot_all_phase_diagrams(model_class, **model_kwargs):
    # Generate beta and gamma values
    beta_values = np.linspace(0.25, 1, 5)
    gamma_values = np.linspace(0.25, 1, 5)
    I0_values = np.linspace(0.05, 1, 10)

    # Define epidemic and extinction combinations
    epidemic_combinations = [
        (beta, gamma)
        for beta in beta_values
        for gamma in gamma_values
        if beta / gamma > 1
    ]
    extinction_combinations = [
        (beta, gamma)
        for beta in beta_values
        for gamma in gamma_values
        if beta / gamma < 1
    ]

    # Create a figure with 6 subplots (2 rows and 3 columns)
    plt.figure(figsize=(18, 12))

    # Plot the extinction phase diagrams (R0 < 1) with different I0 values (top row)
    # Pick the first 3 pairs
    for i, (beta, gamma) in enumerate(extinction_combinations[:3]):
        plot_single_phase_diagram(
            model_class,
            beta,
            gamma,
            I0_values,
            f"Extinction\nβ = {beta:.2f}, γ = {gamma:.2f}",
            i + 1,
            **model_kwargs,
        )

    # Plot the epidemic phase diagrams (R0 > 1) with different I0 values (bottom row)
    # Pick the first 3 pairs
    for i, (beta, gamma) in enumerate(epidemic_combinations[:3]):
        plot_single_phase_diagram(
            model_class,
            beta,
            gamma,
            I0_values,
            f"Epidemic\nβ = {beta:.2f}, γ = {gamma:.2f}",
            i + 4,
            **model_kwargs,
        )

    # Add a main title for the figure
    plt.suptitle("Phase Diagrams for Different β and γ Values", fontsize=16)

    # Adjust layout to make room for the main title and prevent overlapping
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)

    # Show the plots
    plt.show()


# Call the function to generate the plots
plot_single_phase_diagram(
    demographySIR,
    beta=1 / 6,
    gamma=1 / 3,
    I0_values=np.linspace(0.05, 1, 10),
    birth_rate=0.02,
)


sir_model = demographySIR(beta=1 / 6, gamma=1 / 3, birth_rate=1 / 60, I0=0.01)
data = sir_model.numerical_integration(t=300, dt=0.01)
plt.plot(data[:, 0], data[:, 1], label="Susceptible")
plt.plot(data[:, 0], data[:, 2], label="Infected")
plt.plot(data[:, 0], data[:, 3], label="Recovered")
plt.legend()
plt.show()
