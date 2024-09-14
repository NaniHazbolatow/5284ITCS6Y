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
    k4 = step_size * f(
        *[current_state[i] + k3 for i in range(len(current_state))]
    )

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
        return self.beta * S * I - I * ((self.gamma - self.birth_rate)  + I * (1 - self.mortality_probability))
    
    def dRdt(self, I, R):
        return self.gamma * I - self.birth_rate * R
    
    def update_step(self, dt=0.01):
        """
        Perform step update using the Runge–Kutta method, including infection-induced mortality.
        """
        self.S += RK4(self.dSdt, (self.S, self.I), step_size=dt)
        self.I += RK4(self.dIdt, (self.S, self.I), step_size=dt)
        self.R += RK4(self.dRdt, (self.I, self.R), step_size=dt)


sir_model = mortalitySIR(beta=3, gamma=1, birth_rate=0.02, mortality_probability = 0.3, I0=0.01)
data = sir_model.numerical_integration(t=20, dt=0.01)
plt.plot(data[:, 0], data[:, 1], label="Susceptible")
plt.plot(data[:, 0], data[:, 2], label="Infected")
plt.plot(data[:, 0], data[:, 3], label="Recovered")
plt.legend()
plt.show()
