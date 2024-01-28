from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from dataclasses import dataclass, field

from amm import amm
from params import params


N_pools = params['N_pools']
Rx0 = params['Rx0']
Ry0 = params['Ry0']
phi = params['phi']
x_0 = params['x_0']
alpha = params['alpha']
q = params['q']
zeta = params['zeta']
# batch_size = params['batch_size']
batch_size = 200
kappa = params['kappa']
sigma = params['sigma']
p = params['p']
T = params['T']
seed = params['seed']

pools = amm(Rx0, Ry0, phi)


@dataclass
class Theta:
    
    xs_0: NDArray = field(default_factory = lambda: np.random.beta(1, 5, N_pools)) # distribution of initial pool weights, centered around 1/6
    cvar: float = -1.0


    def get_fitness(self) -> float:
        if self.cvar == -1.0:
            self.cvar = self.evaluate()
        return self.cvar
    
    
    def evaluate(self) -> float:
        if np.sum(self.xs_0) > x_0 or np.any(self.xs_0 < 0):
            return float('inf')

        l = pools.swap_and_mint(self.xs_0)
        end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
        pools.simulate(kappa = kappa, p = p, sigma = sigma, T = T, batch_size = batch_size)

        x_T = np.zeros(batch_size)
        for k in range(batch_size):
            x_T[k] = np.sum(end_pools[k].burn_and_swap(l))
        
        sum_xs   = np.sum(self.xs_0)
        log_ret  = np.log(x_T) - np.log(sum_xs) # performance / log return
        
        if len(log_ret[log_ret > zeta]) > q * len(log_ret):
            qtl = -np.quantile(log_ret, 1-alpha)
            print("valid solution found", self.xs_0, np.mean(-log_ret[-log_ret>=qtl]))
            return np.mean(-log_ret[-log_ret>=qtl])
    
        return float('inf')
        
    
    def __add__(self, other: Theta | Theta_increment):
        xs_0 = self.xs_0 + other.xs_0
        return Theta(xs_0)


@dataclass
class Theta_increment():
    xs_0: NDArray = field(default_factory = lambda: mutation_rate * np.random.randn(N_pools))

    def __add__(self, other: Theta | Theta_increment):
        xs_0 = self.xs_0 + other.xs_0
        return Theta(xs_0)



def evolutionary_strategy():
    
    # Initialization
    population = np.array([Theta() for _ in range(population_size)])
    # fitness = np.zeros(population_size)
    
    try:
        for generation in tqdm(range(generations)):

            # Selection: Choose the best individuals
            selected_population = sorted(population, key = lambda x: x.get_fitness())[:population_size // 2]

            # Mutation: Generate new individuals based on selected ones
            mutated_population = selected_population + np.array([Theta_increment() for _ in range( (population_size + 1) // 2)])

            # Combine the selected and mutated populations
            population[: population_size // 2] = selected_population
            population[population_size // 2 :] = mutated_population
            
            best_solution = min(population, key = lambda x: x.get_fitness())
            print("Best Solution:", best_solution)

    except KeyboardInterrupt:
        pass

    finally:
        # Choose the best individual after all generations (or after ctrl+c)
        return best_solution


if __name__ == "__main__":
    population_size = 100
    generations = 10
    mutation_rate = 0.1

    best_solution = evolutionary_strategy()

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_solution.get_fitness())

