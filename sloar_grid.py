import numpy as np
import matplotlib.pyplot as plt

SOLAR_MAX = 4
BATTERY_MAX = 3
DEMAND = 6


NUM_BEES = 10
MAX_ITER = 50
LIMIT = 10 


def initialize():
    solutions = []
    for _ in range(NUM_BEES):
        P_solar = np.random.uniform(0, SOLAR_MAX)
        P_battery = np.random.uniform(0, BATTERY_MAX)
        P_grid = max(0, DEMAND - (P_solar + P_battery))
        solutions.append([P_solar, P_battery, P_grid])
    return np.array(solutions)


def cost_function(sol):
    P_solar, P_battery, P_grid = sol
    
   
    penalty = abs((P_solar + P_battery + P_grid) - DEMAND) * 100
    
    cost = (2 * P_grid) + (0.5 * P_battery) + penalty
    return cost


def generate_neighbor(sol, solutions):
    k = np.random.randint(len(solutions))
    phi = np.random.uniform(-1, 1, size=3)
    
    new_sol = sol + phi * (sol - solutions[k])
    
   
    new_sol[0] = np.clip(new_sol[0], 0, SOLAR_MAX)
    new_sol[1] = np.clip(new_sol[1], 0, BATTERY_MAX)
    
   
    new_sol[2] = max(0, DEMAND - (new_sol[0] + new_sol[1]))
    
    return new_sol

def ABC():
    solutions = initialize()
    fitness = np.array([cost_function(sol) for sol in solutions])
    trial = np.zeros(NUM_BEES)
    convergence_curve = [] 
    for iter in range(MAX_ITER):
        
       
        for i in range(NUM_BEES):
            new_sol = generate_neighbor(solutions[i], solutions)
            new_cost = cost_function(new_sol)
            
            if new_cost < fitness[i]:
                solutions[i] = new_sol
                fitness[i] = new_cost
                trial[i] = 0
            else:
                trial[i] += 1
        
        
        prob = (1 / (1 + fitness))
        prob = prob / np.sum(prob)
        
        for i in range(NUM_BEES):
            if np.random.rand() < prob[i]:
                new_sol = generate_neighbor(solutions[i], solutions)
                new_cost = cost_function(new_sol)
                
                if new_cost < fitness[i]:
                    solutions[i] = new_sol
                    fitness[i] = new_cost
                    trial[i] = 0
                else:
                    trial[i] += 1
        
        
        for i in range(NUM_BEES):
            if trial[i] > LIMIT:
                solutions[i] = initialize()[0]
                fitness[i] = cost_function(solutions[i])
                trial[i] = 0
        best_cost = np.min(fitness)
        convergence_curve.append(best_cost)
    
    best_idx = np.argmin(fitness)
    return solutions[best_idx], fitness[best_idx], convergence_curve
    


best_solution, best_cost, convergence_curve = ABC()
print("Optimal Power Distribution:")
print(f"Solar:   {best_solution[0]:.2f} kW")
print(f"Battery: {best_solution[1]:.2f} kW")
print(f"Grid:    {best_solution[2]:.2f} kW")
print(f"Cost:    {best_cost:.2f}")
plt.plot(convergence_curve)
plt.xlabel("Iterations")
plt.ylabel("Best Cost")
plt.title("ABC Convergence Curve")
plt.grid()
plt.show()