import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import optimization_functions as of

from mealpy import FloatVar, ArchOA, ASO

dimension_count = 2
bench_count = 5
max_epochs = 50
pop_size = 30
bound = 5.12

of_f = of.Rastrigin(dimension_count)
problem_dict = {
    "bounds": FloatVar(lb=(-bound,) * dimension_count, ub=(bound,) * dimension_count, name="delta"),
    "minmax": "min",
    "obj_func": of_f,
    "log_to": None,
    "save_population": True,
}
#model = ArchOA.OriginalArchOA(epoch=max_epochs, pop_size=pop_size ,c1 = 2, c2 = 5, c3 = 2, c4 = 0.5, acc_max = 0.9, acc_min = 0.1)
model = ASO.OriginalASO(epoch=max_epochs, pop_size=pop_size, alpha = 50, beta = 0.2)
result = model.solve(problem_dict)
generations = model.history.list_population

#for agents in generations:
#    solutions = [agent.solution for agent in agents]

x = np.linspace(-bound, bound, 200)
y = np.linspace(-bound, bound, 200)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec = np.array([X[i, j], Y[i, j]])
        Z[i, j] = of_f(vec)

fig, ax = plt.subplots()
background = ax.imshow(
    Z, extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower', cmap='viridis', alpha=0.6
)
scat = ax.scatter([], [])

all_fitnesses = [agent.target.fitness for gen in generations for agent in gen]
min_fit = min(all_fitnesses)
max_fit = max(all_fitnesses)

def init():
    ax.set_xlim(-bound, bound)  # Adjust to your data range
    ax.set_ylim(-bound, bound)
    return scat,

def update(frame):
    gen = generations[frame]
    xs = [agent.solution[0] for agent in gen]
    ys = [agent.solution[1] for agent in gen]
    #fitnesses = [agent.target.fitness for agent in gen]
    #print(min(fitnesses))
    #inverted = [-f for f in fitnesses]
    #normed = [(val - (-max_fit)) / ((-min_fit) - (-max_fit)) for val in inverted]

    scat.set_offsets(list(zip(xs, ys)))
    #scat.set_array(normed)
    ax.set_title(f"Generation {frame}")
    return scat,

ani = animation.FuncAnimation(
    fig, update, frames=len(generations),
    init_func=init, blit=False, interval=500
)

#plt.colorbar(scat, label='-Fitness (lower is better)')
plt.colorbar(background)
plt.show()