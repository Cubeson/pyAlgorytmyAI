import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import optimization_functions as of

from mealpy import FloatVar, ArchOA, ASO

dimension_count = 2
bench_count = 5
max_epochs = 50
pop_size = 30
funcs = [
        [of.Sphere(dimension_count, degree = 2)                    , 'Sphere',-5.12,5.12         ],
        [of.Ackley(dimension_count)                                , 'Ackley',-32.768,32.768         ],
        [of.Griewank(dimension_count)                              , 'Griewank',-600,600       ],
        [of.Rosenbrock(dimension_count)                            , 'Rosenbrock',-2.048,2.048     ],
        [of.Fletcher(dimension_count, seed = None)                 , 'Fletcher',-3,3       ],
        [of.Penalty2(dimension_count, a=5, k=100, m=4)             , 'Penalty2',-50,50       ],
        [of.Quartic(dimension_count)                               , 'Quartic',-1.28,1.28        ],
        [of.Rastrigin(dimension_count)                             , 'Rastrigin',-5.12,5.12      ],
        [of.SchwefelDouble(dimension_count)                        , 'SchwefelDouble',-500,500 ],
        [of.Weierstrass(dimension_count, a = 0.5, b = 3, kmax = 20), 'Weierstrass',-0.5,0.5    ],
        [of.Stairs(dimension_count)                                , 'Stairs',-5,5         ],
        [of.Abs(dimension_count)                                   , 'Abs',-100,100            ],
        [of.Michalewicz(m = 10)                                    , 'Michalewicz',0,math.pi    ],
        [of.Scheffer(dimension_count)                              , 'Scheffer',-100,100       ],
        [of.Eggholder(dimension_count)                             , 'Eggholder',-512,512      ],
]
idx = 2
of_f = funcs[idx][0]
lbound = funcs[idx][2]
ubound = funcs[idx][3]
problem_dict = {
    "bounds": FloatVar(lb=(lbound,) * dimension_count, ub=(ubound,) * dimension_count, name="delta"),
    "minmax": "min",
    "obj_func": of_f,
    "log_to": None,
    "save_population": True,
}
#model = ArchOA.OriginalArchOA(epoch=max_epochs, pop_size=pop_size ,c1 = 2, c2 = 5, c3 = 2, c4 = 0.5, acc_max = 0.9, acc_min = 0.1)
model = ASO.OriginalASO(epoch=max_epochs, pop_size=pop_size, alpha = 50, beta = 0.2)
#result = model.solve(problem_dict)
generations = model.history.list_population

#for agents in generations:
#    solutions = [agent.solution for agent in agents]

x = np.linspace(lbound, ubound, 200)
y = np.linspace(lbound, ubound, 200)
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

#all_fitnesses = [agent.target.fitness for gen in generations for agent in gen]
#min_fit = min(all_fitnesses)
#max_fit = max(all_fitnesses)

def init():
    ax.set_xlim(lbound, ubound)  # Adjust to your data range
    ax.set_ylim(lbound, ubound)
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
ani_writer = animation.PillowWriter(fps=10,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save('anim.gif', writer = ani_writer)
#plt.show()