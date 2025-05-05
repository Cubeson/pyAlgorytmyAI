import numpy as np
import xlsxwriter
import time
import shutil

from matplotlib import pyplot as plt, animation
from mealpy import FloatVar, ArchOA, ASO, CDO, EFO, EO, CoatiOA
#https://github.com/HaaLeo/swarmlib
#import swarmlib
import math
import optimization_functions as of

dimension_count = 2
bench_count = 20
max_epochs = 100
pop_size = 20

# [func, name, min, max]
funcs = [
        [of.Sphere(dimension_count, degree = 2)                    , 'Sphere',-5.12,5.12         ],
        [of.Ackley(dimension_count)                                , 'Ackley',-32.768,32.768         ],
        [of.Griewank(dimension_count)                              , 'Griewank',-600,600       ],
        [of.Rosenbrock(dimension_count)                            , 'Rosenbrock',-5,10     ],
        [of.Fletcher(dimension_count, seed = None)                 , 'Fletcher',-100,100       ],
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

algorithm_name = "CoatiOA"
#model = ArchOA.OriginalArchOA(epoch=max_epochs, pop_size=pop_size ,c1 = 2, c2 = 5, c3 = 2, c4 = 0.5, acc_max = 0.9, acc_min = 0.1)
#model = ASO.OriginalASO(epoch=max_epochs, pop_size=pop_size, alpha = 50, beta = 0.2)
#model = CDO.OriginalCDO(epoch=max_epochs, pop_size=pop_size)
#model = EO.AdaptiveEO(epoch=max_epochs, pop_size=pop_size)
#model = EFO.DevEFO(epoch=max_epochs, pop_size=pop_size, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
model = CoatiOA.OriginalCoatiOA(epoch=max_epochs, pop_size=pop_size)
class FuncResult:
  def __init__(self, name, data):
    self.name = name
    self.data = data

class SheetRow:
    def __init__(self, benchmark:int, positions:list[float], fitness:float, time:float):
        self.benchmark = benchmark
        self.positions = positions
        self.fitness = fitness
        self.time = time

ani_writer = animation.PillowWriter(fps=10,
                                 metadata=dict(artist='Jakub_Lewandowski'),
                                 bitrate=1800)

def save_animation(generations, func, func_name, gen, lbound, ubound):
    x = np.linspace(lbound, ubound, 200)
    y = np.linspace(lbound, ubound, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = np.array([X[i, j], Y[i, j]])
            Z[i, j] = func(vec)
    fig, ax = plt.subplots()
    background = ax.imshow(
        Z, extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower', cmap='viridis', alpha=0.6
    )
    scat = ax.scatter([], [])

    def init():
        ax.set_xlim(lbound, ubound)  # Adjust to your data range
        ax.set_ylim(lbound, ubound)
        return scat,

    def update(frame):
        gen = generations[frame]
        xs = [agent.solution[0] for agent in gen]
        ys = [agent.solution[1] for agent in gen]
        # fitnesses = [agent.target.fitness for agent in gen]
        # print(min(fitnesses))
        # inverted = [-f for f in fitnesses]
        # normed = [(val - (-max_fit)) / ((-min_fit) - (-max_fit)) for val in inverted]

        scat.set_offsets(list(zip(xs, ys)))
        # scat.set_array(normed)
        ax.set_title(f"Generation {frame}")
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(generations),
        init_func=init, blit=False, interval=500
    )
    ani.save(f"exported/{algorithm_name}/{func_name}/{gen}/animation.gif", writer=ani_writer)
    plt.close()

def benchmark_all(funcs):
  wb = xlsxwriter.Workbook(f"exported/{algorithm_name}_wyniki.xlsx")
  for func in funcs:
    fitness_function = func[0]
    func_name = func[1]
    min_val = func[2]
    max_val = func[3]
    problem_dict = {
        "bounds": FloatVar(lb=(min_val,) * dimension_count, ub=(max_val,) * dimension_count, name="delta"),
        "minmax": "min",
        "obj_func": fitness_function,
        "log_to": None,
        "save_population": True,
    }
    result = benchmark_single(func_name, problem_dict, min_val, max_val)
    add_sheet_to_excel_workbook(wb, func_name, result)
  wb.close()

data_names = ["Average","Standard deviation","Average time","Standard deviation time"]
def add_sheet_to_excel_workbook(existing_workbook, sheet_name, sheet_rows:list[SheetRow]):
  sheet = existing_workbook.add_worksheet(sheet_name)
  row = 0
  col = 0
  # A1  B1 C1 D1  E1   F1 G1  H1      I1       J1
  # gen x1 x2 fit time |  avg std_dev avg_time std_dev_time
  sheet.write(row, col + 0, "Benchmark")
  sheet.write(row, col + 1, "x1")
  sheet.write(row, col + 2, "x2")
  sheet.write(row, col + 3, "Best Fitness")
  sheet.write(row, col + 4, "Time")
  sheet.write(row, col + 5, "")
  sheet.write(row, col + 6, "Average Fitness")
  sheet.write(row, col + 7, "StDev Fitness")
  sheet.write(row, col + 8, "Average Time")
  sheet.write(row, col + 9, "StDev Time")

  row_count = len(sheet_rows)
  row = 1
  # G avg
  sheet.write_formula(row, col + 6, f"=AVERAGE(D2:D{row_count+1})")
  # H std_dev
  sheet.write_formula(row, col + 7, f"=STDEV(D2:D{row_count + 1})")
  # I avg time
  sheet.write_formula(row, col + 8, f"=AVERAGE(E2:E{row_count + 1})")
  # J std_dev time
  sheet.write_formula(row, col + 9, f"=STDEV(E2:E{row_count + 1})")

  for sheet_row in sheet_rows:
        #A gen
        sheet.write(row, col+0, sheet_row.benchmark)
        #B x1
        sheet.write(row,col+1, sheet_row.positions[0])
        #C x2
        sheet.write(row,col+2, sheet_row.positions[1])
        #D fit
        sheet.write(row, col + 3, sheet_row.fitness)
        #E time
        sheet.write(row, col + 4, sheet_row.time)
        #F empty

        row += 1

def save_charts(model, algorithm_name, func_name, iteration):
  model.history.save_local_objectives_chart(filename=f"exported/{algorithm_name}/{func_name}/{iteration}/local_objectives_chart")
  model.history.save_local_best_fitness_chart(filename=f"exported/{algorithm_name}/{func_name}/{iteration}/local_best_fitness_chart")
  model.history.save_exploration_exploitation_chart(filename=f"exported/{algorithm_name}/{func_name}/{iteration}/exploration_exploitation_chart")

def benchmark_single(func_name, problem_dict, min_val, max_val):
  sheet_rows = []
  print(f"Function: {func_name}")
  #gen = 1
  for i in range(bench_count):
    print(f"Benchmark: {i+1}")
    tstart = time.time()
    result = model.solve(problem_dict)
    tend = time.time()

    time_elapsed = tend-tstart
    positions = result.solution.tolist()
    fitness = result.target.fitness
    sheet_row = SheetRow(i+1,positions,fitness,time_elapsed)
    sheet_rows.append(sheet_row)
    save_charts(model, algorithm_name, func_name, i+1)
    save_animation(model.history.list_population,problem_dict["obj_func"],func_name,i+1,min_val,max_val)
    #gen = gen + 1
  return sheet_rows

benchmark_all(funcs)
#shutil.make_archive(f"exported/{algorithm_name}", 'zip', f"exported/{algorithm_name}")