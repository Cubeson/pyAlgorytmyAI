import numpy as np
import xlsxwriter
import datetime
import shutil
import csv
from mealpy import FloatVar, ArchOA, ASO, CDO, EFO, EO, CoatiOA
#https://github.com/HaaLeo/swarmlib
#import swarmlib
import math

import optimization_functions as of

dimension_count = 2
bench_count = 10
max_epochs = 200
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

algorithm_name = "ArchOA"
model = ArchOA.OriginalArchOA(epoch=max_epochs, pop_size=pop_size ,c1 = 2, c2 = 5, c3 = 2, c4 = 0.5, acc_max = 0.9, acc_min = 0.1)
#model = ASO.OriginalASO(epoch=max_epochs, pop_size=pop_size, alpha = 50, beta = 0.2)
#model = CDO.OriginalCDO(epoch=max_epochs, pop_size=pop_size)
#model = EO.AdaptiveEO(epoch=max_epochs, pop_size=pop_size)
#model = EFO.DevEFO(epoch=max_epochs, pop_size=pop_size, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
#model = CoatiOA.OriginalCoatiOA(epoch=max_epochs, pop_size=pop_size)
class FuncResult:
  def __init__(self, name, data):
    self.name = name
    self.data = data

def benchmark_all(funcs):
  results = []
  for func in funcs:
    fitness_function = func[0]
    func_name = func[1]
    min = func[2]
    max = func[3]
    problem_dict = {
        "bounds": FloatVar(lb=(min,) * dimension_count, ub=(max,) * dimension_count, name="delta"),
        "minmax": "min",
        "obj_func": fitness_function,
        "log_to": None,
        "save_population": True,
    }
    result = benchmark_single(func_name, problem_dict)
    results.append(result)
  create_excel_file(f"{algorithm_name}_wyniki", results)

def create_excel_file(algorithm_name, results):
  wb = xlsxwriter.Workbook(f"exported/{algorithm_name}.xlsx")
  for res in results:
    add_sheet_to_excel_workbook(wb, res.name, res.data)
  wb.close()
  pass

data_names = ["Average","Standard deviation","Average time","Standard deviation time"]
def add_sheet_to_excel_workbook(existing_workbook, sheetName, data):
  sheet = existing_workbook.add_worksheet(sheetName)
  row = 0
  col = 0
  for value in (data):
      sheet.write(row, col, data_names[row])
      sheet.write(row, col + 1, value)
      row += 1

def save_charts(model, algorithm_name, func_name, iteration):
  model.history.save_local_objectives_chart(filename=f"exported/{algorithm_name}/{func_name}/{iteration}/loc")
  model.history.save_local_best_fitness_chart(filename=f"exported/{algorithm_name}/{func_name}/{iteration}/lbfc")
  model.history.save_exploration_exploitation_chart(filename=f"exported/{algorithm_name}/{func_name}/{iteration}/eec")

def save_history(model, algorithm_name, func_name, iteration):
    current_best = model.history.list_current_best
    flattened = [ind.solution.tolist() + [ind.target.fitness] for ind in current_best]
    with open(f"exported/{algorithm_name}/{func_name}/{iteration}/current_best.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerows(flattened)

def benchmark_single(func_name, problem_dict):
  results = []
  executions_times = []
  print(f"Function: {func_name}")
  for i in range(bench_count):
    result = model.solve(problem_dict)
    tstart = datetime.datetime.now()
    tend = datetime.datetime.now()
    executions_times.append((tend-tstart).total_seconds())
    save_charts(model, algorithm_name, func_name, i)
    save_history(model, algorithm_name, func_name, i)


    results.append(result)

  # średnia wyników
  avg = sum(r.target.fitness for r in results)/bench_count

  # odchylenie standardowe wyników
  std = np.std([r.target.fitness for r in results])

  # średnia czasów
  avg_time = sum(executions_times)/bench_count
  #avg_time = sum(model.history.list_epoch_time)/bench_count

  #odchylenie standardowe czasów
  std_time = np.std(executions_times)
  #std_time = np.std(model.history.list_epoch_time)

  return FuncResult(func_name, [avg, std, avg_time, std_time])

benchmark_all(funcs)
#shutil.make_archive(f"exported/{algorithm_name}", 'zip', f"exported/{algorithm_name}")

#print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
#print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

