import numpy as np
import pandas as pd
from scipy.stats import norm
import argparse

parser = argparse.ArgumentParser(description="Tiger expriment")
parser.add_argument('--model', type=int, help="model number, positive integer > 9", required=True)
parser.add_argument('--length', type=int, help="test length, positive integer > 9", required=True)
parser.add_argument('--redlight', type=int, help="redlight index, start from 1, integer", required=True)
parser.add_argument('--noise_threshold', type=float, help="noise level between 0.0 and 1.0, default 0.5", required=True)
parser.add_argument('--output_csv_path', type=str, help="path to save csv output", required=True)

args = parser.parse_args()

assert args.model in list(range(1,11))
assert args.length >= 10
assert args.redlight >= 1
assert args.noise_threshold >= 0.0
assert args.noise_threshold <= 1.0

index_redlight = args.redlight - 1
N = args.length

true_red_light = np.zeros(N)
true_red_light[index_redlight:] = 1.0


if args.model == 1:
  print("model: linear")
  y = np.linspace(0.0, 1.0, num = N)
  
elif args.model == 2:
  print("model: Quadratic")
  y = np.linspace(0.0, 1.0, num = N) ** 2
  
elif args.model == 3:
  print("model: Square Root")
  y = np.linspace(0.0, 1.0, num = N) ** 0.5
  
elif args.model == 4:
  print("model: Sin")
  y = np.sin( np.linspace(0.0, np.pi / 2, num=N) )
  
elif args.model == 5:
  print("model: Circle")
  x = np.linspace(0.0, 1.0, num = N)
  y = 1.0 - ( (1.0 - x) ** 0.5 )
  
elif args.model == 6:
  print("model: Cumulative Sum")
  x = np.random.uniform(low=0.0, high=1.0, size=N)
  v = (x>args.noise_threshold).astype(np.float64)
  d = v / (N / 2)
  y = np.cumsum(d)
  y[y>1.0] = 1.0
  
elif args.model == 7:
  print("model: Smart")
  y = true_red_light
  
elif args.model == 8:
  print("model: Perturbation")
  x = np.random.uniform(low=0.0, high=1.0, size=N)
  v = (x>args.noise_threshold).astype(np.float64)
  d = v / (N / 2)
  y = np.cumsum(d)
  y[y>1.0] = 1.0
  y[index_redlight:] = 1.0
  
elif args.model == 9:
  print("model: Gaussian Bernoulli")
  G = norm.cdf(np.arange(N), loc=index_redlight, scale=N/4)
  x = np.zeros(N)
  y = np.zeros(N)
  for k in range(N):
    p = G[k]
    x[k] = np.random.binomial(1, p)
  z = np.cumsum(x)
  y = z / (2 * z[index_redlight])
  y[y>1.0] = 1.0
  
elif args.model == 10:
  print("model: Gaussian")
  x = np.arange(N)
  y = norm.cdf(x, loc=index_redlight, scale=N/4)
else:
  raise ValueError()


columns = ["index", "true_red_light" ,"prediction_world_changed"]

output_array = np.zeros((N,3))
output_array[:,0] = np.arange(1,N+1)
output_array[:,1] = true_red_light
output_array[:,2] = y


df = pd.DataFrame(output_array, columns = columns)
df = df.astype({'index': 'int32'})
df = df.astype({'true_red_light': 'int32'})
df.to_csv(args.output_csv_path, index=False)

print("\nEnd\n\n")
