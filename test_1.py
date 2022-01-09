'''
Requirment:

python 3.8
numpy
pandas

============

Running:

You can run by entering following command

 python3.8 test_1.py --model 1 --length 100 --redlight 40 --noise_threshold 0.5 --output_csv_path output_1.csv


============


--model: is an integer between 1 to 10.

model 1: linear, ignoring true red light
model 2: Quadratic, ignoring true red light
model 3: Square Root, ignoring true red light
model 4: Sin, ignoring true red light
model 5: Circle, ignoring true red light
model 6: Cumulative Sum, ignoring true red light
model 7: Perturbation, similar to 6 but jump to 1 aftter red light is on
model 8: Smart, return true red light
model 9: Gaussian, similar to 8 but follow Gassuian CDF, it is 0.5 on first red light
model 10: Gaussian Bernoulli, has corrolatation with red light, no causation. may go 0.5 before or after fist red light.

--length:lenght of test. integer number equal or greater than 10.

--redlight: first time redlight is on. positive integer number. value 1 means first index is also on. you can subtract from 1 to get index in python array.

--noise_threshold: float number between 0.0 and 1.0. default is 0.5. It is used in Cumulative Sum and Perturbation models. It does not affect other models.

--noise_min:  floats  number between 0.0 and 1.0. default is 0.0.  It is used in Cumulative Sum and Perturbation models. It does not affect other models.

--noise_max:  floats  number between 0.0 and 1.0. default is .3.0.  It is used in Cumulative Sum and Perturbation models. It does not affect other models.

--output_csv_path: path for output csv file.

'''

import numpy as np
import pandas as pd
from scipy.stats import norm
import argparse
import pdb

parser = argparse.ArgumentParser(description="Tiger expriment")
parser.add_argument('--model', type=int, help="model number, positive integer > 9", required=True)
parser.add_argument('--length', type=int, help="test length, positive integer > 9", required=True)
parser.add_argument('--redlight', type=int, help="redlight index, start from 1, integer", required=True)
parser.add_argument('--noise_threshold', type=float, help="noise level between 0.0 and 1.0, default 0.5", required=True)
parser.add_argument('--noise_min', type=float,default=0.0, help="Minimum backgorund noise level between 0.0 and 1.0, default 0.0")
parser.add_argument('--noise_max', type=float, default=0.3, help="Max background noise level between 0.0 and 1.0, default 0.2", required=False)
parser.add_argument('--signal', type=float, default=0.2, help="Signal level above  background noise (between 0 and 1), default 0.2", required=False)
parser.add_argument('--output_csv_path', type=str, help="path to save csv output", required=True)

args = parser.parse_args()

assert args.model in list(range(1,15))
assert args.length >= 10
assert args.redlight >= 1
assert args.noise_threshold >= 0.0
assert args.noise_threshold <= 1.0
assert args.noise_min >= 0.0
assert args.noise_min <= 1.0
assert args.noise_max >= 0.0
assert args.noise_max <= 1.0

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
  print("model: Perturbation")
  x = np.random.uniform(low=0.0, high=1.0, size=N)
  v = (x>args.noise_threshold).astype(np.float64)
  d = v / (N / 2)
  y = np.cumsum(d)
  y[y>1.0] = 1.0
  y[index_redlight:] = 1.0
  
elif args.model == 8:
  print("model: Smart")
  y = true_red_light
  
elif args.model == 9:
  print("model: Gaussian")
  x = np.arange(N)
  y = norm.cdf(x, loc=indePx_redlight, scale=N/4)
  
elif args.model == 10:
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

elif args.model == 11:
  print("model: Pure Cumulative Sum on Noisy signal")
  x = np.random.uniform(low=args.noise_min, high=args.noise_max, size=N)
  x[index_redlight:] += args.signal
#  d = x / (N / 2)   don't want to scale.. better to have scaling  baked into the noise_max so its more understandable
  d = x 
  y = np.cumsum(d)
  y[y>1.0] = 1.0

elif args.model == 12:
  print("model: Random guess model 12 of location  ")
  y = np.zeros(N)
  loc  =  int(N*.1 + np.random.normal(N/4, N/4))
  loc = min(loc, N-1)
  y[loc:] = 1.0

elif args.model == 13:
  print("model: Random guess model 13 of location  ")
  y = np.zeros(N)
  loc  =  int(N*.1 + np.random.normal(N/8, N/8))
  loc = min(loc, N-1)
  y[loc:] = 1.0


elif args.model == 14:
  print("model: Random guess model 13 of location  ")
  y = np.zeros(N)
  loc  =  int(N*.1 + np.random.uniform(N/8, N/3))
  loc = min(loc, N-1)
  y[loc:] = 1.0




  
  
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
