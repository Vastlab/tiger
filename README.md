# tiger
tiger

# Requirment

python 3.8

numpy

pandas


# Running

You can run by entering following command


```
 python3.8 test_1.py --model 1 --length 100 --redlight 40 --noise_threshold 0.5 --output_csv_path output_1.csv
 ```
 
 `--model`: is an integer between 1 to 10.
 
 ```
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
```

`--length`:lenght of test. integer number equal or greater than 10.

`--redlight`: first time redlight is on. positive integer number. value 1 means first index is also on. you can subtract from 1 to get index in python array.

`--noise_threshold`: float number between 0.0 and 1.0. default is 0.5. It is used in `Cumulative Sum` and `Perturbation` models. It does not affect other models.

`--output_csv_path`: path for output csv file. 





 
