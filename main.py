import numpy as np

import regression as r

dataset = r.get_dataset('bodyfat.csv')

#print(dataset.shape)
r.print_stats(dataset, 1)
MSE = r.regression(dataset, cols=[2, 3], betas=[0, 0, 0])
#print(MSE)
MSE = r.regression(dataset, cols=[2, 3, 4], betas=[0, -1.1, -.2, 3])
#print(MSE)
gradDesc = r.gradient_descent(dataset, cols=[2, 3], betas=[0, 0, 0])
#print(gradDesc)
gradDesc = r.gradient_descent(dataset, cols=[1, 4], betas=[0, 0, 0])
#print(gradDesc)
r.iterate_gradient(dataset, cols=[1, 8], betas=[400, -400, 300], T=10, eta=1e-4)
r.iterate_gradient(dataset, cols=[1, 4], betas=[400, -400, 10], T=5, eta=1e-4)
#print(r.compute_betas(dataset, cols=[1, 2]))
#print(r.compute_betas(dataset, cols=[1, 2, 8, 9]))
result = r.predict(dataset, cols=[1, 2], features=[1.0708, 23])
#print(result)
syntheticData = r.synthetic_datasets(np.array([0, 2]), np.array([0, 1]), np.array([[4]]), 1)
#print(syntheticData)
r.plot_mse()
