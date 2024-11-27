import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Define the file path (you can change it as needed)
path = '/home/s1989816/Desktop/MAC-MIGS/PhD project/FEM-FCT-PDECO/from_eddie/'
folder = 'advection_Gaussian_drift_c_0.8_T1_beta0.01_tol0.01_job46948349/'
filename = 'advection_Gaussian_drift_c_0.8_T1_beta0.01_tol0.01/gaussian_T1_beta0.01_c.csv'

# Load the CSV into a NumPy array
ck = np.loadtxt(path + folder + filename, delimiter=',')

# Initialize lists for max, min, and mean
max_ck = []
min_ck = []
mean_ck = []

# Define number of steps and nodes
nodes = 6561
num_steps = int(len(ck) / nodes)

# Calculate max, min, and mean for each step
for i in range(num_steps):
    start = i * nodes
    end = (i + 1) * nodes
    max_ck.append(np.amax(ck[start:end]))
    min_ck.append(np.amin(ck[start:end]))
    mean_ck.append(np.mean(ck[start:end]))

# Calculate the means of the max, min, and mean results
mean_max_ck = np.mean(max_ck)
mean_min_ck = np.mean(min_ck)
mean_mean_ck = np.mean(mean_ck)

print(f"Mean of maximums: {mean_max_ck}")
print(f"Mean of minimums: {mean_min_ck}")
print(f"Mean of means: {mean_mean_ck}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(num_steps), max_ck, label='Max Values', color='red', marker='o')
plt.plot(range(num_steps), min_ck, label='Min Values', color='blue', marker='o')
plt.plot(range(num_steps), mean_ck, label='Mean Values', color='green', marker='o')

# Adding labels and legend
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Tracking Max, Min, and Mean over Time Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()