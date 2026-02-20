import numpy as np

def load_instance(filepath):
    
    raw_data = np.loadtxt(filepath)
    cylinders = np.zeros((len(raw_data), 4), dtype=np.float64)
    cylinders[:, 0] = raw_data[:, 0] # x
    cylinders[:, 1] = raw_data[:, 1] # y
    cylinders[:, 2] = raw_data[:, 2] # masse
    cylinders[:, 3] = 2 * raw_data[:, 2] - 1 # points

    return cylinders

if __name__ == "__main__":
    cylinders = load_instance("data/donnees-map1.txt")
    print(cylinders)