import cma, botorch, scipy
import matplotlib.pyplot as plt
import numpy as np


# RBF kernel
def rbf_kernel(x, y, l, sigma):
    return sigma**2 * np.exp(- (np.linalg.norm(x - y)**2)/(2 * l**2))


# Expectation improvement

def main():
    print("Imports worked")


    N = 10
    length_scale = 0.1


    #for i in range(N):
        # Update the posterior probability distribution on f using all data
        # Find x_n - maximizing Acq over x where Acq = EI for now using post
        # Observe y_n = f(x_n)

        # A = something
        #c, low =  scipy.linalg.cho_factor(A)
        # b = something
        #sol = scipy.linalg.cho_solve((c, low), b)


        #plt.plot(something)

    # length-scale = 2.0



if __name__ == "__main__":
    main()