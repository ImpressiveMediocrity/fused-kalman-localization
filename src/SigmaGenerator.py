import numpy as np

class SigmaGenerator:
  def __init__(self, num_states, alpha, beta, kappa):
    self.num_states = num_states
    self.alpha = alpha
    self.beta = beta
    self.kappa = kappa

  def sigma_points(self, state, covariance):
    sigma_points = np.zeros((self.num_states, 2 * self.num_states + 1))
    #an array of num states by 2 num states + 1
    mean_weights = np.zeros(2 * self.num_states + 1)
    #an array of num states by 2 num states + 1
    cov_weights = np.zeros(2 * self.num_states + 1)

    sigma_points[:,0] = state
    # these are the points we are currently at in each "dimension" and the area we are estimating
    mean_weights[0] = (self.alpha**2 * self.kappa - self.num_states) / (self.alpha**2 * self.kappa)
    #think this is 2d in https://www.mathworks.com/help/control/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html#bvgiw03
    cov_weights[0] = mean_weights[0] - self.alpha**2 + self.beta

    A = np.linalg.cholesky(covariance)
    #cholesky decomp of covariace matrix so its lower triangular

    remaining_weights = 1 / (2 * self.alpha**2 * self.kappa)


    for i in range(self.num_states):
      sigma_points[:,1 + i] = state + self.alpha * np.sqrt(self.kappa) * A[:,i]
      sigma_points[:,1 + self.num_states + i] = state - self.alpha * np.sqrt(self.kappa) * A[:,i]
      #the "left" and "right" sigma point for each state
    

      mean_weights[1 + i:: self.num_states] = remaining_weights
      cov_weights[1 + i:: self.num_states] = remaining_weights
      # for these M = 0, I guess this is faster

    return sigma_points, mean_weights, cov_weights
