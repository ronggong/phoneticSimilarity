import numpy as np


def gau_js(pm, pv, qm, qv):
    """
    Jensen-Shannon divergence between two Gaussians.  Also computes JS
    divergence between a single Gaussian pm,pv and a set of Gaussians
    qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    THis divergence may not be in [0, 1]
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverses of diagonal covariances pv, qv
    iqv = 1./qv
    ipv = 1./pv
    # Difference between means pm, qm
    diff = qm - pm
    # KL(p||q)
    kl1 = (0.5 *
           (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)))                     # - N
    # KL(q||p)
    kl2 = (0.5 *
           (np.log(dpv / dqv)            # log |\Sigma_p| / |\Sigma_q|
            + (ipv * qv).sum(axis)          # + tr(\Sigma_p^{-1} * \Sigma_q)
            + (diff * ipv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p)
            - len(pm)))                     # - N
    # JS(p,q)
    return 0.5 * (kl1 + kl2)


def gau_bh(pm, pv, qm, qv):
    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = np.log(pv).sum()
    ldqv = np.log(qv).sum(axis)
    # Log-determinant of pqv
    ldpqv = np.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)
    return dist + norm

def squareBhattacharyya(mu_0, cov_0, mu_1, cov_1):
    """
    Calculate the square BD distance, cov should be diagonal, a value may not in [0, 1]
    :param mu_0: (n, )
    :param cov_0: (n, )
    :param mu_1: (m, )
    :param cov_1: (m, )
    :return:
    """
    # print(mu_0.shape, mu_1.shape)
    # any mu_0 == 0
    if np.all(np.isnan(mu_0)) or np.all(np.isnan(mu_1)):
        return np.nan

    mu_01 = mu_0 - mu_1
    cov_01 = cov_0+cov_1

    left = (mu_01*(1.0/(cov_01/2.0))@mu_01.T)/8.0

    right_numerator = np.prod(cov_01/2.0)
    right_denominator = np.sqrt(np.prod(cov_0))*np.sqrt(np.prod(cov_1))
    right = np.log(right_numerator/right_denominator)/2.0

    return left + right

def BDDistanceMat(list_mu_cov):
    """
    calculate the distance mat between phone pairs, only calculate the lower diagonal
    :param list_mu_cov:
    :return:
    """
    len_mu_cov = len(list_mu_cov)
    distance_mat = np.zeros((len_mu_cov, len_mu_cov))
    for ii in range(0, len_mu_cov-1):
        for jj in range(ii+1, len_mu_cov):
            distance_mat[ii, jj] = squareBhattacharyya(mu_0=list_mu_cov[ii][0], cov_0=list_mu_cov[ii][1],
                                                       mu_1=list_mu_cov[jj][0], cov_1=list_mu_cov[jj][1])
            distance_mat[jj, ii] = distance_mat[ii, jj]
    return distance_mat

if __name__ == '__main__':
    mu_0 = np.array([1,2,3])
    cov_0 = np.array([4,5,6])

    mu_1 = np.array([3, 2, 1])
    cov_1 = np.array([6, 5, 4])

    dis = squareBhattacharyya(mu_0, cov_0, mu_1, cov_1)
    # dis = gau_bh(mu_0, cov_0, mu_1, cov_1)

    print(dis)