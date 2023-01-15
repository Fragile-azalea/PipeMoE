# -*- coding: utf-8 -*-
from typing import Tuple, Union
from scipy.optimize import minimize
import math
import numpy as np

ALPHA_D_100GBIB_64 = 7.83e-4
BETA_D_100GBIB_64 = 3.84e-10
ALPHA_D_100GBIB_32 = 2.09e-4
BETA_D_100GBIB_32 = 3.55e-10
ALPHA_D_100GBIB_16 = 5.70e-4
BETA_D_100GBIB_16 = 2.96e-10
ALPHA_GEMM_RTX2080TI = 1.5e-4
BETA_GEMM_RTX2080TI = 3.73e-14

NBYTES_PER_ELEMENT = 4  # FP32

GPU_PARAMS = {
    'rtx2080ti': (ALPHA_GEMM_RTX2080TI, BETA_GEMM_RTX2080TI)
}

NETWORK_PARAMS = {
    '64-100GbIB':   (ALPHA_D_100GBIB_64, BETA_D_100GBIB_64),
    '32-100GbIB':   (ALPHA_D_100GBIB_32, BETA_D_100GBIB_32),
    '16-100GbIB':   (ALPHA_D_100GBIB_16, BETA_D_100GBIB_16),
}


def get_network_params(network: str,
                       P: Union[int, str]) -> Tuple[float, float]:
    alpha, beta = NETWORK_PARAMS.get(str(P)+'-'+network)
    return alpha, beta


def predict_gemm_time(m: int,
                      n: int,
                      k: int,
                      gpu: str = 'rtx2080ti') -> float:
    r"""
    This is a function to predict gemm time.

    :param m: The shape[0] of first matrix
    :param n: The shape[1] of first matrix
    :param k: The shape[1] of second matrix
    :param gpu: The type of GPU, defaults to 'rtx2080ti'
    :return: predicted time
    """
    x = m * n * k
    assert gpu in GPU_PARAMS, 'We donnot support this GPU: %s current' % gpu
    alpha, beta = GPU_PARAMS.get(gpu)
    t = alpha + beta * x * NBYTES_PER_ELEMENT
    return t


def predict_alltoall_time(x: int,
                          P: Union[int, str],
                          network: str = '100GbIB') -> float:
    r"""
    This is a function to predict alltoall time.

    :param x: # of tokens per GPU
    :param P: # of GPUs for expert parallesim
    :param network: The config of Network, defaults to '100GbIB'
    :return: predicted time
    """
    alpha, beta = get_network_params(network, P)
    return alpha + beta * x * NBYTES_PER_ELEMENT


def solve_k(B: int,
            L: int,
            M: int,
            H: int,
            P: int,
            experts_per_token: int,
            capacity_factor: float,
            gpu: str,
            network: str) -> int:
    r"""
    This is a function to get the best pipeline degree.

    :param B: local mini-batch size
    :param L: sequence length, i.e., # of tokens per training sample
    :param M: dimension of feature map (i.e., embedding dimension or model_dim)
    :param H: size of hidden layer in expert
    :param P: # of GPUs for expert parallesim
    :param experts_per_token: # of experts for each token
    :param capacity_factor: the capacity factor of model
    :param gpu: GPU model, e.g., rtx2080ti
    :param network: network config, e.g., 100GbIB
    :return: the best pipeline degree
    """

    num_tokens_pre_gpu = experts_per_token * int(capacity_factor * B * L)
    len_of_alltoall = num_tokens_pre_gpu * M
    len_of_expert_gemm = num_tokens_pre_gpu * M * H

    n_e = len_of_expert_gemm * NBYTES_PER_ELEMENT
    n_d = len_of_alltoall * NBYTES_PER_ELEMENT

    alpha_a, beta_a = get_network_params(network, P)
    alpha_gemm, beta_gemm = GPU_PARAMS.get(gpu)
    alpha_e = 2 * alpha_gemm
    beta_e = 2 * beta_gemm

    t_e = alpha_e + beta_e * n_e
    t_d = alpha_a + beta_a * n_d

    # No pipeline
    t1 = 2 * t_d + t_e
    r1 = 1

    r2 = max(2, math.ceil((beta_e*n_e - beta_a*n_d)/(alpha_a-alpha_e)))
    t2 = 2 * r2 * alpha_a + 2 * n_d * beta_a
    eps = 1e-10

    def _solve_f3():
        def fun(x): return 2*alpha_a+beta_e*n_e+2*beta_a*n_d/x+alpha_e*x
        cons = ({'type': 'ineq', 'fun': lambda x: beta_e*n_e/x+alpha_e-alpha_a-beta_a*n_d/x-eps},
                {'type': 'ineq', 'fun': lambda x: x * alpha_e + beta_e *
                    n_e - (2*(x-1)*alpha_a+2*(x-1)*beta_a*n_d/x)-eps},
                {'type': 'ineq', 'fun': lambda x: x-2},
                )
        x0 = np.array((2.0))
        res = minimize(fun, x0, method='SLSQP', constraints=cons, tol=1e-11)
        x = int(res.x+0.5)
        minimizer = fun(x)
        return x, minimizer

    def _solve_f4():
        def fun(x): return 2*alpha_a*x+2*n_d*beta_a
        cons = ({'type': 'ineq', 'fun': lambda x: beta_e*n_e/x+alpha_e-alpha_a-beta_a*n_d/x-eps},
                {'type': 'ineq', 'fun': lambda x: 2 *
                    (x-1)*alpha_a+2*(x-1)*beta_a*n_d/x-x*alpha_e-beta_e*n_e-eps},
                {'type': 'ineq', 'fun': lambda x: x-2},
                )
        x0 = np.array((2.0))
        res = minimize(fun, x0, method='SLSQP', constraints=cons,
                       tol=1e-11, options={'maxiter': 100})
        x = int(res.x+0.5)
        minimizer = fun(x)
        return x, minimizer

    r3, t3 = _solve_f3()
    r4, t4 = _solve_f4()

    if min(t2, t3, t4) >= t1:
        return r1
    else:
        l = [t2, t3, t4]
        rs = [r2, r3, r4]
        i = np.argmin(l)
        return rs[i]


if __name__ == '__main__':
    print(solve_k(8, 1024, 4096, 4096, 16, 2, 1.0, 'rtx2080ti', '100GbIB'))
