import math
import secrets


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """
        Cryptographically secure uniform sample..
    """
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53) # in [0, 1)
    return a + (b - a) * u




def exponentialdist(lam: float) -> float:
    """
        Exponentially distributed random sample using inverse transform sampling
    """

    #check if lambda is positive
    if lam <= 0:
        raise ValueError("lambda must be greater than 0")

    #generate U in (0, 1)
    #avoid U = 0 because log(0) is undefined
    u = uniform()

    #loop until u is not zero
    while u == 0:
        u = uniform()

    #apply the inverse CDF of the exponential distribution
    x = -(1 / lam) * math.log(u)

    return x


def poissiondist(lam: float) -> int:
    """
        Poisson distributed random sample using inverse transform sampling
    """

    #check if lambda is positive
    if lam <= 0:
        raise ValueError("lambda must be greater than 0")

    #generate one secure uniform random value
    u = uniform()

    #initialize k 
    k = 0

    #poisson probability for k = 0
    probability = math.exp(-lam)

    #initialize cumulative probability
    cumulative_probability = probability

    #find the first k for which F(k) >= u
    while u > cumulative_probability:
        k += 1

        #recursively calculate the next Poisson probability
        probability = probability * lam / k

        #add it to the CDF
        cumulative_probability += probability

    return k