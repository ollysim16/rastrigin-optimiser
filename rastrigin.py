import math
import random
import numpy as np
from tqdm import tqdm
import time

REPETITIONS = 50
INITIAL_TEMP = 0.95
DOMAIN = 5.12
WIDTH = 2 * DOMAIN
DELTA0 = 0.2
GD_DELTA0 = 0.1
COOLING_COEFFICIENT = 0.99999
MIN_DIMENSION = 2
MAX_DIMENSION = 5

def f(point):
    output = 0
    for item in point:
        output += 10 + item**2 - (10 * math.cos(2 * math.pi * item))
    return output


def f_dash(point):
    output = []
    for item in point:
        output.append(2*item + (20 * math.pi * math.sin(2 * math.pi * item)))
    return np.array(output)


def generate_move(point, step_size):
    dimensions = len(point)
    # generate D decimals using the standard normal distribution
    # (probability density is dependent entirely on distance from origin)
    vector = np.random.normal(0, 1, dimensions)
    vector = vector / np.linalg.norm(vector)
    vector *= step_size
    u = random.random()
    # scale to adjust for clumping near centre
    vector *= u**(1/dimensions)
    new_point = point + vector
    # reflect if OOB
    for dimension in range(dimensions):
        coordinate = new_point[dimension]
        if coordinate > WIDTH * 2:
            coordinate %= WIDTH * 2
        elif coordinate < -WIDTH * 2:
            coordinate %= -WIDTH * 2
        while abs(coordinate) > DOMAIN:
            if coordinate > DOMAIN:
                coordinate = 2*DOMAIN - coordinate
            elif coordinate < -DOMAIN:
                coordinate = -2*DOMAIN - coordinate
        new_point[dimension] = coordinate
    return new_point


def initialise_temp(pos, delta):
    # generate a list of bad possible moves from starting pos and set initial temperature based on average
    fpos = f(pos)
    move_sample = [generate_move(pos, delta) for _ in range(1000)]
    bad_move_deltas = [f(x) - fpos for x in move_sample if f(x) > fpos]
    avg_bad_move_delta = sum(bad_move_deltas)/len(bad_move_deltas) if bad_move_deltas else 0
    return -avg_bad_move_delta / math.log(INITIAL_TEMP)

def main():
    counts = []
    average_sas = []
    average_gds = []
    average_results = []
    average_sa_runtimes = []
    average_gd_runtimes = []
    average_runtimes = []

    for D in range(MIN_DIMENSION, MAX_DIMENSION+1):
        c = 0
        sas = []
        gds = []
        results = []
        sa_runtimes = []
        gd_runtimes = []
        runtimes = []

        for _ in tqdm(range(REPETITIONS)):
            start = time.perf_counter()
            delta = DELTA0
            pos = np.random.uniform(-DOMAIN, DOMAIN, D)
            t = initialise_temp(pos, delta)
            lowest_val = f(pos)
            lowest_point = pos.copy()
            proposed_moves = 0
            accepted_moves = 0
            acceptance_rate = 1
            sa_iterations = 0
            gd_iterations = 0

            # simulated annealing:
            while acceptance_rate > 0.01:
                sa_iterations += 1
                move = generate_move(pos, delta)
                delta_E = f(move) - f(pos)
                if delta_E < 0:
                    pos = move
                    accepted_moves += 1
                else:
                    if random.random() < math.exp(-delta_E/t):
                        pos = move
                        accepted_moves += 1
                proposed_moves += 1
                if f(pos) < lowest_val:
                    lowest_val = f(pos)
                    lowest_point = pos.copy()
                if proposed_moves % 1000 == 0:
                    acceptance_rate = accepted_moves/proposed_moves
                    accepted_moves = proposed_moves = 0

                t *= COOLING_COEFFICIENT

            sa_end = time.perf_counter()
            # gradient descent:
            pos = lowest_point
            gd_delta = GD_DELTA0
            while gd_delta > 0.000001:
                gd_iterations += 1
                minus_gradient = -f_dash(pos)
                minus_gradient /= np.linalg.norm(minus_gradient)
                proposed_move = minus_gradient * gd_delta
                proposed_point = pos + proposed_move
                if f(proposed_point) < f(pos):
                    pos = proposed_point.copy()
                else:
                    gd_delta /= 2.5

            lowest_point = pos.copy()
            lowest_val = f(lowest_point)

            end = time.perf_counter()

            sa_runtime = sa_end - start
            gd_runtime = end - sa_end
            runtime = end - start

            sas.append(sa_iterations)
            gds.append(gd_iterations)
            results.append(lowest_val)
            sa_runtimes.append(sa_runtime)
            gd_runtimes.append(gd_runtime)
            runtimes.append(runtime)

            if lowest_val < 0.01:
                c += 1

        counts.append(c)
        average_sas.append(sum(sas)/len(sas))
        average_gds.append(sum(gds)/len(gds))
        average_results.append(sum(results)/len(results))
        average_sa_runtimes.append(sum(sa_runtimes)/len(sa_runtimes))
        average_gd_runtimes.append(sum(gd_runtimes) / len(gd_runtimes))
        average_runtimes.append(sum(runtimes)/len(runtimes))

    return (counts,
            average_sas,
            average_gds,
            average_results,
            average_sa_runtimes,
            average_gd_runtimes,
            average_runtimes)


count, sa, gd, res, sa_rts, gd_rts, rts = main()
success_rates = list(map(lambda x: int(x/REPETITIONS * 100), count))

for i in range(MAX_DIMENSION - MIN_DIMENSION + 1):
    print(f"Dimensions: {i+MIN_DIMENSION}\n"
          f"Simulated annealing average: {int(sa[i])} iterations over {int(sa_rts[i])}s\n"
          f"Gradient descent average: {int(gd[i])} iterations over {round(gd_rts[i], 3)}s\n"
          f"Runtime average: {int(rts[i])}s\n"
          f"Average objective function value: {round(res[i], 3)}\n"
          f"Success rate: {success_rates[i]}%\n")
