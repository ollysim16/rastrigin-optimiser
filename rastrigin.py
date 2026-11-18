import math
import random
import numpy as np
from tqdm import tqdm
import time

REPETITIONS = 100
INITIAL_TEMP = 0.95
DOMAIN = 5.12
WIDTH = 2 * DOMAIN
DIMENSIONS = 4
DELTA0 = 0.2
GD_DELTA0 = 0.1
COOLING_COEFFICIENT = 0.99999

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
    # generate DIMENSIONS decimals using the standard normal distribution
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
    c = 0
    sas = []
    gds = []
    results = []
    runtimes = []

    for _ in tqdm(range(REPETITIONS)):
        start = time.perf_counter()
        delta = DELTA0
        pos = np.random.uniform(-DOMAIN, DOMAIN, DIMENSIONS)
        t = initialise_temp(pos, delta)
        lowest_val = f(pos)
        lowest_point = pos.copy()
        proposed_moves = 0
        accepted_moves = 0
        acceptance_rate = 1
        sa_iterations = 0
        gd_iterations = 0

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
        runtime = end - start

        sas.append(sa_iterations)
        gds.append(gd_iterations)
        results.append(lowest_val)
        runtimes.append(runtime)

        if lowest_val < 0.01:
            c += 1

    return (c,
            sum(sas)/len(sas),
            sum(gds)/len(gds),
            sum(results)/len(results),
            sum(runtimes)/len(runtimes)
            )

count, sa, gd, res, rts = main()
print((f"{DIMENSIONS} stochastic optimising runs performed on the {DIMENSIONS}-dimensional Rastrigin function\n"
       f"For the simulated annealing step, there were on average {int(sa)} iterations.\n"
       f"For the gradient descent step, there were on average {int(gd)} iterations.\n"
       f"The average objective function value (target 0) was {round(res, 3)}.\n"
       f"The average runtime was {int(rts)} seconds.\n"
       f"Success rate: {int(count/REPETITIONS * 100)}%"))
