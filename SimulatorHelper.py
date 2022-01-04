import numpy as np
import matplotlib.pyplot as plt

# formula constants
LOGISTIC_GROWTH_TIME_STEPS = 8000  # m
LOGISTIC_GROWTH_STEEPNESS = 0.005  # k
HUMP_PROBABILITY = 0.3  # H_s
DECAY_PROBABILITY = 0.004  # E_s
TRANSMISSION_DELAY_LAMBDA = 0.7
EXPONENTIAL_DECAY_LAMBDA = 0.01
ESTABLISHMENT_HUMP = 100
HUMP_K = 1
HUMP_P = 0.2


def calc_neutral_fitness(hosts_num):
    """
    randomly choose parent from previous generation
    :return: chosen parent ID
    """
    return np.full(hosts_num, 1)


def calc_null_establishment_vector(bacteria_species_num):
    """
    calculate the microbes' establishment probability according to a null establishment probability,
    meaning all are equal.
    :return: the establishment probability vector (by order of arrival to the host)
    """
    return np.ones(bacteria_species_num)


def scaled_parabola_function(val):
    parabola_a = 1 / (4 * HUMP_P)
    parabola_b = -ESTABLISHMENT_HUMP / (2 * HUMP_P)
    parabola_c = (ESTABLISHMENT_HUMP ** 2) / ((4 * HUMP_P) + HUMP_K)
    x = np.arange(2 * ESTABLISHMENT_HUMP)
    return (1 / (1 + HUMP_PROBABILITY)) * (HUMP_PROBABILITY + (
            ((-1 * (parabola_a * (val ** 2) + parabola_b * val + parabola_c)) -
             min(-1 * (parabola_a * (x ** 2) + parabola_b * x + parabola_c))) /
            max(((-1 * (parabola_a * (x ** 2) + parabola_b * x + parabola_c)) -
                 min(-1 * (parabola_a * (x ** 2) + parabola_b * x + parabola_c))))))


def scaled_parabola_function2(vec):
    parabola_a = -1 / (4 * HUMP_P)
    parabola_b = 1 / (2 * HUMP_P)
    parabola_c = -(1 ** 2) / ((4 * HUMP_P) + HUMP_K)
    print(parabola_c, parabola_b, parabola_a)
    scaler = 1 / (1 + HUMP_PROBABILITY)
    print(scaler)
    vec = parabola_a * ((vec-ESTABLISHMENT_HUMP) ** 2) + parabola_b * (vec-ESTABLISHMENT_HUMP) + parabola_c
    norm_vec = (vec - min(vec)) / (max(vec) - min(vec))
    return scaler * (HUMP_PROBABILITY + norm_vec)


def calc_hump_establishment_function_vector(bacteria_species_num):
    """
    calculate the microbes' establishment probability according to a "hump" establishment probability function
    :return: the establishment probability vector (by order of arrival to the host)
    """
    vec = np.ones(bacteria_species_num) * (1 / (1 + HUMP_PROBABILITY)) * HUMP_PROBABILITY
    if bacteria_species_num >= 2 * ESTABLISHMENT_HUMP:
        vec[: 2 * ESTABLISHMENT_HUMP] = \
            scaled_parabola_function2(np.arange(1, (2 * ESTABLISHMENT_HUMP) + 1, 1))
    else:
        vec[: bacteria_species_num] = \
            scaled_parabola_function(np.arange(1, bacteria_species_num + 1, 1))
    return vec


def calc_exponential_decay_establishment_vector(bacteria_species_num):
    """
    calculate the microbes' establishment probability according to an "exponential decay" establishment probability.
    :return: the establishment probability vector (by order of arrival to the host)
    """
    return (DECAY_PROBABILITY + np.exp(-EXPONENTIAL_DECAY_LAMBDA * np.arange(bacteria_species_num))) / \
           (1 + DECAY_PROBABILITY)


def calc_waiting_times_between_transmission_events(establishment_type, bacteria_species_num, is_pre_generated_template):
    """
    calculates the waiting times between transmission events according to the desired establishment function
    :return: vector representing the times passed between each microbe transmission event.
    """
    if establishment_type == "hump":
        establishment_vec = calc_hump_establishment_function_vector(bacteria_species_num)
    elif establishment_type == "null":
        establishment_vec = calc_null_establishment_vector(bacteria_species_num)
    elif establishment_type == "decay":
        establishment_vec = calc_exponential_decay_establishment_vector(bacteria_species_num)
    else:
        assert 0, 'unimplemented ESTABLISHMENT_TYPE {}'.format(establishment_type)
    if is_pre_generated_template:
        return (1 / establishment_vec) * np.random.exponential(scale=1/TRANSMISSION_DELAY_LAMBDA,
                                                               size=len(establishment_vec)), establishment_vec
    return 1 / (establishment_vec * TRANSMISSION_DELAY_LAMBDA), establishment_vec


def logistic_growth_function(x, host_carrying_capacity):
    return host_carrying_capacity / (1 + np.exp(
        -LOGISTIC_GROWTH_STEEPNESS * (x - (LOGISTIC_GROWTH_TIME_STEPS / 2))))


def draw_distribution_times_between_transmission_events(file_name):
    time_steps = np.arange(20)
    plt.plot(5, TRANSMISSION_DELAY_LAMBDA * np.exp(-TRANSMISSION_DELAY_LAMBDA * 5), "ro")
    plt.plot(range(time_steps.size), TRANSMISSION_DELAY_LAMBDA * np.exp(-TRANSMISSION_DELAY_LAMBDA * time_steps),
             "c-")
    plt.title("Distribution of waiting times between\ntransmission events", fontsize=18)
    plt.xlabel("Time step", fontsize=18)
    plt.ylabel("Probability", fontsize=18)
    plt.show()


def draw_logistic_growth_function(file_name, host_carrying_capacity):
    time_steps = np.arange(LOGISTIC_GROWTH_TIME_STEPS)
    plt.plot(time_steps, logistic_growth_function(time_steps, host_carrying_capacity), "c-")
    plt.title("Growth function following successful transmission\nand establishment", fontsize=18)
    plt.xlabel("Time step",  fontsize=18)
    plt.ylabel("species' population size",  fontsize=18)
    plt.show()


def draw_establishment_function(establishment_vec, file_name, bacteria_species_num):
    plt.plot(range(bacteria_species_num), establishment_vec, "c-")
    plt.title("Probability of establishment\ndepends on order of arrival", fontsize=18)
    plt.xlabel("# previously established microbial species", fontsize=18)
    plt.ylabel("Probability of establishment", fontsize=18)
    plt.show()


def draw_microbiome_structure(microbiome_structure):
    only_positive_microbiome_structure = microbiome_structure[np.where(microbiome_structure > 0)]
    plt.bar(np.arange(len(only_positive_microbiome_structure)), only_positive_microbiome_structure / (10 ** 6),
            color="c")
    plt.title("Microbiome configurational composition", fontsize=18)
    plt.xlabel("Species' arrival order", fontsize=18)
    plt.ylabel("Microbiome species' population size ($10^6$)", fontsize=16)
    plt.show()
