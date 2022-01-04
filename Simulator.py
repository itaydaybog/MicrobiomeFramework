import os
import random
import shutil
import numpy as np
import SimulatorHelper as Sh

#########################
# simulation parameters #
#########################

###########################################################################################
# as described in the paper, for time efficiency we've pre-generated microbiome           #
# configurations to be randomly selected during the simulation.                           #
# To use this functionality set IS_PRE_GENERATED_TEMPLATE = False.                        #
# to randomly generate blank composition on the fly use IS_PRE_GENERATED_TEMPLATE = True. #
###########################################################################################
IS_PRE_GENERATED_TEMPLATE = True

# transmission parameters
PEERS_TRANSMISSION = 0  # T_h
PARENT_TRANSMISSION = 1  # T_v
ENV_TRANSMISSION = 0  # coefficient for introduction of random microbes into the system (for further exploration)
TRANSMISSION_NAME = "vertical"  # for naming of save files

# microbiome parameters
MICROBIOME_CLASS = "Vertebrates"  # Vertebrates, Insects  (Only for IS_PRE_GENERATED_TEMPLATE = True)
BACTERIA_SPECIES_NUM = 2000  # B
GLOBAL_CARRYING_CAPACITY = 10**8  # C_g (only relevant for IS_PRE_GENERATED_TEMPLATE == False)
SPECIE_CARRYING_CAPACITY = 10**7  # C_s (only relevant for IS_PRE_GENERATED_TEMPLATE == False)

# host parameters
HOSTS_NUM = 50  # N
COALESCENCE_THRESHOLD = 2  # AC

# selection parameters
SELECTION_TYPE = 'step_fitness'  # neutral_fitness, decay_fitness, step_fitness
FITNESS_DECAY = 0.7  # 𝜆_3 (when decay_fitness) (preferable - 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.005)
RELEVANT_STEP_BACTERIA_NUM = 30  # 𝜆_3 (when step_fitness)
MAX_CONTRIBUTION = 50  # C_max
MIN_CONTRIBUTION = 0.005  # C_min
MIN_FITNESS = 1
ESTABLISHMENT_TYPE = 'hump'  # null, hump, decay (Only for IS_PRE_GENERATED_TEMPLATE == False)

# misc
REPEATS = 100
np.set_printoptions(suppress=True)
FILE_TO_SAVE = 'results_new/%s/%s/%s_%s/' % (TRANSMISSION_NAME, MICROBIOME_CLASS, SELECTION_TYPE, str(FITNESS_DECAY))


###################
# simulation code #
###################

class Simulator:
    """
    Class running the microbiome-host co-evolution model
    """
    def __init__(self):
        """
        inits model parameters.
        """
        self.fitness_vec = np.zeros(HOSTS_NUM)
        self.bacteria_id_matrix = np.array([np.arange(BACTERIA_SPECIES_NUM)] * HOSTS_NUM, order='K')
        self.bacteria_contribution = None
        if IS_PRE_GENERATED_TEMPLATE:
            self.microbiome_structure_pool = np.load("mb_structures/insects_mbs.npz")["microbiome_structures"] if MICROBIOME_CLASS == "Insects" else np.load("mb_structures/vertebrates_mbs.npz")["microbiome_structures"]

    def simulate(self, repeat):
        """
        runs the full simulation.
        """
        # setup simulation's data location and buffers
        os.makedirs(FILE_TO_SAVE, exist_ok=True)
        shutil.copy("Simulator.py", FILE_TO_SAVE + "/Simulator.py")
        self.simulate_single_reapet(repeat)

    def simulate_single_reapet(self, repeat):
        self.bacteria_contribution = Simulator.generate_bacteria_contribution()
        curr_repeat_full_holobiome_population_each_gen, curr_repeat_parent_distributions_each_gen, curr_repeat_full_lineages_each_gen, curr_repeat_fitness_each_gen = [], [], [], []
        environment_vec = np.ones(BACTERIA_SPECIES_NUM)
        holobiome_population, transmission_matrix = self.calc_first_generation()
        # run generations of repeat
        generation = 0
        while True:  # run until lineage coalescence
            # setup generation's arguments and buffers
            self.fitness_vec = self.calc_fitness_vec(holobiome_population)
            peers = np.sum(holobiome_population, axis=0)
            next_holobiome_population = np.empty((HOSTS_NUM, BACTERIA_SPECIES_NUM))
            next_transmission_matrix = np.empty((HOSTS_NUM, 4))
            curr_parent_count = np.zeros(HOSTS_NUM)

            # document current state
            if HOSTS_NUM < 100:  # save full state only with a  memory scalable number of hosts
                curr_repeat_full_holobiome_population_each_gen.append(holobiome_population.copy())
            else:
                curr_repeat_full_holobiome_population_each_gen.append(holobiome_population.sum(axis=0))
            curr_repeat_full_lineages_each_gen.append(transmission_matrix[:, -1].copy())
            curr_repeat_fitness_each_gen.append(self.fitness_vec)

            # calc microbiome & parent for each future host
            for curr_host in range(HOSTS_NUM):
                if IS_PRE_GENERATED_TEMPLATE:
                    microbiome_structure = self.microbiome_structure_pool[random.randint(0, self.microbiome_structure_pool.shape[0] - 1)]
                else:
                    microbiome_structure, _ = Simulator.calc_microbiome_structure()
                parent_id = self.choose_parent_from_pool()
                curr_parent_count[parent_id] += 1
                next_transmission_matrix[curr_host] = transmission_matrix[parent_id]
                next_holobiome_population[curr_host] = Simulator.populate_host_microbiome(
                    (holobiome_population[parent_id]), peers, environment_vec, next_transmission_matrix[curr_host],
                    microbiome_structure, is_init=False)

            # document current state and advance buffers
            curr_repeat_parent_distributions_each_gen.append(curr_parent_count)
            holobiome_population = next_holobiome_population.copy()
            transmission_matrix = next_transmission_matrix.copy()
            generation += 1

            # check lineage convergence
            if np.unique(transmission_matrix[:, -1]).size <= COALESCENCE_THRESHOLD:
                break
        # document current repeat's outputs
        if SELECTION_TYPE == "neutral_fitness":
            np.savez_compressed(FILE_TO_SAVE + "repeat_%d_" % repeat + "output", generation=generation, holobiomes=curr_repeat_full_holobiome_population_each_gen, lineages=curr_repeat_full_lineages_each_gen, parent_distribution=curr_repeat_parent_distributions_each_gen)
        else:
            np.savez_compressed(FILE_TO_SAVE + "repeat_%d_" % repeat + "output", generation=generation, holobiomes=curr_repeat_full_holobiome_population_each_gen, lineages=curr_repeat_full_lineages_each_gen, parent_distribution=curr_repeat_parent_distributions_each_gen, contributions=self.bacteria_contribution, fitnesses=curr_repeat_fitness_each_gen)
        print("finished repeat %d" % repeat)

    def calc_first_generation(self):
        init_holobiome_population = np.ones((HOSTS_NUM, BACTERIA_SPECIES_NUM))
        for curr_host in range(HOSTS_NUM):
            if IS_PRE_GENERATED_TEMPLATE:
                microbiome_structure = \
                    self.microbiome_structure_pool[random.randint(0, self.microbiome_structure_pool.shape[0] - 1)]
            else:
                microbiome_structure, _ = Simulator.calc_microbiome_structure()
            init_holobiome_population[curr_host] = Simulator.populate_host_microbiome(
                None, None, None, None,
                microbiome_structure, is_init=True)
        transmission_matrix = np.array([[ENV_TRANSMISSION, PARENT_TRANSMISSION, PEERS_TRANSMISSION, 1]] * HOSTS_NUM, order='K')
        transmission_matrix[:, -1] = np.arange(HOSTS_NUM)
        return init_holobiome_population, transmission_matrix

    def calc_fitness_vec(self, holobiome_population):
        if SELECTION_TYPE == "neutral_fitness":
            return Sh.calc_neutral_fitness(HOSTS_NUM)
        else:
            return self.calc_non_neutral_fitness(holobiome_population)

    def calc_non_neutral_fitness(self, holobiome_population):
        """
        randomly choose parent
        :param holobiome_population: the previous generation population
        :return: chosen parent ID
        """
        fitness_vec = np.tile(self.bacteria_contribution, (HOSTS_NUM, 1))
        fitness_vec[holobiome_population == 0] = 0
        fitness_vec = fitness_vec.sum(axis=1) + MIN_FITNESS
        return fitness_vec

    def choose_parent_from_pool(self):
        return np.random.choice(range(HOSTS_NUM), 1, p=self.fitness_vec/sum(self.fitness_vec))[0]

    @staticmethod
    def generate_bacteria_contribution():
        if SELECTION_TYPE == "neutral_fitness":
            return np.ones(BACTERIA_SPECIES_NUM)
        elif SELECTION_TYPE == "decay_fitness":
            contribution_dist = MIN_CONTRIBUTION + np.exp(-FITNESS_DECAY * np.arange(MAX_CONTRIBUTION))
            return np.random.choice(np.arange(MAX_CONTRIBUTION), BACTERIA_SPECIES_NUM,
                                    p=contribution_dist / contribution_dist.sum())
        elif SELECTION_TYPE == "step_fitness":
            contribution_dist = np.full(BACTERIA_SPECIES_NUM, MIN_CONTRIBUTION)
            contribution_dist[np.random.choice(np.arange(BACTERIA_SPECIES_NUM), RELEVANT_STEP_BACTERIA_NUM,
                                               replace=False)] = MAX_CONTRIBUTION
            return contribution_dist
        else:
            assert 0, 'unimplemented SELECTION_TYPE {}'.format(SELECTION_TYPE)

    @staticmethod
    def populate_host_microbiome(parent_microbiome, peers, environment_vec, host_transmission_vec,
                                 microbiome_structure, is_init):
        if is_init:
            new_arrival_order = np.random.choice(BACTERIA_SPECIES_NUM, np.where(microbiome_structure > 0)[0].size,
                                                 replace=False)
        else:
            normalized_environment_vec = environment_vec / np.sum(environment_vec)
            normalized_parent_vec = parent_microbiome / np.sum(parent_microbiome)
            normalized_peers_vec = peers / np.sum(peers)
            microbiome_pool = (normalized_environment_vec * host_transmission_vec[0]) \
                              + (normalized_parent_vec * host_transmission_vec[1]) + (
                                          normalized_peers_vec * host_transmission_vec[2])
            if np.where(microbiome_structure > 0)[0].size <= np.count_nonzero(microbiome_pool /
                                                                              np.sum(microbiome_pool)):
                new_arrival_order = np.random.choice(BACTERIA_SPECIES_NUM, np.where(microbiome_structure > 0)[0].size,
                                                     replace=False, p=microbiome_pool / np.sum(microbiome_pool))
            else:
                new_arrival_order = np.random.choice(BACTERIA_SPECIES_NUM, np.count_nonzero(microbiome_pool /
                                                                                            np.sum(microbiome_pool)),
                                                     replace=False, p=microbiome_pool / np.sum(microbiome_pool))
        result = np.zeros(BACTERIA_SPECIES_NUM)
        result[new_arrival_order[:]] = microbiome_structure[0:len(new_arrival_order)]
        return result

    @staticmethod
    def calc_microbiome_structure():
        """
        calculates the mold for microbiome structure in the hosts to be later filled by specific taxa.
        :return: the microbiome structure mold and the waiting time between transmissions vector.
        """
        def distribute(available, weights):
            distributed_amounts = []
            total_weights = sum(weights)
            for weight in weights:
                weight = float(weight)
                p = weight / total_weights
                distributed_amount = round(p * available)
                distributed_amounts.append(distributed_amount)
                total_weights -= weight
                available -= distributed_amount
            return np.array(distributed_amounts)
        # empty vec to hold the future microbiome structure
        microbiome_structure = np.zeros(BACTERIA_SPECIES_NUM)
        # calculates the waiting times between transmission events according to the desired establishment
        # probability function
        transmission_times, establishment_vec = Sh.calc_waiting_times_between_transmission_events(ESTABLISHMENT_TYPE,
                                                                                                  BACTERIA_SPECIES_NUM,
                                                                                                  IS_PRE_GENERATED_TEMPLATE)
        # calculates the microbiome's structure based on microbes' logistic growth and time between transmission events
        for arrival_num in range(BACTERIA_SPECIES_NUM):
            for i in range(arrival_num + 1):
                abundance = round(Sh.logistic_growth_function(
                    np.sum(transmission_times[i:arrival_num]), SPECIE_CARRYING_CAPACITY))
                if np.sum(microbiome_structure) + abundance > GLOBAL_CARRYING_CAPACITY:
                    exces_dist = distribute(GLOBAL_CARRYING_CAPACITY - np.sum(microbiome_structure),
                                            microbiome_structure[microbiome_structure > 0])
                    microbiome_structure[microbiome_structure > 0] += exces_dist
                    break
                microbiome_structure[i] = abundance
            if np.sum(microbiome_structure) >= GLOBAL_CARRYING_CAPACITY:
                break
        return microbiome_structure, establishment_vec


##################
# simulation run #
##################

if __name__ == '__main__':
    sim = Simulator()
    for rep in range(REPEATS):  # multiprocess should be implemented per running environment
        sim.simulate(rep)
