import os
import random
import shutil
import simulator_math_utils as Sh
from simulator_params import *
from simulator_IO_utils import *
from multiprocess import Pool

np.set_printoptions(suppress=True)


class MicrobiomeHostCoEvolutionSimulator:
    """
    Class running the microbiome-host co-evolution model
    """

    def __init__(self, results_parent_dir, save_only_population_sums=False):
        self.fitness_vec = np.zeros(HOSTS_NUM)
        self.bacteria_id_matrix = np.array([np.arange(BACTERIA_SPECIES_NUM)] * HOSTS_NUM, order='K')
        self.save_only_population_sums = save_only_population_sums
        self.results_output_path = os.path.join(results_parent_dir,
                                                get_transmission_name(PARENT_TRANSMISSION, PEERS_TRANSMISSION, ENV_TRANSMISSION),
                                                MICROBIOME_CLASS, get_results_path(SELECTION_TYPE, FITNESS_DECAY, RELEVANT_STEP_BACTERIA_NUM))
        if IS_PRE_GENERATED_TEMPLATE:
            with gzip.GzipFile(os.path.join('mb_structures', f'{MICROBIOME_CLASS.lower()}_mbs.npy.gz'), "r") as f:
                self.microbiome_structure_pool = np.load(f)

    def run_simulations(self, num_of_repeats):
        """
        runs the simulation, multiple repeats ar run with multiprocessing.
        :param num_of_repeats - how many repeats of the simulation tor run.
        """
        # freeze a copy of the current code
        os.makedirs(self.results_output_path, exist_ok=True)
        for file in ['microbiome_host_co_evolution_simulator', 'simulator_params', 'simulator_IO_utils', 'simulator_math_utils']:
            shutil.copy(f"{file}.py", os.path.join(self.results_output_path, f"{file}.py"))

        # run
        with Pool() as p:
            print(f'Running microbiome-host co-evolution simulation with {num_of_repeats} repeats')
            p.map(self.simulate_single_repeat, list(range(num_of_repeats)))

    def init_single_repeat(self):
        bacteria_contribution = MicrobiomeHostCoEvolutionSimulator.generate_bacteria_contribution()
        full_symbiotes_population_per_gen, parent_distributions_per_gen, full_lineages_per_gen, fitness_each_gen = [], [], [], []
        environment_vec = np.ones(BACTERIA_SPECIES_NUM)
        symbiotes_population, transmission_matrix = self.calc_first_generation()
        return bacteria_contribution, environment_vec, fitness_each_gen, full_lineages_per_gen, full_symbiotes_population_per_gen, parent_distributions_per_gen, \
            symbiotes_population, transmission_matrix

    def init_single_generation(self, symbiotes_population, bacteria_contribution):
        self.fitness_vec = self.calc_fitness_vec(symbiotes_population, bacteria_contribution)
        peers = np.sum(symbiotes_population, axis=0)
        next_symbiotes_population = np.empty((HOSTS_NUM, BACTERIA_SPECIES_NUM))
        next_transmission_matrix = np.empty((HOSTS_NUM, 4))
        curr_parent_count = np.zeros(HOSTS_NUM)
        return curr_parent_count, next_symbiotes_population, next_transmission_matrix, peers

    def simulate_single_repeat(self, repeat):
        """
        runs a single repeat of the simulation from start (gen 0) to end (number of host lineages < COALESCENCE_THRESHOLD).
        :param repeat:
        """
        bacteria_contribution, environment_vec, fitness_each_gen, full_lineages_per_gen, full_symbiotes_population_per_gen, \
            parent_distributions_per_gen, symbiotes_population, transmission_matrix = self.init_single_repeat()

        # run generations of repeat
        generation = 0
        while True:  # run until lineage coalescence
            # setup generation's arguments and buffers
            curr_parent_count, next_symbiotes_population, next_transmission_matrix, peers = self.init_single_generation(symbiotes_population, bacteria_contribution)

            # document current state
            if HOSTS_NUM > 100 and self.save_only_population_sums:
                full_symbiotes_population_per_gen.append(symbiotes_population.sum(axis=0))
            else:  # save full state only with a memory scalable number of hosts
                full_symbiotes_population_per_gen.append(symbiotes_population.copy())

            full_lineages_per_gen.append(transmission_matrix[:, -1].copy())
            fitness_each_gen.append(self.fitness_vec)

            # calc microbiome & parent for each future host
            self.calculate_cur_gen_parents_and_microbiomes(curr_parent_count, environment_vec, next_symbiotes_population,
                                                           next_transmission_matrix, peers, symbiotes_population, transmission_matrix)

            # document current state and advance buffers
            parent_distributions_per_gen.append(curr_parent_count)
            symbiotes_population = next_symbiotes_population.copy()
            transmission_matrix = next_transmission_matrix.copy()
            generation += 1

            # check lineage convergence
            if np.unique(transmission_matrix[:, -1]).size <= COALESCENCE_THRESHOLD:
                break

        # document results
        self.save_results(repeat, generation, full_lineages_per_gen, full_lineages_per_gen, parent_distributions_per_gen, fitness_each_gen, bacteria_contribution)
        print("finished repeat %d" % repeat)

    def save_results(self, repeat, generation, full_symbiotes_population_per_gen, full_lineages_per_gen, parent_distributions_per_gen, fitness_each_gen, bacteria_contribution):
        repeat_results_path = os.path.join(self.results_output_path, f'repeat_{repeat}_output')
        os.makedirs(repeat_results_path, exist_ok=True)
        save_single_result_as_npy_gz(os.path.join(repeat_results_path, 'generation'), generation)
        save_single_result_as_npy_gz(os.path.join(repeat_results_path, 'symbiotes'), full_symbiotes_population_per_gen)
        save_single_result_as_npy_gz(os.path.join(repeat_results_path, 'lineages'), full_lineages_per_gen)
        save_single_result_as_npy_gz(os.path.join(repeat_results_path, 'parent_distribution'), parent_distributions_per_gen)
        if SELECTION_TYPE != 'neutral_fitness':
            save_single_result_as_npy_gz(os.path.join(repeat_results_path, 'fitnesses'), fitness_each_gen)
            save_single_result_as_npy_gz(os.path.join(repeat_results_path, 'contributions'), bacteria_contribution)

    def calculate_cur_gen_parents_and_microbiomes(self, curr_parent_count, environment_vec, next_symbiotes_population, next_transmission_matrix, peers, symbiotes_population, transmission_matrix):
        for curr_host in range(HOSTS_NUM):
            if IS_PRE_GENERATED_TEMPLATE:
                microbiome_structure = self.microbiome_structure_pool[random.randint(0, self.microbiome_structure_pool.shape[0] - 1)]
            else:
                microbiome_structure, _ = MicrobiomeHostCoEvolutionSimulator.calc_microbiome_structure()
            parent_id = self.choose_parent_from_pool()
            curr_parent_count[parent_id] += 1
            next_transmission_matrix[curr_host] = transmission_matrix[parent_id]
            next_symbiotes_population[curr_host] = MicrobiomeHostCoEvolutionSimulator.populate_host_microbiome(
                (symbiotes_population[parent_id]), peers, environment_vec, next_transmission_matrix[curr_host],
                microbiome_structure, is_init=False)

    def calc_first_generation(self):
        init_symbiotes_population = np.ones((HOSTS_NUM, BACTERIA_SPECIES_NUM))
        for curr_host in range(HOSTS_NUM):
            if IS_PRE_GENERATED_TEMPLATE:
                microbiome_structure = \
                    self.microbiome_structure_pool[random.randint(0, self.microbiome_structure_pool.shape[0] - 1)]
            else:
                microbiome_structure, _ = MicrobiomeHostCoEvolutionSimulator.calc_microbiome_structure()
            init_symbiotes_population[curr_host] = MicrobiomeHostCoEvolutionSimulator.populate_host_microbiome(
                None, None, None, None,
                microbiome_structure, is_init=True)
        transmission_matrix = np.array([[ENV_TRANSMISSION, PARENT_TRANSMISSION, PEERS_TRANSMISSION, 1]] * HOSTS_NUM, order='K')
        transmission_matrix[:, -1] = np.arange(HOSTS_NUM)
        return init_symbiotes_population, transmission_matrix

    @staticmethod
    def calc_fitness_vec(symbiotes_population, bacteria_contribution):
        if SELECTION_TYPE == "neutral_fitness":
            return Sh.calc_neutral_fitness(HOSTS_NUM)
        else:
            return MicrobiomeHostCoEvolutionSimulator.calc_non_neutral_fitness(symbiotes_population, bacteria_contribution)

    @staticmethod
    def calc_non_neutral_fitness(symbiotes_population, bacteria_contribution):
        """
        randomly choose parent
        :param symbiotes_population: the previous generation population
        :return: chosen parent ID
        """
        fitness_vec = np.tile(bacteria_contribution, (HOSTS_NUM, 1))
        fitness_vec[symbiotes_population == 0] = 0
        fitness_vec = fitness_vec.sum(axis=1) + MIN_FITNESS
        return fitness_vec

    def choose_parent_from_pool(self):
        return np.random.choice(range(HOSTS_NUM), 1, p=self.fitness_vec / sum(self.fitness_vec))[0]

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


def main():
    results_parent_dir = 'results'
    num_of_repeats = 100

    validate_params()
    sim = MicrobiomeHostCoEvolutionSimulator(results_parent_dir=results_parent_dir)
    sim.run_simulations(num_of_repeats)


if __name__ == '__main__':
    main()
