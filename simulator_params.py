###########################################################################################
# as described in the paper, for time efficiency we've pre-generated microbiome           #
# configurations to be randomly selected during the simulation.                           #
# To use this functionality set IS_PRE_GENERATED_TEMPLATE = False.                        #
# to randomly generate blank composition on the fly use IS_PRE_GENERATED_TEMPLATE = True. #
###########################################################################################
IS_PRE_GENERATED_TEMPLATE = True

# --- transmission parameters --- #
PEERS_TRANSMISSION = 0  # T_h
PARENT_TRANSMISSION = 1  # T_v
ENV_TRANSMISSION = 0  # coefficient for introduction of random microbes into the system (for further exploration)

# --- microbiome parameters --- #
MICROBIOME_CLASS = "Vertebrates"  # Vertebrates, Insects  (Only for IS_PRE_GENERATED_TEMPLATE = True)
BACTERIA_SPECIES_NUM = 2000  # B
GLOBAL_CARRYING_CAPACITY = 10**8  # C_g
SPECIE_CARRYING_CAPACITY = 10**7  # C_s
ESTABLISHMENT_TYPE = 'hump'  # null, hump, decay

# --- host parameters --- #
HOSTS_NUM = 50  # N
COALESCENCE_THRESHOLD = 2  # AC

# --- selection parameters --- #
# type #
SELECTION_TYPE = 'decay_fitness'  # neutral_fitness, decay_fitness, step_fitness
# params if "step" #
RELEVANT_STEP_BACTERIA_NUM = 30  # 𝜆_3
# params if "decay" #
FITNESS_DECAY = 0.005  # 𝜆_3
# general#
MAX_CONTRIBUTION = 50  # C_max
MIN_CONTRIBUTION = 0.005  # C_min
MIN_FITNESS = 1


def validate_params():
    if SELECTION_TYPE not in ['decay_fitness', 'step_fitness', 'neutral_fitness']:
        raise ValueError("Unsupported fitness type. Please use decay_fitness, step_fitness, neutral_fitness")
    if HOSTS_NUM <= COALESCENCE_THRESHOLD:
        raise ValueError("HOSTS_NUM must be greater than COALESCENCE_THRESHOLD")
    if BACTERIA_SPECIES_NUM < RELEVANT_STEP_BACTERIA_NUM:
        raise ValueError("BACTERIA_SPECIES_NUM must be greater than RELEVANT_STEP_BACTERIA_NUM")
    if PARENT_TRANSMISSION + PEERS_TRANSMISSION + ENV_TRANSMISSION != 1:
        raise ValueError("transmission probability factors should add up to 1")

