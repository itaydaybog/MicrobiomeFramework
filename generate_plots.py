import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import ScalarFormatter
import matplotlib.font_manager as mf
import SimulatorHelper as sh
from itertools import combinations
from scipy.stats import ttest_ind

header_font = {'fontname': 'Helvetica', 'fontsize': 24, 'fontstyle': "oblique"}
palettes = ['#eea8a9', '#b3cede', '#ffcb9e', '#aad9aa']
np.set_printoptions(suppress=True)


def jaccard_dist(first, second):
    return 1 - len(set(first).intersection(second)) / len(set(first).union(second))


def calc_jaccard_distance(scenario):
    jaccard_means = []
    for repeat in range(100):
        holobiomes = np.load(f'results/{scenario[0]}/{scenario[1]}/{scenario[2]}/repeat_{repeat}_output.npz', allow_pickle=True)['holobiomes'][scenario[3]]
        existing_microbes = dict()
        for host in range(holobiomes.shape[0]):
                existing_microbes[host] = set(np.argwhere(holobiomes[host] > 0).flatten().tolist())
        keys = list(existing_microbes.keys())
        all_result_dict = {}
        for k in keys:
            for l in keys:
                if k == l:
                    continue
                all_result_dict[(k, l)] = all_result_dict.get((l, k), jaccard_dist(existing_microbes[k], existing_microbes[l]))
        jaccard_means.append(np.mean(list(all_result_dict.values())))
    print(np.mean(jaccard_means), round(np.mean(jaccard_means), 3))


def plot_fitness_score_distribution(scenarios, labels, palettes):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # plot fitness score distribution
    results = []
    for i, scenario in enumerate(scenarios):
        all_fitnesses = []
        for repeat in range(100):
            fitnesses = np.load(f'results/{scenario[0]}/{scenario[1]}/{scenario[2]}/repeat_{repeat}_output.npz', allow_pickle=True)['fitnesses'][scenario[3]]
            norm_fitnesses = fitnesses / max(fitnesses)
            all_fitnesses += norm_fitnesses.tolist()
        results.append(all_fitnesses)
        sns.distplot(all_fitnesses, label=labels[i], ax=axs[0])
    axs[0].set_xlabel("Fitness score")
    axs[0].set_xlim([-0.1, 1.1])
    axs[0].set_title("Fitness scores distributions", **header_font)
    font = mf.FontProperties(family='Helvetica', size=14, style='normal')
    axs[0].legend(prop=font)
    axs[0].text(-0.1, 11.15, "(a)")

    # plot fitness score variance
    results = []
    for i, scenario in enumerate(scenarios):
        curr_variances = []
        for repeat in range(100):
            fitnesses = np.load(f'results/{scenario[0]}/{scenario[1]}/{scenario[2]}/repeat_{repeat}_output.npz', allow_pickle=True)['fitnesses'][scenario[3]]
            norm_fitnesses = fitnesses / max(fitnesses)
            curr_variances.append(np.var(norm_fitnesses))
        results.append(curr_variances)
    t, p = ttest_ind(results[0], results[1])
    print(p)
    df = pd.DataFrame(results, index=labels).T
    sns.boxplot(data=df, palette=palettes, showfliers=False, ax=axs[1])
    axs[1].set_xticklabels(labels)
    axs[1].set_ylabel("Variance")
    axs[1].set_title("Fitness scores variance", **header_font)
    axs[1].text(-0.5, 0.0547, "(b)")
    plt.tight_layout()
    plt.show()


def plot_fig_4():
    plot_fitness_score_distribution(scenarios=[["vertical", "Vertebrates", "decay_fitness_0.005", 0],
                                               ["vertical", "Insects", "decay_fitness_0.005", 0]],
                                    labels=['species-rich', 'species-poor'],
                                    palettes=palettes[1:3])


def plot_fig_6():
    plot_fitness_score_distribution(scenarios=[["vertical", "Vertebrates", "step_fitness_30", 0],
                                               ["vertical", "Vertebrates", "decay_fitness_0.7", 0],
                                               ["vertical", "Vertebrates", "decay_fitness_0.005", 0]],
                                    labels=['step', 'midpoint', 'uniform'],
                                    palettes=palettes[1:])


def plot_fig_8():
    plot_fitness_score_distribution(scenarios=[["vertical", "Vertebrates", "pop_size_20", 0],
                                              ["vertical", "Vertebrates", "pop_size_200", 0],
                                              ["vertical", "Vertebrates", "pop_size_2000", 0]],
                                    labels=['20 hosts', '200 hosts', '2000 hosts'],
                                    palettes=palettes[1:])


def plot_convergence_generation_boxplots(transmission, scenarios, labels, y_label, pallete):
    results = []
    for scenario in scenarios:
        results.append([k / scenario[2] for k in [np.load(f'results/{transmission}/{scenario[0]}/{scenario[1]}/repeat_{x}_output.npz',
                                                  allow_pickle=True)['generation'] for x in range(100)]])
    df = pd.DataFrame(results, index=labels).T
    ax = sns.boxplot(data=df, palette=pallete, showfliers=False)
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_label)
    plt.title(transmission)
    for i, inds in enumerate(combinations(range(len(results)), 2)):
        ind1, ind2 = inds
        t, p = ttest_ind(results[ind1], results[ind2])
        print(labels[ind1], labels[ind2], round(p, 5) if p > 0.1 else "{:.5e}".format(p))
        y, h, col = (95 + i * 7, 2, 'k') if scenarios[-1][-1] == 1 else (1.7 + i*0.2, 0.1, 'k')
        plt.plot([ind1, ind1, ind2, ind2], [y, y + h, y + h, y], lw=1.5, c=col)
        plt.text((ind1 + ind2) * .5, y + h, str(round(p, 5) if p > 0.1 else "{:.5e}".format(p)), ha='center',
                 va='bottom', color=col)
    plt.show()


def plot_fig_5(transmission):
    plot_convergence_generation_boxplots(transmission=transmission,
                                         scenarios=[['Vertebrates', 'neutral_fitness', 1],
                                                    ['Vertebrates', 'decay_fitness_0.005', 1],
                                                    ['Insects', 'decay_fitness_0.005', 1]],
                                         labels=['neutral dynamics', 'species-rich', 'species-poor'],
                                         y_label="Convergence generation",
                                         pallete=palettes[:3])


def plot_fig_7(transmission):
    plot_convergence_generation_boxplots(transmission=transmission,
                                         scenarios=[['Vertebrates', 'neutral_fitness', 1],
                                                    ['Vertebrates', 'step_fitness_30', 1],
                                                    ['Vertebrates', 'decay_fitness_0.7', 1],
                                                    ['Vertebrates', 'decay_fitness_0.005', 1]],
                                         labels=['neutral dynamics', 'step', 'midpoint', 'uniform'],
                                         y_label="Convergence generation",
                                         pallete=palettes)


def plot_fig_9(transmission):
    plot_convergence_generation_boxplots(transmission=transmission,
                                         scenarios=[['Vertebrates', 'pop_size_20', 20],
                                                    ['Vertebrates', 'pop_size_200', 200],
                                                    ['Vertebrates', 'pop_size_2000', 2000]],
                                         labels=['20 hosts', '200 hosts', '2000 hosts'],
                                         y_label="Normalized convergence generation",
                                         pallete=palettes[1:])


def plot_fig_s1a():
    repeat = random.randint(0, 99)
    contributions = np.load(f'results/vertical/Vertebrates/step_fitness_30/repeat_{repeat}_output.npz', allow_pickle=True)['contributions']
    plt.figure(figsize=(7, 7))
    sns.distplot(contributions, color="#5698c6", kde=False, norm_hist=False, hist_kws=dict(alpha=1))
    plt.xlabel("Contribution to host fitness")
    plt.ylabel('number of microbe species')
    plt.title("\"step\" contribution distribution", **header_font)
    plt.text(-2, 2100, "(a)")
    plt.tight_layout()
    plt.show()


def plot_fig_s1b():
    f, axes = plt.subplots(3, 3, figsize=(7, 7))
    decay_params = [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.005]
    for i, a in enumerate([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]):
        repeat = random.randint(0, 99)
        contributions = np.load(f'results/vertical/Vertebrates/decay_fitness_{decay_params[i]}/repeat_{repeat}_output.npz', allow_pickle=True)['contributions']
        sns.distplot(contributions, ax=a, color="#5698c6", kde=False, norm_hist=False, hist_kws=dict(alpha=1))
        a.title.set_text(f"$\lambda_3 = {decay_params[i]}$")
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Contribution to host fitness")
    f.text(0.5, 0.985, "\"exponential decay\" contribution distribution", ha='center', va='center', **header_font)
    f.text(0.1, 0.98, "(b)", ha='center', va='center', **header_font)
    f.text(0.03, 0.5, 'number of microbe species', ha='center', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


def plot_fig_s2a(num_of_reps):
    class FixedOrderFormatter(ScalarFormatter):
        """Formats axis ticks using scientific notation with a constant order of
        magnitude"""

        def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
            self._order_of_mag = order_of_mag
            ScalarFormatter.__init__(self, useOffset=useOffset,
                                     useMathText=useMathText)

        def _set_orderOfMagnitude(self, range):
            """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
            self.orderOfMagnitude = self._order_of_mag

    mb_structure_pool = np.load("mb_structures/insects_mbs.npz")["microbiome_structures"]
    for i in range(num_of_reps):
        f, axes = plt.subplots(2, 2, figsize=(7, 7))
        for a in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
            mb_struct = mb_structure_pool[random.randint(0, mb_structure_pool.shape[0] - 1)]
            mb_struct = mb_struct[np.where(mb_struct > 0)]
            hist = []
            for j in range(len(mb_struct)):
                hist += [j] * int(mb_struct[j])
            sns.distplot(hist, color="#5698c6", ax=a, kde=False, norm_hist=False, hist_kws=dict(alpha=1))
        f.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("establishment order within host")
        f.text(0.03, 0.5, 'number of microbes', ha='center', va='center', rotation='vertical')
        plt.title("species-poor microbiome structures", **header_font)
        plt.text(0,1.015, "(a)")
        plt.tight_layout()
        plt.show()


def plot_fig_s2b(num_of_reps):
    class FixedOrderFormatter(ScalarFormatter):
        """Formats axis ticks using scientific notation with a constant order of
        magnitude"""

        def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
            self._order_of_mag = order_of_mag
            ScalarFormatter.__init__(self, useOffset=useOffset,
                                     useMathText=useMathText)

        def _set_orderOfMagnitude(self, range):
            """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
            self.orderOfMagnitude = self._order_of_mag

    mb_structure_pool = np.load("mb_structures/vertebrates_mbs.npz")["microbiome_structures"]
    for i in range(num_of_reps):
        f, axes = plt.subplots(2, 2, figsize=(7, 7))
        for a in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
            mb_struct = mb_structure_pool[random.randint(0, mb_structure_pool.shape[0] - 1)]
            mb_struct = mb_struct[np.where(mb_struct > 0)]
            hist = []
            for j in range(len(mb_struct)):
                hist += [j] * int(mb_struct[j])
            sns.distplot(hist, color="#5698c6", ax=a, kde=False, norm_hist=False, hist_kws=dict(alpha=1))
        f.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("establishment order within host")
        f.text(0.03, 0.5, 'number of microbes ($x10^7$)', ha='center', va='center', rotation='vertical')
        plt.title("species-rich microbiome structures", **header_font)
        plt.text(0,1.07, "(b)")
        plt.tight_layout()
        plt.show()


def plot_fig_s3():
    f, axes = plt.subplots(1, 3, figsize=(10, 4))

    step_hist = np.zeros(50)
    step_hist[0] = 0.975
    step_hist[49] = 0.025
    axes[0].bar(range(50), step_hist, color="#5698c6")
    axes[0].set_ylim((0,1))
    axes[0].set_xlabel('contribution amount')
    axes[0].set_ylabel('probability')
    axes[0].set_title('step', **header_font)
    axes[0].text(-1, 1.015, "(a)")

    uniform_hist = 0.005 * np.exp(np.arange(50) * -0.005)
    axes[1].bar(range(50), uniform_hist / uniform_hist.sum(), color="#5698c6")
    axes[1].set_ylim((0,1))
    axes[1].set_xlabel('contribution amount')
    axes[1].set_ylabel('probability')
    axes[1].set_title('uniform', **header_font)
    axes[1].text(-1, 1.015, "(b)")

    midpoint_hist = 0.7 * np.exp(np.arange(50) * -0.7)
    axes[2].bar(range(50), midpoint_hist / midpoint_hist.sum(), color="#5698c6")
    axes[2].set_ylim((0,1))
    axes[2].set_xlabel('contribution amount')
    axes[2].set_ylabel('probability')
    axes[2].set_title('midpoint', **header_font)
    axes[2].text(-1, 1.015, "(c)")

    plt.tight_layout()
    plt.show()


def plot_fig_s4():
    f, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].plot(range(2000), np.ones(2000), color="#5698c6")
    axes[0].set_xlabel('arrival order to host')
    axes[0].set_ylabel('success probability')
    axes[0].set_title('neutral', **header_font)
    axes[0].text(-60, 1.058, "(a)")

    vec = sh.calc_exponential_decay_establishment_vector(2000)
    axes[1].plot(range(2000), vec, color="#5698c6")
    axes[1].set_xlabel('arrival order to host')
    axes[1].set_ylabel('success probability')
    axes[1].set_title('exponential decay', **header_font)
    axes[1].text(-60, 1.08, "(b)")

    vec = sh.calc_hump_establishment_function_vector(2000)
    axes[2].plot(range(2000), vec, color="#5698c6")
    axes[2].set_xlabel('arrival order to host')
    axes[2].set_ylabel('success probability')
    axes[2].set_title('hump', **header_font)
    axes[2].text(-60, 1.06, "(c)")

    plt.tight_layout()
    plt.show()


def plot_fig_s5():
    plot_fig_5('horizontal')  # a
    plot_fig_5('midway')      # b


def plot_fig_s6():
    plot_fig_7('horizontal')  # a
    plot_fig_7('midway')      # b


def plot_fig_s7():
    plot_fig_9('horizontal')  # a
    plot_fig_9('midway')      # b


if __name__ == '__main__':
    mf._rebuild()
    # main paper
    plot_fig_4()
    plot_fig_5('vertical')
    plot_fig_6()
    plot_fig_7('vertical')
    plot_fig_8()
    plot_fig_9('vertical')

    # supplementary
    plot_fig_s1a()
    plot_fig_s1b()
    plot_fig_s2a(4)
    plot_fig_s2b(4)
    plot_fig_s3()
    plot_fig_s4()
    plot_fig_s5()
    plot_fig_s6()
    plot_fig_s7()
