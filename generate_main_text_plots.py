import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gzip
import matplotlib.font_manager as mf
from itertools import combinations
from scipy.stats import ttest_ind

header_font = {'fontsize': 15}#, 'fontstyle': "oblique"}
axis_font = {'fontsize': 13}
palettes = ['#eea8a9', '#b3cede', '#ffcb9e', '#aad9aa']
np.set_printoptions(suppress=True)


def read_np_gzip_file(path, field):
    with gzip.GzipFile(f'{path}/{field}', "r") as f:
        return np.load(f)


def plot_fitness_score_distribution(scenarios, labels, palettes,
                                    output_path,
                                    results_dir='results',
                                    num_to_sample=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # plot fitness score distribution
    results = []
    for i, scenario in enumerate(scenarios):
        all_fitnesses = []
        for repeat in range(100):
            fitnesses = read_np_gzip_file(f'{results_dir}/{scenario[0]}/{scenario[1]}/{scenario[2]}/repeat_{repeat}_output', 'fitnesses')[scenario[3]]
            norm_fitnesses = fitnesses / max(fitnesses)
            all_fitnesses += norm_fitnesses.tolist()
        if num_to_sample is not None:
            thresh = int(100 * num_to_sample)
            all_fitnesses = all_fitnesses[:thresh]
        results.append(all_fitnesses)
        sns.distplot(all_fitnesses, label=labels[i], ax=axs[0])
    axs[0].set_xlabel("Fitness score", **axis_font)
    axs[0].set_xlim([-0.1, 1.1])
    axs[0].set_title("Fitness scores distributions", **header_font)
    font = mf.FontProperties(size=13)
    axs[0].legend(prop=font)

    # plot fitness score variance
    results = []
    for i, scenario in enumerate(scenarios):
        curr_variances = []
        for repeat in range(100):
            fitnesses = read_np_gzip_file(f'{results_dir}/{scenario[0]}/{scenario[1]}/{scenario[2]}/repeat_{repeat}_output', 'fitnesses')[scenario[3]]
            if num_to_sample is not None:
                thresh = num_to_sample
                fitnesses = fitnesses[:int(thresh)]
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
    plt.tight_layout()

    plt.savefig(output_path, format=output_path.split('.')[-1])
    plt.show()


def plot_fig_4():
    plot_fitness_score_distribution(scenarios=[["vertical", "Vertebrates", "decay_fitness_0.005", 0],
                                               ["vertical", "Insects", "decay_fitness_0.005", 0]],
                                    labels=['species-rich', 'species-poor'],
                                    palettes=palettes[1:3], output_path='fig4.pdf')


def plot_fig_6():
    plot_fitness_score_distribution(scenarios=[["vertical", "Vertebrates", "step_fitness_30", 0],
                                               ["vertical", "Vertebrates", "decay_fitness_0.7", 0],
                                               ["vertical", "Vertebrates", "decay_fitness_0.005", 0]],
                                    labels=['step', 'midpoint', 'uniform'],
                                    palettes=palettes[1:],
                                    output_path='fig6.pdf')


def plot_fig_8():
    plot_fitness_score_distribution(scenarios=[["vertical", "Vertebrates_from_drive", "pop_size_20", 0],
                                              ["vertical", "Vertebrates_from_drive", "pop_size_200_sum", 0],
                                              ["vertical", "Vertebrates_from_drive", "pop_size_2000_sum", 0]],
                                    labels=['20 hosts', '200 hosts', '2000 hosts'],
                                    palettes=palettes[1:], output_path='fig8.pdf', num_to_sample=20)


def plot_convergence_generation_boxplots(transmission, title, scenarios, labels, y_label, pallete, output_path,
                                         results_dir="results"):
    results = []
    for scenario in scenarios:
        results.append([k / scenario[2] for k in [read_np_gzip_file(f'{results_dir}/{transmission}/{scenario[0]}/{scenario[1]}/repeat_{x}_output', 'generation')
                                                  for x in range(100)]])
    df = pd.DataFrame(results, index=labels).T
    ax = sns.boxplot(data=df, palette=pallete, showfliers=False)
    ax.set_xticklabels(labels, **axis_font)
    ax.set_ylabel(y_label, **axis_font)
    plt.title(title, **header_font)
    for i, inds in enumerate(combinations(range(len(results)), 2)):
        ind1, ind2 = inds
        t, p = ttest_ind(results[ind1], results[ind2])
        print(labels[ind1], labels[ind2], round(p, 5) if p > 0.1 else "{:.5e}".format(p))
        y, h, col = (95 + i * 7, 2, 'k') if scenarios[-1][-1] == 1 else (1.7 + i*0.2, 0.1, 'k')
        plt.plot([ind1, ind1, ind2, ind2], [y, y + h, y + h, y], lw=1.5, c=col)
        plt.text((ind1 + ind2) * .5, y + h, str(round(p, 5) if p > 0.1 else "{:.5e}".format(p)), ha='center',
                 va='bottom', color=col)
    plt.savefig(output_path, format=output_path.split('.')[-1])
    plt.show()


def plot_fig_5(transmission, title, output_path):
    plot_convergence_generation_boxplots(transmission=transmission, title=title,
                                         scenarios=[['Vertebrates', 'neutral_fitness', 1],
                                                    ['Vertebrates', 'decay_fitness_0.005', 1],
                                                    ['Insects', 'decay_fitness_0.005', 1]],
                                         labels=['neutral dynamics', 'species-rich', 'species-poor'],
                                         y_label="Convergence generation",
                                         pallete=palettes[:3], output_path=output_path)


def plot_fig_7(transmission, title, output_path, results_dir="results"):
    plot_convergence_generation_boxplots(transmission=transmission, title=title,
                                         scenarios=[['Vertebrates', 'neutral_fitness', 1],
                                                    ['Vertebrates', 'step_fitness_30', 1],
                                                    ['Vertebrates', 'decay_fitness_0.7', 1],
                                                    ['Vertebrates', 'decay_fitness_0.005', 1]],
                                         labels=['neutral dynamics', 'step', 'midpoint', 'uniform'],
                                         y_label="Convergence generation",
                                         pallete=palettes,
                                         output_path=output_path, results_dir=results_dir)


def plot_fig_9(transmission, title, output_path):
    plot_convergence_generation_boxplots(transmission=transmission, title=title,
                                         scenarios=[['Vertebrates', 'pop_size_20', 20],
                                                    ['Vertebrates', 'pop_size_200', 200],
                                                    ['Vertebrates', 'pop_size_2000', 2000]],
                                         labels=['20 hosts', '200 hosts', '2000 hosts'],
                                         y_label="Normalized convergence generation",
                                         pallete=palettes[1:],
                                         output_path=output_path)


if __name__ == '__main__':
    # main paper
    plot_fig_4()
    plot_fig_5('vertical', 'vertical transmission', 'fig5a.pdf')
    plot_fig_5('midway', 'midway transmission', 'fig5b.pdf')
    plot_fig_5('horizontal', 'horizontal transmission', 'fig5c.pdf')
    plot_fig_6()
    plot_fig_7('vertical', '', 'fig7.pdf')
    plot_fig_8()
    plot_fig_9('vertical', '', 'fig9.pdf')
