import pickle
from glob import glob
import sys

DIR_DATA = "/home/ext/mr/smer/regnierc/SAVE/codes/validation_tools_drifters/results/"
list_SYSTEM = ["GLO12", "GLO12_TEST5", "GLOBCURR"]
LEAD = "HDCT"
def load_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    # ==========================================================
    # Collect data
    # ==========================================================

    all_sep_daily = []
    all_liu_daily = []

    for res in results:
        if res is None:
            continue

        if "sep_dist_daily" in res:
            all_sep_daily.append(res["sep_dist_daily"])

        if "liu_skill_daily" in res:
            all_liu_daily.append(res["liu_skill_daily"])

    all_sep_daily = np.concatenate(all_sep_daily, axis=0)
    all_liu_daily = np.concatenate(all_liu_daily, axis=0)
    return all_sep_daily, all_liu_daily


for SYSTEM in list_SYSTEM:

    pickle_file = glob(f"{DIR_DATA}Parcels_Drift-{SYSTEM}_{LEAD}*.pkl")[0]
    all_sep_daily, all_liu_daily = load_results(pickle_file)

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))

    fig.suptitle(f"{MODEL}, {DATE_START}-{DATE_END}", fontsize=14)

    colors = ['r', 'b', 'g', 'k', 'orange']

    # ==========================================================
    # Separation distance histogram
    # ==========================================================

    ax = axs[0]
    bin_dict = {}
    bin_dict[0] = 1
    bin_dict[1] = 2
    bin_dict[2] = 2
    bin_dict[3] = 5
    bin_dict[4] = 5
    bin_dict[5] = 5
    bin_dict[6] = 5
    bin_dict[7] = 10
    bin_dict[8] = 10
    bin_dict[9] = 10


    for i, day in enumerate(days):
        bins = np.arange(0, 200, bin_dict[day-1])

        idx = day - 1
        if idx >= all_sep_daily.shape[1]:
            continue

        dist = all_sep_daily[:, idx]
        dist = dist[np.isfinite(dist)]

        if len(dist) == 0:
            continue

        counts, _ = np.histogram(dist, bins=bins)
        freq = counts / len(dist)

        bin_centres = bins[:-1] + (bins[1] - bins[0]) / 2
        ax.plot(bin_centres, freq, lw=2, color=colors[i], label=f"{day} day")

    ax.set_xlabel(f"Separation distance [km] {scheme} dt:{sampling}")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.15)
    ax.grid(True, which="major", axis="both", alpha=0.3,linestyle="--")
    ax.legend()

    # ==========================================================
    # Liu score histogram
    # ==========================================================

    ax = axs[1]
    bin_dict2 = {}
    bin_dict2[0] = 0.1 #0.01
    bin_dict2[1] = 0.1 #0.02
    bin_dict2[2] = 0.1 #0.02
    bin_dict2[3] = 0.1 #0.05
    bin_dict2[4] = 0.1 #0.05
    bin_dict2[5] = 0.1 #0.05
    bin_dict2[6] = 0.1 #0.05
    bin_dict2[7] = 0.1 #0.1
    bin_dict2[8] = 0.1 #0.1
    bin_dict2[9] = 0.1 #0.1
    for i, day in enumerate(days):

        #bins2 = np.arange(0, 3, 0.1)
        bins2 = np.arange(0, 3, bin_dict2[day-1])
        bin_centres2 = bins2[:-1] + (bins2[1] - bins2[0]) / 2
        idx = day - 1
        if idx >= all_liu_daily.shape[1]:
            continue

        liu = all_liu_daily[:, idx]
        liu = liu[np.isfinite(liu)]

        if len(liu) == 0:
            continue

        counts, _ = np.histogram(liu, bins=bins2)
        freq = counts / len(liu)

        ax.plot(bin_centres2, freq, lw=2, color=colors[i], label=f"{day} day")

    ax.set_xlabel("Liu index")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 0.20)
    ax.grid(True, which="major", axis="both", alpha=0.3,linestyle="--")
    ax.legend()

    # ==========================================================
    # Save
    # ==========================================================

    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()

    log.info(f"Saved: {fig_out}")
