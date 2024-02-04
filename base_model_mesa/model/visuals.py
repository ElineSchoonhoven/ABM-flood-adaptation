import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import dill
import scipy as sc

used_cols = np.array(["olive", "dodgerblue","green", "cyan", "red", 
                      "m", 'orange', 'gold' ])

def load_data(batch_id):
    with open(f"results/aggregated_outcomes_{batch_id}", "rb") as file:
        agg_res = dill.load(file)

    with open(f"results/indexes_used_{batch_id}", "rb") as file:
        ind_used = dill.load(file)

    with open(f"results/inputs_used_{batch_id}", "rb") as file:
        sam = dill.load(file)


    return agg_res, ind_used, sam

def get_vars_from_batch_id(batch_id):
    if batch_id == "25_1":
        pol_names = ["base"]
    elif batch_id == "25_pol":
        pol_names = ["base", "information", "subsidies", "both"]
    elif batch_id == "26_pol" or "31" in batch_id or "final" in batch_id:
        pol_names = ["base", "information", "subsidies", "both", "base_30", "information_30", "subsidies_30", "both_30"]

    pol_names_array = np.array(pol_names)
    ind_array = np.zeros(len(agg_res["final_costs"])).astype(int)
    for i in ind_used.keys():
        ind_array[int(i[3:])] = ind_used[i]

    ys = [agg_res["final_costs"] / agg_res["total_house_size"], 
          agg_res["final_costs"], agg_res["total_adapted"], agg_res["utilities_adapted"],
          agg_res["barrier_adapted"], agg_res["drainage_adapted"]]
    y_labels = ["Total costs per m2 house size ($)", "Total final costs ($)", 
                "Total adapted households", "Total adapted utilities",
                "Total adapted barrier", "Total adapted drainage"]

    colors = np.zeros_like(ind_array).astype(str)
    runs_per_pol = int(sam.num_samples / sam.num_policies)
    colors = used_cols[np.floor(ind_array / runs_per_pol).astype(int)]
    pol_names_array_long = pol_names_array[np.floor(ind_array / runs_per_pol).astype(int)]

    return pol_names, pol_names_array_long, ind_array, ys, y_labels, runs_per_pol, colors

def policy_scatter_plots(batch_id, show = False, alpha = 0.05, col = "seagreen"):
    num_plots = len(sam.uncert_names)
    for pol in range(len(pol_names)):
        for y in range(len(ys)):
            if batch_id == "25_pol":
                rows = 6
                cols = 4
            elif batch_id == "26_pol" or "31" in batch_id or "final" in batch_id:
                rows = 8
                cols = 4
            fig, axs = plt.subplots(nrows=rows, ncols=cols, sharey=True)
            ax_ind = 0
            for i in range(rows):
                for j in range(cols):
                    if ax_ind < num_plots:
                        axs[i,j].set_xlabel(sam.uncert_names[ax_ind])
                        if sam.uncert_names[ax_ind] in log_scales or y_labels[y] == "Total final costs ($)":
                            axs[i,j].set_yscale("log")
                        if i == 0:
                            axs[i,j].set_ylabel(f"")
                        sample_values = sam.samples[sam.uncert_names[ax_ind]][ind_array]
                        axs[i,j].scatter(x=sample_values[np.where(np.floor(ind_array/runs_per_pol).astype(int) == pol)],
                                    y=ys[y][np.where(np.floor(ind_array/runs_per_pol).astype(int) == pol)],
                                         color = col, s= 5, alpha= alpha)

                    else:
                        axs[i,j].axis("off")
                    ax_ind += 1

            fig.set_figheight(22)
            fig.set_figwidth(16)
            plt.tight_layout()
            plt.savefig(f"results/figures/{batch_id}_scatter_{y_labels[y]}_{pol_names[pol]}.png", dpi=150)
            if show:
                plt.show()
            else:
                plt.close()
        print(f"Done with policy {pol_names[pol]}")
    return

def one_scatter_plot(axs, ind, uncert, outcome, batch_id, pol = None, alpha=0.05, col = "mediumseagreen", s=5):
    cols = np.array(["olive", "dodgerblue", "green", "cyan", "red", "m", 'orange', 'gold'])
    if uncert in log_scales:
        axs[ind].set_yscale("log")
    sample_values = sam.samples[uncert][ind_array]
    if pol != None:
        if type(pol) == int:
            sns.regplot(x=sample_values[np.where(np.floor(ind_array / runs_per_pol).astype(int) == pol)],
                        y=agg_res[outcome][np.where(np.floor(ind_array / runs_per_pol).astype(int) == pol)],
                        ax=axs[ind], color="black", scatter_kws={"color": col,
                                                                 "s": s, "alpha": alpha})
        elif type(pol) == list:
            for i in range(len(pol)):
                sns.regplot(x=sample_values[np.where(np.floor(ind_array / runs_per_pol).astype(int) == pol[i])],
                            y=agg_res[outcome][np.where(np.floor(ind_array / runs_per_pol).astype(int) == pol[i])],
                            ax=axs[ind], color="black", scatter_kws={"color": col,
                                                                     "s": s, "alpha": alpha})
    else:
        sns.regplot(x=sample_values, y=agg_res[outcome], ax=axs[ind], color="black", scatter_kws={"color": col,
                    "s": s, "alpha": alpha})
    axs[ind].set_xlabel(uncert)
    axs[ind].set_ylabel(outcome)
    return
def scatter_plots(batch_id, pols = False):
    num_plots = len(sam.uncert_names)
    for y in range(len(ys)):
        rows = 6
        cols = 4
        fig, axs = plt.subplots(nrows=rows, ncols=cols, sharey=True)
        ax_ind = 0
        for i in range(rows):
            for j in range(cols):
                if ax_ind < num_plots:
                    axs[i,j].set_xlabel(sam.uncert_names[ax_ind])
                    if sam.uncert_names[ax_ind] in log_scales or y_labels[y] == "Total final costs ($)":
                        axs[i,j].set_yscale("log")
                    if i == 0:
                        axs[i,j].set_ylabel(f"") #Check of dit nodig is

                    if not pols:
                        axs[i,j].scatter(x=sam.samples[sam.uncert_names[ax_ind]][ind_array],
                                    y=ys[y], color = "mediumseagreen", s= 5, alpha=  0.05)
                    else:
                        axs[i,j].scatter(x=sam.samples[sam.uncert_names[ax_ind]][ind_array],
                                    y=ys[y],
                                    color= colors, s=5, alpha=0.05)
                else:
                    axs[i,j].axis("off")
                ax_ind += 1

        fig.set_figheight(22)
        fig.set_figwidth(16)
        plt.tight_layout()
        plt.savefig(f"results/figures/{batch_id}_scatter_{y_labels[y]}.png", dpi=150)
        plt.show()
        print(f"Done with {y_labels[y]}")
        plt.close()
    return


def box_plots(batch_id, show = False):
    if len(ys) > 4 and len(ys) <= 6:
        fig, axs = plt.subplots(ncols=3, nrows = 2)
        ind = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        #[(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    else:
        fig, axs = plt.subplots(ncols=len(ys))
        ind = [*range(len(ys))]
        
    for y in range(len(ys)):
        
        pols = {}
        for i in range(sam.num_policies):
            pols[pol_names[i]] = ys[y][np.where(np.floor(ind_array / runs_per_pol).astype(int) == i)]
        sns.boxplot(pols, ax=axs[ind[y]])
        if y == 1:
            axs[ind[y]].set_yscale("log")
        if "adapted" in y_labels[y]:
            axs[ind[y]].set_ylim((-2,52))
        axs[ind[y]].set_ylabel(y_labels[y])
        axs[ind[y]].set_xticklabels(axs[ind[y]].get_xticklabels(), rotation=90)

    fig.set_figheight(6)
    fig.set_figwidth(10)
    plt.tight_layout()
    plt.savefig(f"results/figures/boxplot_{batch_id}")
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pairplot( batch_id = "26_pol", show = False):
    agg_res["policies"] = pol_names_arr
    agg_res["final_cost_per_m2"] = agg_res["final_costs"]/ agg_res["total_house_size"]
    pairplot = pd.DataFrame(agg_res)[["final_cost_per_m2",  "utilities_adapted",
                                      "barrier_adapted", "drainage_adapted", "total_adapted", "policies"]]

    g = sns.pairplot(pairplot, hue="policies", plot_kws={"s": 5})
    for i in range(5):
        for j in range(5):
            if i != j and i > 0:
                g.axes[i, j].set_ylim((-5,50))
            if i != j and j > 0:
                g.axes[i, j].set_xlim((-5, 50))
    plt.savefig(f"results/figures/pairplot_{batch_id}.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def plot_errorbars(batch_id):
    if batch_id == "25_1":
        fig = plt.figure()
        ind = [0]
        num_figs = 1
    elif batch_id == "25_pol":
        num_figs = 1
        ind = [(0,0), (0,1), (1,0), (1,1)]
        fig, ax = plt.subplots(nrows = 2, ncols = 2)
    elif batch_id == "26_pol":
        num_figs = 2
        ind = [(0,0), (0,1), (1,0), (1,1)]
        fig, ax = plt.subplots(nrows = 2, ncols = 2)

    plots = 0
    for j in range(len(pol_names)):
        if plots == 4:
            ax[0,0].set_ylabel("Total adapted households")
            ax[1,0].set_ylabel("Total adapted households")
            ax[1,0].set_xlabel("Average final costs per m2 ($)")
            ax[1,1].set_xlabel("Average final costs per m2 ($)")
            plt.tight_layout()
            plt.savefig(f"results/figures/errorbar_{i}_part0.png", dpi=150)
            plt.close()
            fig, ax = plt.subplots(nrows=2, ncols=2)
        f_costs = np.squeeze(final_costs_per_scen[:, np.where(np.floor(ind_array[:sam.num_samples] / runs_per_pol).astype(int) == j)])
        tot_adap = np.squeeze(total_adapted[:, np.where(np.floor(ind_array[:sam.num_samples] / runs_per_pol).astype(int) == j)])

        if batch_id == "25_1":
            plot_obj = plt
        else:
            plot_obj = ax[ind[j%4]]
            plot_obj.set_title(pol_names[j])
        plot_obj.errorbar(np.mean(f_costs, axis=0), np.mean(tot_adap, axis=0),
                    xerr=np.std(f_costs, axis=0), yerr=np.std(tot_adap, axis=0), linestyle='',
                     linewidth=0.3, color = 'mediumseagreen')

        if batch_id != "25_1":
            ax[ind[j%4]].set_xlim(0,650)
            ax[ind[j%4]].set_ylim(-5,50)
        plots += 1

    if i != "25_1":
        ax[0, 0].set_ylabel("Total adapted households")
        ax[1, 0].set_ylabel("Total adapted households")
        ax[1, 0].set_xlabel("Average final costs per m2 ($)")
        ax[1, 1].set_xlabel("Average final costs per m2 ($)")
    else:
        plt.ylabel("Total adapted households")
        plt.xlabel("Average final costs per m2")
        plt.title(f"run {i}")
    plt.tight_layout()
    plt.savefig(f"results/figures/errorbar_{i}.png", dpi=150)
    plt.close()


def mann_whitney_utest(outcome, prnt =False):
    res_dict = {}
    for i in range(len(pol_names)):
        if i%4 != 0:
            result = sc.stats.mannwhitneyu(
                                x=outcome[np.where(pol_names_arr == pol_names[i])] ,
                                y=outcome[np.where(pol_names_arr == pol_names[4*int(np.floor(i/4))])]
                            )
            if prnt:
                print(f"For policy {pol_names[i]} and {pol_names[4*int(np.floor(i/4))]}: {result.pvalue}")
            res_dict[pol_names[i]] = result.pvalue
    
    return res_dict

log_scales = [] #["p_cost", "actual_cost", "savings", "actual_barrier_cost", "actual_drainage_cost", "p_barrier_cost",
#               "p_drainage_cost"]

batch_ids = ["final_pol"]
for i in batch_ids:
    agg_res, ind_used, sam = load_data(i)
    pol_names, pol_names_arr, ind_array, ys, y_labels, runs_per_pol, colors = get_vars_from_batch_id(i)
    
    #Hoi Eline!
    #Dit volgende blokje is allemaal nodig voor de errobar plot
    
    
    # final_costs_per_scen = np.zeros((sam.num_runs_per_scen, sam.num_samples))
    # total_adapted = np.zeros_like(final_costs_per_scen)
    # for j in range(sam.num_runs_per_scen):
    #     final_costs_per_scen[j] = agg_res["final_costs"][j*sam.num_samples: (j+1)*sam.num_samples]/agg_res["total_house_size"][j*sam.num_samples: (j+1)*sam.num_samples]
    #     total_adapted[j] = agg_res["total_adapted"][j*sam.num_samples: (j+1)*sam.num_samples]
    # plot_errorbars(i)
    

    #scatter_plots(i, pols = True) Dit maakt die hele grote scatter plot, 
    # duurt even om te runnen
    
    
    plot_pairplot(i)
    
    #Dit blokje doet de statistical_tests
    tests = {}
    for i in range(len(ys)):
        tests[y_labels[i]] = mann_whitney_utest(ys[i], prnt=True)
    df_tests = pd.DataFrame(tests)
    df_tests.to_excel(f"results/utests_{i}.xlsx")

    #En dit de boxplots
    box_plots(i)

    