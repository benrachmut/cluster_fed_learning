from Graph_global import *
from main_ import *


def switch_algo_and_seed_client_server(merged_dict, dich):
    rds = {}
    for seed in seeds_dict[dich][data_type]:
        for algo in merged_dict[seed]:
            algo_name = algo_names[algo]

            algo_name_list = get_PseudoLabelsClusters_name(algo,merged_dict[seed][algo])
            for name_ in algo_name_list:
                if name_ not in rds.keys() :
                    name_to_place = ""
                    if name_ == "MAPL,VGG":
                        name_to_place = "VGG"
                    else:
                        name_to_place = "AlexNet"

                    rds[name_to_place] = []
                rd_output = extract_rd_PseudoLabelsClusters_server_client(algo,merged_dict[seed][algo])#extract_rd(algo, )
                for k,v in rd_output.items():
                    if k not in rds:
                        rds[k]=[]
                    rds[k].append(v)
    return rds



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties

def build_grouped_legend(ax=None, *,
                         group_by="linestyle",   # "linestyle" or "label"
                         label_groups=("Server", "Client"),
                         where=("upper center", (0.5, -0.05)),
                         ncols=2,
                         frameon=False):
    """
    Build a grouped legend with two headings (e.g., Server / Client).
    - group_by="linestyle": maps solid->label_groups[0], dashed->label_groups[1]
    - group_by="label":     groups by substring match of label_groups in line labels
    """
    if ax is None:
        ax = plt.gca()

    lines = [l for l in ax.get_lines() if l.get_label() and not l.get_label().startswith('_')]

    def classify(line):
        if group_by == "linestyle":
            ls = (line.get_linestyle() or "-")
            # map common linestyles to the two buckets
            return label_groups[0] if ls in ("-", "solid") else label_groups[1]
        elif group_by == "label":
            lab = line.get_label().lower()
            if label_groups[0].lower() in lab:
                return label_groups[0]
            if label_groups[1].lower() in lab:
                return label_groups[1]
            # default bucket if neither found
            return label_groups[0]
        else:
            raise ValueError("group_by must be 'linestyle' or 'label'")

    buckets = {label_groups[0]: [], label_groups[1]: []}
    for ln in lines:
        buckets[classify(ln)].append(ln)

    # Build legend entries with bold headers as proxy artists
    header_fp = FontProperties(weight='bold')
    handles, labels = [], []
    for header in label_groups:
        # header row
        handles.append(Patch(alpha=0, linewidth=0))  # invisible patch as header proxy
        labels.append(header)
        # items under header
        for ln in buckets[header]:
            # create a proxy line that matches style/color/marker but short for legend
            proxy = Line2D([0], [0],
                           linestyle=ln.get_linestyle(),
                           linewidth=ln.get_linewidth(),
                           marker=ln.get_marker(),
                           markersize=ln.get_markersize(),
                           color=ln.get_color())
            handles.append(proxy)
            labels.append(ln.get_label())

    leg = ax.legend(handles, labels,
                    loc=where[0], bbox_to_anchor=where[1],
                    ncol=ncols, frameon=frameon, handlelength=2.5,
                    columnspacing=1.2, borderaxespad=0.0)

    # Style the header rows in bold
    for text, lab in zip(leg.get_texts(), labels):
        if lab in label_groups:
            text.set_fontproperties(header_fp)

    return leg

if __name__ == '__main__':

    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group

    all_data = read_all_pkls("diff_algo")
    merged_dict1 = merge_dicts(all_data)
    top_what_list = [1,5,10]
    data_type = DataSet.CIFAR100.name
    top_what = 1
    data_for_graph = {}
    for dich in [5]:
        merged_dict = merged_dict1[data_type][25][5][1][dich]
        merged_dict = switch_algo_and_seed(merged_dict,dich,data_type)
        new_name_dict = {}
        for k,v in merged_dict.items():
            new_name_dict[k]=v
        data_for_graph[dich]= collect_data_per_server_client_iteration(new_name_dict,top_what,data_type)
    print()

    # --- usage example with your code ---
    # after you make the plot:
    plt,the_plot = plot_model_server_client(data_for_graph)

    # If plot_model_server_client returns an Axes, pass it in; otherwise grab current axes.
    ax = the_plot if hasattr(the_plot, 'get_lines') else plt.gca()

    # Option A: group by linestyle (solid = Server, dashed = Client)
    build_grouped_legend(ax, group_by="linestyle", label_groups=("Server", "Client"),
                         where=("upper center", (0.5, -0.15)), ncols=2, frameon=False)
    plt.tight_layout()
    the_plot.savefig("figures/client_server_alpha.pdf", format="pdf")
    plt.show()
    # Option B: group by label keywords (if your line labels contain 'server'/'client')
    # build_grouped_legend(ax, group_by="label", label_groups=("Server", "Client"))
