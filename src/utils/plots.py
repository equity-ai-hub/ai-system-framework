import pathlib

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px


ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
PATH_TO_SAVE = f"{ROOT_PATH}/assets/baseline/"


RGB_COLORS_PASTEL = px.colors.qualitative.Pastel
RGB_COLORS_G10 = px.colors.qualitative.G10
TRANSPARENCY = "rgba(0,0,0,0)"
GRID_COLOR = "rgb(159, 197, 232)"
LINES_TYPES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
LINE_COLOR = "rgb(82, 82, 82)"
PAPER_BG_COLOR = "rgba(0,0,0,0)"
PLOT_BGCOLOR = "rgba(0,0,0,0)"

config = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",  # one of png, svg, jpeg, webp
        "scale": 6,  # Multiply title/legend/axis/canvas sizes by this factor
        "width": 500,
        "height": 500,
    },
}


def bold_title(title: str):
    return f"<b>{title}</b>"


def italic_title(title: str):
    return f"<i>{title}</i>"


def get_confidence_interval(scores):
    from scipy import stats

    mean = scores.mean()
    sem = stats.sem(scores)
    ci = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=sem)
    return ci


# =========== Plot with the metric and confidence interval for each model =========== #
def add_new_trace(fig: go.Figure, model_scores: pd.DataFrame, metric: str, trace_color: str, legend_name: str):
    """Add a new trace to the figure with the mean score and confidence interval for each model.

    Args:
        fig (go.Figure): Plotly figure.
        model_scores (pandas Dataframe) : Dataframe containing the scores of each model.
        metric (stt): Metric to plot. B or Precision. # Recall or F1-score.
        trace_color (str): Color of the trace.
        legend_name (str): Name of the trace.

    Returns:
        Plotly figure.
    """
    scores = model_scores.groupby(["model_name"])[[metric]]
    means, accs, upper_ci, lower_ci, models = [], [], [], [], []

    for model in scores.groups.keys():
        models.append(model)
        mean_acc = scores.get_group(model)[metric].mean()
        means.append(mean_acc)
        accs = scores.get_group(model)[metric]
        ci = get_confidence_interval(accs)
        lower_ci.append(np.abs(mean_acc - ci[0]))
        upper_ci.append(np.abs(mean_acc - ci[1]))

    fig.add_trace(
        go.Scatter(
            x=means,
            y=models,
            mode="markers",
            name=legend_name,
            marker=dict(color=trace_color, size=10),
            error_x=dict(array=upper_ci, arrayminus=lower_ci),
        )
    )
    return fig


def plot_score_ci(df: pd.DataFrame, metric: list, plot_title: str, dataset_name: str, export_to_image=False):
    """Plot the mean score and confidence interval for each model.

    Args:
        df (pandas Dataframe) : Dataframe containing the scores of each model.
        metric (stt): Metric to plot. Accuracy or Precision. # Recall or F1-score.
        plot_title (str): Title of the plot.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    # undersampling_scores = df[df["sampling_method"] == "undersampling"]
    # oversampling_scores = df[df["sampling_method"] == "oversampling"]
    no_sampling = df[df["sampling_method"] == "no_sampling"]

    metric_colors = [RGB_COLORS_PASTEL[0], RGB_COLORS_PASTEL[1], RGB_COLORS_PASTEL[2]]

    for m in metric:
        fig = add_new_trace(fig, no_sampling, m, metric_colors[metric.index(m)], legend_name=m)
    # fig = add_new_trace(fig, no_sampling, "bal_acc", metric_colors[1], legend_name="bal_acc")

    fig.update_layout(
        # autosize=True,
        font=dict(size=10, family="IBM Plex Sans, sans-serif"),
        title=dict(
            font=dict(size=13),
            text=bold_title(plot_title),
        ),
        xaxis_title=f"{metric}",
        yaxis_title="model",
        # yaxis_autorange="reversed",
        xaxis=dict(showticklabels=True, showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(showticklabels=True, showgrid=True, gridcolor=GRID_COLOR),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        # paper_bgcolor=PAPER_BG_COLOR,
        showlegend=True,
    )
    fig.show(config=config)
    if export_to_image:
        pathlib.Path(f"{PATH_TO_SAVE}/{dataset_name}/").mkdir(parents=True, exist_ok=True)
        fig.write_image(f"{PATH_TO_SAVE}/{dataset_name}/{plot_title}.png", width=1200, scale=6)


def plot_confusion_matrix(
    cm: np.array,
    dataset_name: str,
    labels: list,
    img_name="confusion_matrix",
    title: str = "Confusion Matrix",
    cmap: str = "bupu",
    export_to_image=False,
):
    """Plot the confusion matrix.

    Args:
        cm (numpy array): Confusion matrix.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
        cmap (str, optional): Color map. Defaults to "bupu".
        img_name (str, optional): Name of the image. Defaults to "confusion_matrix".

        NOTE: Thus in binary classification, the count of confusion matrix is:
        - true negatives is :math:`C_{0,0}`,
        - false negatives is :math:`C_{1,0}`,
        - true positives is :math:`C_{1,1}`
        - false positives is :math:`C_{0,1}`.

    Returns:
        Plotly figure.
    """

    tn, fp, fn, tp = cm.ravel()
    groups = np.array([[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]])
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        showscale=True,
        colorscale=cmap,
        annotation_text=groups,
    )
    fig.update_xaxes(title_text="Predicted label")
    fig.update_yaxes(title_text="True label")
    fig.update_layout(
        title=dict(
            font=dict(size=13),
            text=bold_title(title),
        ),
        yaxis_autorange="reversed",
        font=dict(size=9),
    )
    fig.show(config=config)
    if export_to_image:
        pathlib.Path(f"{PATH_TO_SAVE}/{dataset_name}/").mkdir(parents=True, exist_ok=True)
        fig.write_image(f"{PATH_TO_SAVE}/{dataset_name}/{img_name}.png", width=1200, scale=6)


def export_df_to_image(df, filename, dataset_name, agg=False):
    import dataframe_image as dfi

    pathlib.Path(f"{ROOT_PATH}/assets/baseline/{dataset_name}").mkdir(parents=True, exist_ok=True)

    if agg:
        dfi.export(
            df.style.highlight_max(
                subset=[
                    ("acc", "mean"),
                    ("bal_acc", "mean"),
                    ("f1", "mean"),
                    ("f1_macro", "mean"),
                    ("precision", "mean"),
                    ("recall", "mean"),
                    ("roc_auc", "mean"),
                ],
                color="rgb(161, 202, 228)",
            ),
            f"{PATH_TO_SAVE}/{dataset_name}/{filename}.png",
        )
    else:
        dfi.export(df, f"{PATH_TO_SAVE}/{dataset_name}/{filename}.png")


def _plot_group_bar_metrics(metrics_dict, metric_ci, interventions):
    fig = go.Figure()

    width = 0.15
    multiplier = 0

    for attribute, measurement in metrics_dict.items():
        offset = width * multiplier
        fig.add_trace(
            go.Bar(
                x=np.arange(len(interventions)) + offset,
                y=measurement,
                name=attribute,
                text=[f"{x:.2f}" for x in measurement],
                textposition="auto",
                width=width,
                # error_y=dict(
                #     type="data",
                #     visible=True,
                #     array=metric_ci[interventions[0]][0],
                #     arrayminus=metric_ci[interventions[0]][1],
                # ),
                marker=dict(color=RGB_COLORS_PASTEL[multiplier]),
                x0=5,
            )
        )
        multiplier += 1

    fig.update_layout(
        barmode="group",
        # bargap=bar_spacing,
        # autosize=True,
        font=dict(size=10, family="IBM Plex Sans, sans-serif"),
        title=dict(font=dict(size=13), text=bold_title("Metrics")),
        xaxis=dict(
            title="Overall metrics",
            tickvals=np.arange(len(interventions) + 1) + width,
            ticktext=interventions,
            tickmode="array",
            tickfont=dict(size=10),
            # zerolinecolor=GRID_COLOR,
            fixedrange=True,
        ),
        yaxis=dict(
            title="Metrics values",
            # range=([0, 1]),
            tickfont=dict(size=10),
            showticklabels=True,
            showgrid=True,
            gridcolor=GRID_COLOR,
            tickvals=np.arange(0, 1.1, 0.1),
            zerolinecolor=GRID_COLOR,
            fixedrange=True,
        ),
        margin=dict(l=60, r=50, t=100, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=0,
            y=1.15,
            orientation="h",
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="rgba(255, 255, 255, 0)",
        ),
    )
    fig.show(config=config)


def _grouped_bar_chart(metrics: dict, interventions: list, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(interventions))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for i, (attribute, measurement) in enumerate(metrics.items()):
        # ci = metrics_ci[attribute]
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            measurement,
            width,
            label=attribute,
            alpha=1,
            edgecolor="black",
            # yerr=[[measurement[0] - ci[0]], [ci[1] - measurement]],
        )
        ax.bar_label(rects, padding=1, fmt="%.3f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_ylabel("fairness scale")
    ax.set_xticks(x + width, interventions)
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    # ax.set_ylim(0, 1)
    ax.autoscale(enable=True, axis="y")
    # ax.set_yticks(np.arange(0.70, 0.9, step=0.02))

    for text in ax.texts:
        text.set_fontsize(9)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (5, 3)
    plt.show()


def grouped_bar_chart(metrics: dict, interventions: list, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(interventions))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for i, (attribute, measurement) in enumerate(metrics.items()):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            measurement,
            width,
            label=attribute,
            alpha=1,
            edgecolor="black",
        )
        ax.bar_label(rects, padding=1, fmt="%.3f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_ylabel("Metrics values")
    ax.set_xticks(x + width, interventions)
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    ax.autoscale(enable=True, axis="y")
    # ax.set_ylim(0, 1)
    # ax.set_yticks(np.arange(0.70, 0.9, step=0.02))

    for text in ax.texts:
        text.set_fontsize(9)
    plt.rcParams["figure.dpi"] = 100
    # plt.rcParams["savefig.dpi"] = 300
    # plt.rcParams["figure.figsize"] = (5, 3)
    plt.show()


def plot_group_bar_metrics(scores, interventions, title):
    import numpy as np
    import plotly.graph_objects as go

    import src.utils.plots_helper as ph

    fig = go.Figure()

    width = 0.2
    multiplier = 0

    for attribute, measurement in scores.items():
        # offset = width * multiplier
        fig.add_trace(
            go.Bar(
                x=np.arange(0, len(interventions) + 1),
                y=measurement,
                name=attribute,
                # text=[f"{x:.3f}" for x in measurement],
                textposition="auto",
                width=width,
                marker=dict(color=RGB_COLORS_PASTEL[multiplier]),
            )
        )
        print(measurement)
        multiplier += 1

    fig.update_layout(
        barmode="group",
        width=500,  # Set the width of the plot
        height=400,
        font=dict(size=10, family="IBM Plex Sans, sans-serif"),
        title=dict(font=dict(size=15), text=title),
        hovermode=False,
        xaxis=dict(
            title="Interventions",
            tickvals=np.arange(len(interventions) + 1),
            ticktext=interventions,
            tickmode="array",
            tickfont=dict(size=10),
            fixedrange=True,
            showgrid=False,
        ),
        yaxis=dict(
            title="Fairness Scale",
            tickfont=dict(size=12),
            showticklabels=True,
            tickvals=np.arange(0.05, 1.1, 0.05),
            zeroline=False,
            fixedrange=True,
            showgrid=False,
        ),
        margin=dict(l=50, r=50, t=50, b=50, pad=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        legend=dict(
            # x=-0.02,
            # y=1.1,
            # orientation="h",
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="rgba(255, 255, 255, 0)",
        ),
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="black",
                    width=1,
                ),
            )
        ],
    )
    fig.add_annotation(
        text="bias",
        x=-0.15,
        y=1,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="red"),
    )
    fig.add_annotation(
        text="fair",
        x=-0.15,
        y=0,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="red"),
    )
    fig.show(
        config=dict(
            displaylogo=False,
            toImageButtonOptions=dict(
                format="jpeg",
                scale=6,
                width=500,
                height=400,
            ),
        )
    )


def group_box_plots(group_type, test_set, test_prob, groups=None, group_names=None):
    fig = go.Figure()
    scores = test_prob

    if group_type == "sex":
        gender_map = {0: "Male", 1: "Female"}
        attr = test_set.sex.map(gender_map)
    elif group_type == "race":
        race_map = {
            "amer_indian_eskimo": "American Indian / Eskimo",
            "asian_pac_islander": "Asian / Pacific Islander",
            "black": "Black",
            "other": "Other",
            "white": "White",
        }
        attr = test_set.race.map(race_map)

    if groups is None:
        groups = np.zeros_like(scores)
        group_names = [""]

    unique_groups = sorted(set(groups))

    fig = go.Figure(
        data=[
            go.Box(
                x=scores[attr == a],
                y=groups[attr == a],
                name=a,
                orientation="h",
                marker={"color": RGB_COLORS_PASTEL[i], "opacity": 0.5},
                line_color=RGB_COLORS_PASTEL[i % len(RGB_COLORS_PASTEL)],
                hoverinfo="name+x",
                jitter=0.2,
            )
            for i, a in enumerate(sorted(set(attr)))
        ]
    )

    fig.update_layout(
        boxmode="group",
        height=200 + 40 * len(set(attr)) * len(set(groups)),
        hovermode="closest",
    )

    fig.update_xaxes(
        hoverformat=".3f",
        fixedrange=True,
        gridcolor=GRID_COLOR,
        title_text="Score",
    )

    fig.update_yaxes(
        tickvals=unique_groups,
        ticktext=group_names or unique_groups,
        fixedrange=True,
    )
    # pathlib.Path(f"{ROOT_PATH}/assets/").mkdir(parents=True, exist_ok=True)
    # fig.write_image(f"{ROOT_PATH}/assets/group_box_plots.png", width=800, scale=6)
    fig.show(config=config)


def plot_conditional_separation(scores_sep, scores_base, title):
    # https://plotly.com/python/marker-style/
    import plotly.graph_objects as go
    import src.utils.plots_helper as ph

    scatter_x = scores_base.index.tolist()
    scatter_y = scores_base.values.tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            name="baseline measure",
            marker=dict(size=20, symbol="line-ew", line=dict(width=2, color="DarkSlateGrey")),
        )
    )
    fig.add_trace(
        go.Bar(
            x=scores_sep.index.tolist(),
            y=scores_sep.values.tolist(),
            # name=scores_sep.index.tolist(),
            textposition="auto",
            width=0.6,
            marker=dict(color=RGB_COLORS_PASTEL),
        )
    )

    fig.update_layout(
        barmode="group",
        width=500,  # Set the width of the plot
        height=400,
        font=dict(size=10, family="IBM Plex Sans, sans-serif"),
        title=dict(font=dict(size=15), text="Fairness Measures XGBoost model"),
        hovermode=False,
        xaxis=dict(
            title="Measure by Sensitive Attribute",
            tickfont=dict(size=10),
            fixedrange=True,
            showgrid=False,
        ),
        yaxis=dict(
            title="Fairness Scale",
            tickfont=dict(size=12),
            showticklabels=True,
            # tickvals=np.arange(0, 1.1, 0.1),
            zeroline=True,
            fixedrange=True,
            showgrid=True,
        ),
        margin=dict(l=50, r=50, t=80, b=50, pad=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="rgba(255, 255, 255, 0)",
        ),
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="black",
                    width=1,
                ),
            )
        ],
    )
    fig.show(
        config=dict(
            displaylogo=False,
            toImageButtonOptions=dict(
                format="jpeg",
                scale=6,
                width=500,
                height=400,
            ),
        )
    )


def calibration_curves(labels, scores, attr, title="", xlabel="", ylabel="", n_bins=10):
    from src.utils.helper import calibration_probabilities

    COLORS = ["#f2603b", "#262445", "#00a886", "#edc946", "#70cfcf"]
    LINE_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
    GRID_COLOR = "rgb(159, 197, 232)"

    def _hex_to_rgba(hex_code, a=0.3):
        def cast(s):
            return int(s, 16)

        r = cast(hex_code[1:3])
        g = cast(hex_code[3:5])
        b = cast(hex_code[5:7])
        return f"rgba({r},{g},{b},{a})"

    bins = np.linspace(0, 1, n_bins + 1)
    x = (bins[1:] + bins[:-1]) / 2

    # Outline colours predefined for adjustability
    x_grid_color = GRID_COLOR
    y_grid_color = GRID_COLOR
    x_zero_line_color = x_grid_color
    y_zero_line_color = y_grid_color

    return go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=calibration_probabilities(labels[attr == a], scores[attr == a], n_bins),
                name=a,
                mode="lines+markers",
                line={"dash": LINE_STYLES[i % len(LINE_STYLES)]},
                marker={
                    "color": _hex_to_rgba(COLORS[i], 1),
                    "symbol": i,
                    "size": 10,
                },
            )
            for i, a in enumerate(sorted(set(attr)))
        ],
        layout={
            "autosize": True,
            "hovermode": "closest",
            "title": title,
            "xaxis": {
                "hoverformat": ".3f",
                "title": xlabel,
                "gridcolor": x_grid_color,
                "zerolinecolor": x_zero_line_color,
                "fixedrange": True,
            },
            "yaxis": {
                "title": ylabel,
                "gridcolor": y_grid_color,
                "zerolinecolor": y_zero_line_color,
                "fixedrange": True,
            },
        },
    )


def create_custom_boxplot(data_lists, labels, legend_labels=["Females", "Males"]):

    fig, ax = plt.subplots(figsize=(6, 4))
    num_groups = len(data_lists)
    group_width = 1
    box_width = group_width / 4

    positions = np.arange(num_groups)

    box_properties = {
        "patch_artist": True,
        "showfliers": False,
        "medianprops": {"color": "black"},
        "whiskerprops": {"color": "black"},
        "capprops": {"color": "black"},
        "flierprops": {"markeredgecolor": "black"},
    }

    for i, (data1, data2) in enumerate(data_lists):
        bp1 = ax.boxplot([data1], positions=[positions[i] - box_width / 2], widths=box_width, **box_properties)
        bp2 = ax.boxplot([data2], positions=[positions[i] + box_width / 2], widths=box_width, **box_properties)

        for patch in bp1["boxes"]:
            patch.set_facecolor(box_colors(3))
        for patch in bp2["boxes"]:
            patch.set_facecolor(box_colors(4))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("false positive rates")
    ax.set_ylabel("error rates scale")

    # ax.legend(handles=legend_elements, title="groups", loc="upper left")
    ax.legend(labels=legend_labels, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title("False positive rates across groups")
    plt.tight_layout()
    plt.show()


# Example usage:
# data_lists = [
#     ([list_for_yes_thu], [list_for_no_thu]),
#     ([list_for_yes_fri], [list_for_no_fri]),
#     ([list_for_yes_sat], [list_for_no_sat]),
#     ([list_for_yes_sun], [list_for_no_sun])
# ]
# labels = ['Thur', 'Fri', 'Sat', 'Sun']
# create_custom_boxplot(data_lists, labels

# performance = ["BAL_ACC", "RECALL", "PRECISION"]
# fairness = ["FPR", "FNR", "FDR", "FOR"]
# interventions = ["baseline", "separation", "independence"]

# from src.utils.plots import plot_group_bar_metrics

# baseline_fair_means = df_base[fairness].mean()
# separation_fair_means = df_sep[fairness].mean()
# independence_fair_means = df_ind[fairness].mean()

# independence_fair_means

# metrics = dict(zip(fairness, zip(baseline_fair_means, separation_fair_means, independence_fair_means)))
# metrics

# plot_group_bar_metrics(scores=metrics, interventions=interventions, title="")

# def plot_separation_overall_scores(scores, title):
#     import numpy as np
#     import plotly.graph_objects as go

#     import src.utils.plots_helper as ph

#     fig = go.Figure()

#     width = 0.15
#     multiplier = 0
#     interventions = ["baseline", "separation"]

#     for attribute, measurement in scores.items():
#         offset = width * multiplier
#         fig.add_trace(
#             go.Bar(
#                 x=np.arange(0, len(interventions) + 1),
#                 y=measurement,
#                 name=attribute,
#                 # text=[f"{x:.3f}" for x in measurement],
#                 textposition="auto",
#                 width=width,
#                 marker=dict(color=RGB_COLORS_PASTEL[multiplier]),
#             )
#         )
#         print(measurement)
#         multiplier += 1

#     fig.update_layout(
#         barmode="group",
#         width=500,  # Set the width of the plot
#         height=400,
#         font=dict(size=10, family="IBM Plex Sans, sans-serif"),
#         title=dict(font=dict(size=15), text=title),
#         hovermode=False,
#         xaxis=dict(
#             title="Interventions",
#             tickvals=np.arange(len(interventions) + 1),
#             ticktext=interventions,
#             tickmode="array",
#             tickfont=dict(size=10),
#             fixedrange=True,
#             showgrid=False,
#         ),
#         yaxis=dict(
#             title="Fairness Scale",
#             tickfont=dict(size=12),
#             showticklabels=True,
#             tickvals=np.arange(0, 1.1, 0.1),
#             zeroline=True,
#             fixedrange=True,
#             showgrid=False,
#         ),
#         margin=dict(l=50, r=50, t=80, b=50, pad=0),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="white",
#         legend=dict(
#             x=-0.02,
#             y=1.1,
#             orientation="h",
#             bgcolor="rgba(255, 255, 255, 0)",
#             bordercolor="rgba(255, 255, 255, 0)",
#         ),
#         shapes=[
#             dict(
#                 type="rect",
#                 xref="paper",
#                 yref="paper",
#                 x0=0,
#                 y0=0,
#                 x1=1,
#                 y1=1,
#                 line=dict(
#                     color="black",
#                     width=1,
#                 ),
#             )
#         ],
#     )
#     fig.add_annotation(
#         text="fair",
#         x=-0.07,
#         y=1,
#         xref="paper",
#         yref="paper",
#         showarrow=False,
#         font=dict(size=12, color="red"),
#     )
#     fig.add_annotation(
#         text="unfair",
#         x=-0.1,
#         y=0,
#         xref="paper",
#         yref="paper",
#         showarrow=False,
#         font=dict(size=12, color="red"),
#     )
#     fig.show(
#         config=dict(
#             displaylogo=False,
#             toImageButtonOptions=dict(
#                 format="jpeg",
#                 scale=6,
#                 width=500,
#                 height=400,
#             ),
#         )
#     )

# FAIR_SEP = ["AVG_ODDS_DIFF", "FPR", "FNR", "FDR", "FOR"]

# base_means = df_base[FAIR_SEP].mean()
# sep_means = df_sep[FAIR_SEP].mean()

# metrics_sep = dict(zip(FAIR_SEP, zip(base_means, sep_means)))
# plot_separation_overall_scores(metrics_sep, title="Xgboost Model | Fairness Measures Overall Scores")
