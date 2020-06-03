from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResultsWrapper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def weight_plot(model_results: RegressionResultsWrapper):
    summary = model_results.summary2().tables[1]
    summary['abs.t'] = summary['t'].abs()
    summary = summary.sort_values('abs.t', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.despine(fig, left=True, bottom=True)
    for i, (coef, row) in enumerate(summary.iterrows()):
        # plot the points
        ax.plot(row[['[0.025', 'Coef.', '0.975]']], [i, i, i], 'ko-', ms=5., lw=2., markevery=[1])
        # add the vertical markers
        ax.vlines(row['[0.025'], i - 0.15, i + 0.15)
        ax.vlines(row['0.975]'], i - 0.15, i + 0.15)
        ax.annotate("%.2f" % row['Coef.'], (row['Coef.'], i), xytext=(-6, 4), textcoords='offset points')
    # add the horizontal lines
    ax.hlines(list(range(len(summary))), [ax.get_xlim()[0]] * len(summary), summary['[0.025'], colors='lightgray',
              linestyle='--')
    ax.xaxis.set_visible(False)

    # add the y labels
    ax.set_yticks(list(range(len(summary))))
    ax.set_yticklabels(summary.index)
    ax.vlines([0], -1, len(summary), colors='k', linestyle='--')
    ax.set_title("Weight plot", loc='left', size=18, pad=-20)

def effect_plot(model: RegressionResultsWrapper, ds:pd.DataFrame):
    effects = model.params * ds
    feature_points_map = []
    for feat in model.params.index:
        feat_points = {
            'feat': feat,
            'lower': np.percentile(effects[feat], 2.5),
            'median': np.percentile(effects[feat], 50),
            'upper': np.percentile(effects[feat], 97.5)}
        feature_points_map.append(feat_points)
    feature_points_df = pd.DataFrame(feature_points_map).sort_values('median').set_index('feat')
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.despine(fig, bottom=True)
    for i, (feat, row) in enumerate(feature_points_df.iterrows()):
        ax.plot(row[['lower', 'median', 'upper']], [i, i, i], 'ko-', ms=5., lw=2., markevery=[1])
        ax.vlines(row['lower'], i - 0.15, i + 0.15)
        ax.vlines(row['upper'], i - 0.15, i + 0.15)
        ax.annotate("%.2f" % row['median'], (row['median'], i), xytext=(-6, 4), textcoords='offset points')

    ax.hlines(list(range(len(feature_points_df))), [ax.get_xlim()[0]] * len(feature_points_df), feature_points_df['lower'], colors='lightgray',
              linestyle='--')
    ax.xaxis.set_visible(False)
    ax.set_yticks(list(range(len(feature_points_df.index))))
    ax.set_yticklabels(feature_points_df.index)
    ax.vlines([0], -1, len(feature_points_df.index), colors='k', linestyle='--')
    ax.set_title("Effect plot", loc='left', size=18, pad=-20)
    return ax, feature_points_df


def effect_plot_for_example(model, ds, example):
    ax, feature_points_df = effect_plot(model, ds)
    example_effects = model.params * example
    pred = example_effects.sum()
    ax.set_title(f"prediction: {pred: .2f}")
    for i, param in enumerate(feature_points_df.index):
        ax.plot(example_effects[param], i, 'rx', ms=20)
        ax.annotate("%.2f" % example_effects[param], (example_effects[param], i), xytext=(-6, -20), textcoords='offset points', color='r',size=14)