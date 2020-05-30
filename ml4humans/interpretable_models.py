from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResultsWrapper
import matplotlib.pyplot as plt
import seaborn as sns


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

def effect_plot(model: LinearRegression, ds):
    pass

def effect_plot_for_example(model, ds, example):
    pass