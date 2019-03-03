import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def compare_classifiers(models, data, y, silent=False, plot=False, args={
    'cv': 5,
    'scoring': 'accuracy',
    'n_jobs': 5
}):
    results = []
    for name, model in models:
        current_model_results = cross_val_score(model, data, y, **args)
        results.append(current_model_results)
        print(name, np.average(current_model_results))

    if plot:
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels([name for name, model in models])
        plt.show()

    return results
