import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate


def compare_classifiers(models, data, y, silent=False, plot=False, args={
    'cv': 5,
    'scoring': 'accuracy',
    'n_jobs': 5
}):
    results = []
    for name, model in models:
        current_model_results = cross_validate(
            model,
            data,
            y,
            return_train_score=True,
            **args)

        results.append(current_model_results['test_score'])
        if not silent:
            train_score = np.average(current_model_results['train_score'])
            test_score = np.average(current_model_results['test_score'])
            info = f'train: {train_score}, test: {test_score}',
            print(name, info, flush=True)

    if plot:
        fig = plt.figure()
        # fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels([name for name, model in models])
        plt.show()

    return results
