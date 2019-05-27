import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression


def compare_classifiers(models, data, y, silent=False, plot=False, args={
    'cv': 5,
    'scoring': ['accuracy', 'f1_macro'],
    'n_jobs': 5
}):
    results = []
    for name, model in models:
        # current_model_results = cross_validate(
        #     model,
        #     data,
        #     y,
        #     return_train_score=True,
        #     **args)

        # pred = cross_val_predict(
        #     model,
        #     data,
        #     y,
        #     cv=5,
        #     method='predict_proba')

        current_model_results = cross_validate(
            model,
            data,
            y,
            return_train_score=True,
            **args)

        if not silent:
            train_acc = np.average(current_model_results['train_accuracy'])
            test_acc = np.average(current_model_results['test_accuracy'])
            train_f1 = np.average(current_model_results['train_f1_macro'])
            test_f1 = np.average(current_model_results['test_f1_macro'])
            info = (f'{train_acc}\t{test_acc}\t{train_f1}\t{test_f1}')

            print(f'{info}', flush=True)

    if plot:
        fig = plt.figure()
        # fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels([name for name, model in models])
        plt.show()

    return results
