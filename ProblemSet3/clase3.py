
import pandas as pd
import numpy as np

import sklearn.model_selection as sk_ms
import sklearn.svm as sk_svm
import sklearn.tree as sk_tree
import sklearn.neighbors as sk_neig

class AnalistaDeRiesgo():
    def __init__(self,models,scoring):
        self.models = models
        self.scoring = scoring

    def load_data(self,historic_data,default_name):
        self.historic_data = historic_data
        self.default_name = default_name

    def get_report(self,current_data,features_list):

        best_score = 0
        best_classifier = []

        X_tr = self.historic_data[features_list]
        y_tr = self.historic_data[self.default_name]

        X = current_data[features_list]
        y = current_data[self.default_name]

        for model, params in self.models:
            model = sk_ms.GridSearchCV(model,params, cv=5, scoring=self.scoring)
            model.fit(X_tr,y_tr)
            model_best = model.best_estimator_
            if model.best_score_ > best_score:
                best_score = model.best_score_
                best_classifier = model_best

        y_p = best_classifier.predict(X)

        percentage = np.count_nonzero(y_p == 1) * 100 / len(y_p)
        TA = (10**X['log_TA']).sum()
        ADR = (10**X.loc[y_p == 1]['log_TA']).sum()
        print("Entidades en riesgo de default = " + str('{0:.2f}'.format(percentage)) + '%')
        print("Total de activos del sistema (USD B): " + str('{0:.3f}'.format(TA/1e3)))
        print("Porcentaje de activos en riesgo de default: "+ str('{0:.2f}'.format(100*ADR/TA)) + "%")

if __name__ == '__main__':
    historic_data = pd.read_hdf("central_bank_data.h5","bank_defaults_FDIC")
    current_data = pd.read_hdf("central_bank_data.h5","regulated_banks")

    # %% Entrenamos KNN

    param_dict_knn = {'n_neighbors': range(3, 30),
                      'metric': ['euclidean', 'manhattan']}

    param_dist_svc = {'C': [1,10,100,500,1000],
                      'kernel': ['rbf','linear']}

    param_grid_tree = {'min_samples_split': range(2, 15)}

    models = [
        (sk_neig.KNeighborsClassifier(),param_dict_knn),
        (sk_svm.SVC(),param_dist_svc),
        (sk_tree.DecisionTreeClassifier(),param_grid_tree)
    ]

    scoring = 'roc_auc'

    analyst = AnalistaDeRiesgo(models,scoring)
    analyst.load_data(historic_data,"defaulter")

    analyst.get_report(current_data,['log_TA','NI_to_TA','Equity_to_TA','NPL_to_TL',
                                     'REO_to_TA','ALLL_to_TL','core_deposits_to_TA',
                                     'brokered_deposits_to_TA','liquid_assets_to_TA',
                                     'loss_provision_to_TL','NIM','assets_growth'])

