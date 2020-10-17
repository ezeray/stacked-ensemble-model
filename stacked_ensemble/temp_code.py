# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from time import time
import pickle
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold
)
import base_funcs.BaseHelpFuncs as hf


class StackedEnsembleModel():
    def __init__(
        self,
        base_models=None,
        base_hyper_param_tuners=None,
        preprocessor=None,
        meta_model=None,
        meta_hyper_param_tuners=None,
        rand_state=100,
        n_folds=10,
        scorer='accuracy'
    ):
        if isinstance(base_models, dict):
            self.base_models = base_models
        else:
            raise TypeError(
                'The models object should be a dictionary specifying model '
                'name and the actual model object.'
            )

        if base_hyper_param_tuners:
            if isinstance(base_hyper_param_tuners, dict):
                self.base_hyper_param_tuners = base_hyper_param_tuners
            else:
                raise TypeError(
                    'The hyper_param_tuners object should be a dictionary '
                    'specifying the model name and then the grid tune as '
                    'it would be passed to sklearn\'s GridSearchCV.'
                )
        else:
            self.base_hyper_param_tuners = None

        self.meta_model = meta_model
        self.meta_hyper_param_tuners = meta_hyper_param_tuners
        self.preprocessor = preprocessor

        if isinstance(rand_state, int):
            self.rand_state = rand_state
            np.random.seed(self.rand_state)
        elif rand_state is None:
            self.rand_state = None
        else:
            raise TypeError('rand_state should be an integer')

        if isinstance(n_folds, int):
            self.n_folds = n_folds
        else:
            raise TypeError('n_folds should be an integer')

        self.scorer = scorer

        self.is_stacked = {}
        self.fit_stacker_models_estimators = {}

        for k in self.base_models.keys():
            self.is_stacked[k] = False
            self.fit_stacker_models_estimators[k] = []
        self.is_tuned = {}
        for k in self.base_hyper_param_tuners.keys():
            self.is_tuned[k] = False

        self.stacker_models_are_fit = {}

        self.stacked_train_features = []
        self.stacked_train_labels = []
        self.stacked_names = []
        self.stacked_test_features = []
        self.preprocessor_is_fit = False
        self.stacker_is_fit = False
        self.meta_model_is_fit = False

    def preprocess_fit_transform(self, X):
        if self.preprocessor is None:
            raise TypeError('No preprocessor was passed so cannot fit and '
                            'transform any data.'
                            )

        X = hf.check_input_data_type(X)

        self.preprocess_fit(X)
        return (
            self.preprocess_transform(X)
        )

    def preprocess_fit(self, X):
        if self.preprocessor is None:
            raise TypeError('No preprocessor was passed so cannot fit '
                            'any data.'
                            )
        X = hf.check_input_data_type(X)
        if not self.preprocessor_is_fit:
            self.preprocessor.fit(X)
            self.preprocessor_is_fit = True

    def preprocess_transform(self, X):
        if self.preprocessor is None:
            raise TypeError('No preprocessor was passed so cannot '
                            'transform any data.'
                            )
        X = hf.check_input_data_type(X)
        if self.preprocessor_is_fit:
            return self.preprocessor.transform(X)
        else:
            raise AttributeError(
                'The preprocessor has not been fit so cannot apply transform.'
            )

    def base_tune_params_fit(
        self, X_train, y_train, additional_grid_arguments={}
    ):
        if self.base_hyper_param_tuners is None:
            raise Exception(
                'No parameters were passed for tuning.'
            )

        X_train = hf.check_input_data_type(X_train)
        y_train = hf.check_input_data_type(y_train, one_dim=True)

        logger = hf.create_logger('Hyper-Param-Tuning')
        logger.info('Tuning for hyperparameters has started.')
        tune_start = time()

        models_for_tuning = list(self.base_hyper_param_tuners.keys())
        logger.info(
            f'Will be tuning hyper-parameters for {len(models_for_tuning)}'
        )

        models_missing = [miss for miss in models_for_tuning
                          if miss not in self.base_models.keys()]
        if len(models_missing) > 0:
            logger.error(
                'Models were passed for tuning that are not present in the '
                'base models dictionary. Check spelling and make sure to '
                'include.'
            )
            raise ValueError(
                'Models were passed for tuning that are not present in '
                'the models dictionary. Check spelling and make sure to '
                'include.'
            )

        logger.info('Will now start fitting the grid search.\n')

        for m in models_for_tuning:
            logger.info(f'Tuning hyper-parameters for {m} using '
                        f'scorer {self.scorer}')
            if not self.is_tuned[m]:
                try:
                    grid_search = GridSearchCV(
                        self.base_models[m],
                        param_grid=self.base_hyper_param_tuners[m],
                        cv=self.n_folds,
                        scoring=self.scorer,
                        **additional_grid_arguments
                    )
                    fit_start = time()
                    grid_search.fit(X_train, y_train)
                    fit_end = time()
                    fit_time = hf.convert_time(fit_end - fit_start)
                    logger.info(f'The fit took {fit_time}')

                    best_estimator_str = '\t' +\
                        grid_search.best_estimator_.__str__().replace(
                            '\n', '\n\t')
                    best_params_str = '\t' +\
                        grid_search.best_params_.__str__().replace(
                            '\n', '\n\t')
                    best_score = float(grid_search.best_score_)
                    best_score_std = float(
                        grid_search.cv_results_['std_test_score']
                        [grid_search.best_index_]
                    )
                    train_score = grid_search.score(X_train, y_train)

                    logger.info(f'The best estimator found is:\n'
                                f'{best_estimator_str}')
                    logger.info(f'The best parameters found are:\n'
                                f'{best_params_str}')
                    logger.info(f'The best score found in the grid search:\n'
                                f'\t{best_score:0.2f} with std '
                                f'{best_score_std:0.2f}')
                    logger.info(f'The score of the estimator on the complete '
                                f'training set is:\n\t{train_score:0.2f}')

                    best_params = grid_search.best_estimator_.get_params()
                    self.base_models[m].set_params(**best_params)
                    logger.info(f'The base model for {m} has been updated '
                                f'with the new parameters.\n\n')
                    self.is_tuned[m] = True
                except Exception:
                    logger.exception(
                        f'Error encountered when tuning for {m}.\n\n'
                    )
            else:
                logger.info(f'Model {m} had previously been tuned.')
        tune_end = time()
        tune_timer = hf.convert_time(tune_end - tune_start)
        logger.info(f'The copmlete tuning process took {tune_timer}')
        logging.shutdown()

    def stacker_fit_predict(self, X_train, X_test, y_train):
        # cuando sea posible es preferible usar este método
        # porque este hace un trabajo más fino
        #
        # los resultados que van a pasar a ser los features
        # es decir, la 'metadata', son las predicciones que se
        # haga sobre el validation set, NO sobre el total

        X_train = hf.check_input_data_type(X_train)
        X_test = hf.check_input_data_type(X_test)
        y_train = hf.check_input_data_type(y_train, one_dim=True)

        logger = hf.create_logger('Stacker-Model')
        logger.info('Stacking has started.')
        stack_start = time()
        models_for_stacking = list(self.base_models.keys())
        logger.info(
            f'The ensemble feature space will be built with the predicted '
            f'probabilities from {len(models_for_stacking)} different '
            f'classifiers.'
        )
        skf = StratifiedKFold(
                n_splits=self.n_folds
        )

        for i, m in enumerate(models_for_stacking):
            logger.info(f'Will now fit and generate output for {m}.')
            if not self.is_stacked[m]:
                try:
                    fit_start = time()
                    validation_pred = []
                    true_val = []
                    test_pred = []
                    estimator = self.base_models[m]
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        trainer_X = X_train[train_idx]
                        val_X = X_train[val_idx]
                        trainer_y = y_train[train_idx]
                        val_y = y_train[val_idx]

                        estimator.fit(X=trainer_X, y=trainer_y)
                        self.fit_stacker_models_estimators[m].append(estimator)

                        validation_pred.append(
                            estimator.predict_proba(val_X)[:, 1]
                        )

                        test_pred.append(
                            estimator.predict_proba(X_test)[:, 1]
                            .reshape(-1, 1)
                        )

                        if i == 0:
                            true_val.append(val_y)

                    estimator.fit(X_train, y_train)

                    fit_end = time()
                    fit_timer = fit_end - fit_start
                    logger.info(
                        f'The stacking process for {m} took {fit_timer}\n'
                    )

                    self.stacked_train_features.append(
                        np.hstack(validation_pred).reshape(-1, 1)
                    )

                    test_pred = np.mean(
                        np.hstack(test_pred),
                        axis=1
                    )
                    self.stacked_test_features.append(
                        test_pred.reshape(-1, 1)
                    )

                    if i == 0:
                        self.stacked_train_labels = np.hstack(true_val)

                    self.stacker_models_are_fit[m] = estimator
                    self.stacked_names.append(m)
                    self.is_stacked[m] = True
                except Exception:
                    logger.exception(
                        f'Error encountered when stacking for {m}'
                    )
            else:
                logger.info(f'Model {m} was previously stacked.')

        self.stacker_is_fit = True
        stack_end = time()
        stack_timer = hf.convert_time(stack_end - stack_start)
        logger.info(f'The copmlete stacking process took {stack_timer}')
        logging.shutdown()

    def get_stacked_results(self, return_df=False):
        if not all(self.is_stacked.values()):
            # raise Error
            pass
        if return_df:
            train_data_temp = self.stacked_train_features.copy()
            train_data_temp.append(self.stacked_train_labels.reshape(-1, 1))
            train = pd.DataFrame(
                data=np.hstack(
                    train_data_temp
                ),
                columns=self.stacked_names + ['label']
            )
            test = pd.DataFrame(
                data=np.hstack(
                    self.stacked_test_features
                ),
                columns=self.stacked_names
            )
            return train, test
        else:
            return (
                self.stacked_names,
                np.hstack(self.stacked_train_features),
                self.stacked_train_labels,
                np.hstack(self.stacked_test_features)
            )

    def stacker_fit(self, X_train, y_train):
        # este hace un fit sin devolver nada
        # todavía no la terminé porque no me hace falta usarla
        # TERMINAR
        X_train = hf.check_input_data_type(X_train)
        y_train = hf.check_input_data_type(y_train, one_dim=True)

        # self.stacker_is_fit = True

    def stacker_predict(self, X_test, return_df=False):
        # este viene a funcionar para tratar data completamente nueva
        # sin tener que volver a ajustar los modelos
        # la gran diferencia es que en el método stacker, las predicciones
        # que se calculan sobre el test set son el resultado del promedio
        # de las predicciones en cada uno de las iteraciones del k-fold
        # por modelo. en este caso, es solo la predicción simple.
        #
        # la principal utilidad de este método sería en un caso donde
        # haya mucha data de entrenamiento y no sea factible reutilizarla
        # por ejemplo, si se hizo un deployment y solo hace falta procesar
        # los datos para usarlos en el metaclasificador

        X_test = hf.check_input_data_type(X_test)

        not_yet_fit = [mod for mod in self.base_models.keys()
                       if mod not in self.stacker_models_are_fit.keys()]
        if len(not_yet_fit) > 0:
            raise AttributeError('Missing fit models')

        names = []
        predictions = []

        for mod, estim_list in self.fit_stacker_models_estimators.items():
            names.append(mod)
            est_predictions = []
            for sub_estim in estim_list:
                est_predictions.append(
                    sub_estim.predict_proba(X_test)[:, 1].reshape(-1, 1)
                )
            est_predictions_mean = np.mean(
                np.hstack(est_predictions),
                axis=1
            )
            predictions.append(est_predictions_mean)

        if return_df:
            return (
                pd.DataFrame(
                    data=np.hstack(predictions),
                    columns=names
                ),
                names
            )
        else:
            return (
                np.hstack(predictions),
                names
            )

    def meta_model_tune(self, X_train, y_train, verbosity):
        logger = hf.create_logger('Meta-Mdodel-Tuner')
        logger.info('Started tuning hyperparameters for the metamodel')
        tune_start = time()
        if self.meta_hyper_param_tuners is None:
            logger.error('No hyper param grid was passed for the meta model')
            raise ValueError(
                'No hyper param grid was passed for the meta model.'
            )
        if self.meta_model is None:
            logger.error(
                'No meta model was passed so cannot tune hyper parameters '
                'for it.'
            )
            raise ValueError(
                'No meta model was passed so cannot tune hyper parameters '
                'for it.'
            )
        X_train = hf.check_input_data_type(X_train)
        y_train = hf.check_input_data_type(y_train, one_dim=True)

        try:
            meta_grid_search = GridSearchCV(
                self.meta_model,
                param_grid=self.meta_hyper_param_tuners,
                cv=self.n_folds,
                verbose=verbosity
            )
            fit_start = time()
            meta_grid_search.fit(X_train, y_train)
            fit_end = time()
            fit_timer = hf.convert_time(fit_end - fit_start)
            logger.info(
                f'Fitting the grid search for the meta model took {fit_timer}'
            )
            best_estimator_str = '\t' +\
                meta_grid_search.best_estimator_.__str__().replace(
                    '\n', '\n\t')
            best_params_str = '\t' +\
                meta_grid_search.best_params_.__str__().replace(
                    '\n', '\n\t')
            best_score = float(meta_grid_search.best_score_)
            best_score_std = float(
                meta_grid_search.cv_results_['std_test_score']
                [meta_grid_search.best_index_]
            )
            train_score = meta_grid_search.score(X_train, y_train)

            logger.info(f'The best estimator found is:\n'
                        f'{best_estimator_str}')
            logger.info(f'The best parameters found are:\n'
                        f'{best_params_str}')
            logger.info(f'The best score found in the grid search:\n'
                        f'\t{best_score:0.2f} with std '
                        f'{best_score_std:0.2f}')
            logger.info(f'The score of the estimator on the complete '
                        f'training set is:\n\t{train_score:0.2f}')

            best_params = meta_grid_search.best_estimator_.get_params()
            self.meta_model.set_params(**best_params)
            logger.info(
                'The meta model has been updated with the best params.'
            )
        except Exception:
            logger.exception('An error occurred when tuning the meta model.')

        tune_end = time()
        tune_timer = hf.convert_time(tune_end - tune_start)
        logger.info(f'Tuning the meta model took {tune_timer}.')
        logging.shutdown()

    def meta_model_fit_predict(self, X_train, y_train, X_test):
        logger = hf.create_logger('Meta-Model-Fit')
        logger.info('Fitting of the meta model has started.')
        if self.meta_model is None:
            logger.error(
                'No meta model was passed so cannot do fit and predict.'
            )
            raise ValueError(
                'No meta model was passed so cannot do fit and predict.'
            )
        meta_model_str = self.meta_model.__str__().replace('\n', '\t\n')
        logger.info(
            f'The meta model that was passed to the StackerEnsembleModel is:\n'
            f'{meta_model_str}'
        )
        X_train = hf.check_input_data_type(X_train)
        X_test = hf.check_input_data_type(X_test)
        y_train = hf.check_input_data_type(y_train, one_dim=True)

    def meta_model_fit(self):
        pass

    def meta_model_predict(self):
        pass

    def get_preprocessor(self):
        pass

    def set_preprocessor(self):
        pass

    def get_base_models_hyper_param_grid(self):
        pass

    def set_base_models_hyper_param_grid(self):
        pass

    def get_base_models(self):
        pass

    def set_base_models(self):
        pass

    def get_meta_model_hyper_param_grid(self):
        pass

    def set_meta_model_hyper_paam_grid(self):
        pass

    def get_meta_model(self):
        pass

    def set_meta_model(self):
        pass

    def save_stacker(self, out_path='.', out_name='stacked-ensemble'):
        with open(f'{out_path}/{out_name}.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_stacker(in_path='.', in_name='stacked-ensemble'):
        with open(f'{in_path}/{in_name}.pkl', 'rb') as f:
            return pickle.load(f)
