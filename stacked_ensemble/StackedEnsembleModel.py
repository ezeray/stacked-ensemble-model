# -*- coding: utf-8 -*-
import numpy as np
from time import time
import pickle
from sklearn.model_selection import (GridSearchCV, StratifiedKFold) # noqa
import BaseHelpFuncs as hf


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

    def save_stacker(self, out_path='.', out_name='stacked-ensemble'):
        """Save the full model object in a pickle.

        Args:
            out_path (str, optional): Path to save pickled model.
                Defaults to '.'.
            out_name (str, optional): Name to save pickled model.
                Defaults to 'stacked-ensemble'.
        """
        with open(f'{out_path}/{out_name}.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_stacker(in_path='.', in_name='stacked-ensemble'):
        """Static method to load a previously-saved pickeld model.

        Usage for this is as follow:

        my_new_stacked = StackedEnsembleModel.load_stacker(...)

        Args:
            in_path (str, optional): Path for saved pickled model.
                Defaults to '.'.
            in_name (str, optional): Name for saved pickled model.
                Defaults to 'stacked-ensemble'.

        Returns:
            [type]: [description]
        """
        with open(f'{in_path}/{in_name}.pkl', 'rb') as f:
            return pickle.load(f)

    def preprocess_fit_transform(self, X):
        """If a preprocessing step/pipeline has been set, fit and apply
        transformation to the feature set. The preprocessor can be:


        Args:
            X ([type]): Dataset on which to fit and transform preprocessor.

        Raises:
            TypeError: If no preprocessor was set.

        Returns:
            np.ndarray: If using a sklearn preprocessing step or pipeline,
                returns the datatype of the output, which is typically
                a numpy array. However, returns whichever type is returned
                by the preprocessor class in general, so can be a sparse
                matrix, a pandas dataframe, or anything else.
        """
        if self.preprocessor is None:
            raise TypeError('No preprocessor was passed so cannot fit and '
                            'transform any data.'
                            )

        X = hf.check_input_data_type(X)

        self.preprocessor_fit(X)
        return (
            self.preprocessor_transform(X)
        )

    def preprocessor_fit(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Raises:
            TypeError: [description]
        """
        if self.preprocessor is None:
            raise TypeError('No preprocessor was passed so cannot fit '
                            'any data.'
                            )
        X = hf.check_input_data_type(X)
        if not self.preprocessor_is_fit:
            self.preprocessor.fit(X)
            self.preprocessor_is_fit = True

    def preprocessor_transform(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Raises:
            TypeError: [description]
            AttributeError: [description]

        Returns:
            np.ndarray: If using a sklearn preprocessing step or pipeline,
                returns the datatype of the output, which is typically
                a numpy array. However, returns whichever type is returned
                by the preprocessor class in general, so can be a sparse
                matrix, a pandas dataframe, or anything else.
        """
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

    def base_models_tune(
        self, X, y, additional_grid_args={}, log_path='.'
    ):
        """Runs grid search to tune hyper-parameters for the specified base
        models only. The best estimator is then saved for future use.

        Args:
            X (np.ndarray, pd.DataFrame, scipy.sparse): The training feature
                set, which can be any type as long as it is accepted by the
                base models passed by the user. If using sklearn, generally
                accepted are a numpy ndarray or a pandas DataFrame, but could
                also be a scipy sparse matrix.
            y (np.ndarray, pd.DataFrame): The training label set, same as
                above.
            additional_grid_args (dict, optional): Additional parameters to be
                passed to the grid search, e.g. verbose or n_jobs.
                Defaults to {}.
            log_path (str, optional): Path to write log. Defaults to '.'.

        Raises:
            AttributeError: [description]
            ValueError: [description]
        """
        logger = hf.create_logger('Base-Hyper-Param-Tuning', out_path=log_path)
        logger.info('Base models hyper parameter tuning has started.')
        tune_start = time()

        if self.base_hyper_param_tuners is None:
            err_msg = 'No parameters were passed for tuning.'
            logger.error(err_msg)
            raise AttributeError(err_msg)

        X = hf.check_input_data_type(X)
        y = hf.check_input_data_type(y, one_dim=True)

        n_models = len(self.base_hyper_param_tuners.keys())
        logger.info(
            f'Will be tuning hyper-parameters for {n_models} models. '
        )

        models_missing = [
            miss for miss in self.base_hyper_param_tuners.keys()
            if miss not in self.base_models.keys()
            ]
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

        for m in self.base_hyper_param_tuners.keys():
            logger.info(
                f'Tuning hyper-parameters for {m} using scorer {self.scorer}.'
            )
            if not self.is_tuned[m]:
                try:
                    grid_search = GridSearchCV(
                        self.base_models[m],
                        param_Grid=self.base_hyper_param_tuners[m],
                        cv=self.n_folds,
                        scoring=self.scorer,
                        **additional_grid_args
                    )
                    fit_start = time()
                    grid_search.fit(X, y)
                    logger.info(
                        f'The fit took {hf.convert_time(time() - fit_start)}'
                    )

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
                    train_score = grid_search.score(X, y)

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
                prev_params = self.base_models[m].get_params()
                logger.info(
                    f'Mode {m} had previously been tuned.\n\tHere are its '
                    f'parameters:\n\t\t{prev_params}'
                )

        logger.info(
            f'The complete tuning process for the base models took '
            f'{hf.convert_time(time() - tune_start)}'
        )
        hf.logging.shutdown()

    def base_models_fit(self, X, y=None):
        pass

    def base_models_predict(self, X, y=None):
        pass

    def base_models_fit_predict(self, X, y=None):
        pass
