from dask.distributed import Client, as_completed

from evalml.automl.engine import EngineBase


class ParallelEngine(EngineBase):
    """A parallel engine for the AutoML search. Trains and scores pipelines locally, in parallel."""

    def __init__(self, X_train=None, y_train=None, automl=None, should_continue_callback=None, pre_evaluation_callback=None,
                 post_evaluation_callback=None, n_workers=4):
        """Base class for the engine API which handles the fitting and evaluation of pipelines during AutoML.

        Arguments:
            X_train (ww.DataTable): training features
            y_train (ww.DataColumn): training target
            automl (AutoMLSearch): a reference to the AutoML search. Used to access configuration and by the error callback.
            should_continue_callback (function): returns True if another pipeline from the list should be evaluated, False otherwise.
            pre_evaluation_callback (function): optional callback invoked before pipeline evaluation.
            post_evaluation_callback (function): optional callback invoked after pipeline evaluation, with args pipeline and evaluation results. Expected to return a list of pipeline IDs corresponding to each pipeline evaluation.
            n_workers (int): how many workers to use for the ParallelEngine's Dask client

        Raises:
            ValueError: if n_workers is not a positive integer greater than or equal to 1.
        """
        super().__init__(X_train=X_train, y_train=y_train, automl=automl,
                         should_continue_callback=should_continue_callback,
                         pre_evaluation_callback=pre_evaluation_callback,
                         post_evaluation_callback=post_evaluation_callback)

        if n_workers < 1 or not isinstance(n_workers, int):
            raise ValueError("n_workers must be a positive integer.")
        self.client = Client(n_workers=n_workers)

    def evaluate_batch(self, pipelines):
        """Evaluate a batch of pipelines using the current dataset and AutoML state.

        Arguments:
            pipelines (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.

        Returns:
            list (int): a list of the new pipeline IDs which were created by the AutoML search.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset has not been loaded into the engine.")

        if self._pre_evaluation_callback:
            for pipeline in pipelines:
                self._pre_evaluation_callback(pipeline)

        pipeline_futures = self.client.map(self.train_and_score_pipeline, pipelines, automl=self.automl,
                                           full_X_train=self.X_train, full_y_train=self.y_train)

        new_pipeline_ids = []
        for future in as_completed(pipeline_futures):
            evaluation_result = future.result()
            new_pipeline_ids.append(self._post_evaluation_callback(pipeline, evaluation_result))

        return new_pipeline_ids
