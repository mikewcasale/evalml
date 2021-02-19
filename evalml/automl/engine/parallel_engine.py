from evalml.automl.engine import EngineBase
from dask import delayed
from dask.distributed import Client, as_completed


class ParallelEngine(EngineBase):
    """A parallel engine for the AutoML search. Trains and scores pipelines locally, in parallel."""

    def __init__(self, X_train=None, y_train=None, automl=None, should_continue_callback=None, pre_evaluation_callback=None, post_evaluation_callback=None, n_workers=4):
        """Base class for the engine API which handles the fitting and evaluation of pipelines during AutoML.

        Arguments:
            X_train (ww.DataTable): training features
            y_train (ww.DataColumn): training target
            automl (AutoMLSearch): a reference to the AutoML search. Used to access configuration and by the error callback.
            should_continue_callback (function): returns True if another pipeline from the list should be evaluated, False otherwise.
            pre_evaluation_callback (function): optional callback invoked before pipeline evaluation.
            post_evaluation_callback (function): optional callback invoked after pipeline evaluation, with args pipeline and evaluation results. Expected to return a list of pipeline IDs corresponding to each pipeline evaluation.
            n_workers (int): how many workers to use for the ParallelEngine's Dask client
        """
        super().__init__(X_train=X_train, y_train=y_train, automl=automl,
                       should_continue_callback=should_continue_callback,
                       pre_evaluation_callback=pre_evaluation_callback,
                       post_evaluation_callback=post_evaluation_callback)
        self.client = Client(n_workers=n_workers)

    def evaluate_batch(self, pipelines):
        """Evaluate a batch of pipelines using the current dataset and AutoML state.

        Arguments:
            pipelines (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.

        Returns:
            list (int): a list of the new pipeline IDs which were created by the AutoML search.
        """
        # super().evaluate_batch()
        # result = EngineResult()

        if self._pre_evaluation_callback:
            for pipeline in pipelines:
                self._pre_evaluation_callback(pipeline)

        dask_pipelines = []
        while len(pipelines) > 0:
            dask_pipelines.append(pipelines.pop())

        # def train_and_score_pipeline(pipeline, automl, full_X_train, full_y_train):
        #     self.train_and_score_pipeline(pipeline, automl, full_X_train, full_y_train)
        #     return pipeline, func(pipeline, automl, full_X_train, full_y_train)

        pipeline_futures = self.client.map(self.train_and_score_pipeline, dask_pipelines, automl=self.automl,
                                           full_X_train=self.X_train, full_y_train=self.y_train)

        new_pipeline_ids=[]
        for future in as_completed(pipeline_futures):
            evaluation_result = future.result()
            new_pipeline_ids.append(self._post_evaluation_callback(pipeline, evaluation_result))

        return new_pipeline_ids
        # if self.X_train is None or self.y_train is None:
        #     raise ValueError("Dataset has not been loaded into the engine.")
        # new_pipeline_ids = []
        # index = 0
        # while self._should_continue_callback() and index < len(pipelines):
        #     pipeline = pipelines[index]
        #     self._pre_evaluation_callback(pipeline)
        #     evaluation_result = EngineBase.train_and_score_pipeline(pipeline, self.automl, self.X_train, self.y_train)
        #     new_pipeline_ids.append(self._post_evaluation_callback(pipeline, evaluation_result))
        #     index += 1
        return new_pipeline_ids

    # def train_and_score_pipeline(self, pipeline, automl, full_X_train, full_y_train):
    #     import pdb; pdb.set_trace()
    #     result = super().train_and_score_pipeline(pipeline, automl, full_X_train, full_y_train)
    #     return result, pipeline