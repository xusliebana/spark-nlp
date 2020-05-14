import sparknlp.internal as _internal


def get_config_path():
    return _internal._ConfigLoaderGetter().apply()


class CoNLLGenerator:
    @staticmethod
    def exportConllFiles(spark, files_path, pipeline, output_path):
        _internal._CoNLLGeneratorExport(spark, files_path, pipeline, output_path).apply()

class NerEvaluator:
    def evaluateNer(spark, ground_truth, predictions, percent=True, output_df=False, mode="entity_level"):
        i = _internal._NerEvaluatorEvaluate(spark, ground_truth, predictions, percent, output_df, mode).apply()
        return i