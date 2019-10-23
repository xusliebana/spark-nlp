import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.HasWordEmbeddings
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.spark.MapAccumulator
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.classification._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.{array, col, explode, udf}

class TextClassifierApproach(override val uid: String) 
    extends AnnotatorApproach[TextClassifierModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, WORD_EMBEDDINGS)// we will use sentencemebddings later on
  override val outputAnnotatorType: AnnotatorType = INTENT
        
  override val description = "ML based Text Classifier"

  def this() = this(Identifiable.randomUID("CLF"))
        
  private val logger = LoggerFactory.getLogger("TextClassifierApproach")

  val labelCol = new Param[String](this, "labelCol", "column with the intent result of every row.")
  def setLabelColumn(value: String): this.type = set(labelCol, value)
  def getLabelColumn(value: String): String = $(labelCol)
        
  val classifierName = new Param[String](this, "classifierName", "the name of ML algorithm.. options: rf, logreg, nb") // not sure if we have Param[String] or just StringParam??
  def setClassifierName(value: String): this.type = set(classifierName, value)
  def getClassifierName(value: String): String = $(classifierName)
        
  val featureCol = new Param[String](this, "featureCol", "the type of features that is going to be used in ML model.. options: tfidf, cvec, w2v, bert") // not sure if we have Param[String] or just StringParam??
  def setFeatureCol(value: String): this.type = set(featureCol, value)
  def getFeatureCol(value: String): String = $(featureCol)

  val multiLabel = new BooleanParam (this, "multiLabel", "is this a multilable or single lable classification problem")
  def setMultiLabel(value: String): this.type = set(multiLabel, value)
  def getMultiLabel(value: String): String = $(multiLabel)
        
  // TODO: accuracyMetrics, cv, etc.

  setDefault(
    labelCol -> "label",
    classifierName -> "rf",
    featureCol -> "w2v",
    multiLabel -> false
  )
    
        
  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TextClassifierModel = {

    require(get(labelCol).isDefined, "TextClassifierApproach needs 'labelCol' to be set for training")
    require(get(featureCol).isDefined, "TextClassifierApproach needs 'featureCol' to be set for training")
    require(get(classifierName).isDefined, "TextClassifierApproach needs 'classifierName' to be set for training")

    // If column is not Vector, convert 
    def getFeatureColumn(df: DataFrame): DataFrame = {
      df.schema("features").dataType match {
        ...
      }
    }
            
    // this could be out of this annotator.. but the user should give an array of floats.. 
    /*
    val finisher = new Finisher()
      .setInputCols($(featureCol)) // what if featurecol is already an array of floats (rather than sparknlp struct col) ?
      .setOutputCols(Array("features")) 
      .setOutputAsArray(true)
      .setCleanAnnotations(true)
     */
    val label_stringIdx = new StringIndexer()
      .setInputCol($(labelCol))
      .setOutputCol("label")

    val processed_dataset = label_stringIdx.fit(dataset).transform(dataset)
    
    val Array(trainingData, testData) = processed_dataset.randomSplit(Array(0.7, 0.3), seed)
      // TODO: use testData to return metrics
   
    val model = {
      if ($(classifierName) == "rf") new RandomForestClassifier()
      if ($(classifierName) == "lr") new LogisticRegression()
      if ($(classifierName) == "nb") new NaiveBayes()
    }
  // full list of available classifiers here: https://spark.apache.org/docs/latest/ml-classification-regression.html

    model.train(trainingData)
      
    val model = new TextClassifierModel()
      .setLabelColumn($(labelCol))
      .setClassifier($(classifierName))
      .setFeatureType($(featureCol))

    model
  }
  /* where to use these ?
  // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(label_stringIdx.labels)

  // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
  */
        
}

object TextClassifierApproach extends DefaultParamsReadable[TextClassifierApproach]

   

