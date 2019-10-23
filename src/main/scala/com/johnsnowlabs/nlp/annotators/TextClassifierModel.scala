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


class TextClassifierModel(override val uid: String) extends AnnotatorModel[TextClassifierModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  def this() = this(Identifiable.randomUID("CLF"))
    
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, WORD_EMBEDDINGS)
  override val outputAnnotatorType: AnnotatorType = INTENT
    
  private val logger = LoggerFactory.getLogger("TextClassifierModel")
        
  //val textCol = new Param[String](this, "textCol", "The text we are trying to classify.")
  //def setTextColumn(value: String): this.type = set(textCol, value)
  //def getTextColumn(value: String): String = $(textCol)

  val labelCol = new Param[String](this, "labelCol", "column with the intent result of every row.")
  def setLabelColumn(value: String): this.type = set(labelCol, value)
  def getLabelColumn(value: String): String = $(labelCol)
        
  val classifierName = new Param[String](this, "classifier", "the name of ML algorithm") // not sure if we have Param[String] or just StringParam??
  def setClassifierName(value: String): this.type = set(classifierName, value)
  def getClassifierName(value: String): String = $(classifierName)

  val featureCol = new Param[String](this, "featureCol", "the type of features that is going to be used in ML model.. options: tfidf, cvec, w2v, bert") // not sure if we have Param[String] or just StringParam??
  def setFeatureCol(value: String): this.type = set(featureCol, value)
  def getFeatureCol(value: String): String = $(featureCol)

  val multiLabel = new BooleanParam (this, "multiLabel", "is this a multilable or single lable classification problem")
  def setMultiLabel(value: String): this.type = set(multiLabel, value)
  def getMultiLabel(value: String): String = $(multiLabel)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    ....
  }

}
    
object TextClassifierModel extends ParamsAndFeaturesReadable[TextClassifierModel]