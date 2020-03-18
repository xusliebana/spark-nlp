package com.johnsnowlabs.nlp.embeddings

import java.io.File

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Embeddings from a language model trained on the 1 Billion Word Benchmark.
  *
  * Note that this is a very computationally expensive module compared to word embedding modules that only perform embedding lookups.
  * The use of an accelerator is recommended.
  *
  */
class AlbertEmbeddings(override val uid: String) extends
  AnnotatorModel[AlbertEmbeddings]
  with WriteTensorflowModel
  with HasEmbeddingsProperties
  with HasStorageRef
  with HasCaseSensitiveProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT) // todo requires only ID tokens i think
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS
  val batchSize = new IntParam(this, "batchSize", "Batch size. Large values allows faster processing but requires more memory.")
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  val poolingLayer = new Param[String](this, "poolingLayer", "Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or elmo")
  private var _model: Option[Broadcast[TensorflowAlbert]] = None

  def this() = this(Identifiable.randomUID("ALBERT_EMBEDDINGS"))

  def setBatchSize(size: Int): this.type = {
    if (get(batchSize).isEmpty)
      set(batchSize, size)
    this
  }

  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this

  }

  def setConfigProtoBytes(bytes: Array[Int]): AlbertEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /** Function used to set the embedding output layer of the ALBERT model. See https://tfhub.dev/google/albert_xlarge/3 for reference.
    * * pooled_output: pooled output of the entire sequence with shape [batch_size, hidden_size].
    * * sequence_output: representations of every token in the input sequence with shape [batch_size, max_sequence_length, hidden_size].
    *
    * @param layer Layer specification
    */
  def setPoolingLayer(layer: String): this.type = {
    layer match {
      case "sentence_embeddings" => set(poolingLayer, "pooled_output")
      case "token_embeddings" => set(poolingLayer, "sequence_output")
      case _ => throw new MatchError("poolingLayer must be either pooled_output or sequence_output")
    }
  }

  def getPoolingLayer: String = $(poolingLayer)

  setDefault(
    batchSize -> 32,
    poolingLayer -> "word_emb",
    dimension -> 512
  )

  private var tfHubPath: String = ""

  def setTFhubPath(value: String): Unit = {
    tfHubPath = value
  }

  def getTFhubPath: String = tfHubPath

  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowAlbert(
            tensorflow,
            batchSize = $(batchSize),
            configProtoBytes = getConfigProtoBytes
          )
        )
      )
    }

    this
  }


  /**
    * Tokenize Annotations containing raw sentences.
    *
    * @param sentences
    * @return
    */
  def tokenizeAndEncodeIds(sentences: Seq[Annotation]): Seq[WordpieceTokenizedSentence] = {
    // The SP model will look for a spiece.model file in the ROOT of the spark-nlp project.
    // The SP  file can be downoaded from [URL] (its named 30k-clean.model, you have to rename it to spiece.model
    // The dict and soOperatiosnPath can be an absolute path on the system


    // The Sentence Piece model can be re-used for different models. You just have to swap the spiece.model file in the root of spark-nlp project
    val spModelPath = "/home/loan/Documents/JohnSnowLabs/XLNet/jupyter/SentencePiece/exported_model"
    val soOperationsPath = "/home/loan/venv/XLNET_jupyter_venv/lib/python2.7/site-packages/tf_sentencepiece/_sentencepiece_processor_ops.so.1.14.0"
    //    val dictPath = "/home/loan/Documents/JohnSnowLabs/Docs/PR/src/test/scala/com/johnsnowlabs/nlp/embeddings/768_xlnet_dict.txt"
    //    val spModelPath = "/home/loan/Documents/JohnSnowLabs/spark-nlp-training/python/tensorflow/albert/albert_base_v2/albert_base"
    val dictPath = "/home/loan/Documents/JohnSnowLabs/spark-nlp-training/python/tensorflow/albert/albert_base_v2/albert_base/30k-clean.vocab"
    val sentencePieceTokenizer = SentencePieceTokens.loadSavedModel(spModelPath, soOperationsPath, dictPath, SparkNLP.start()) //Create Spiece Model


    sentencePieceTokenizer.encode(sentences)
  }


  /**
    * takes a document column and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = tokenizeAndEncodeIds(annotations)
    /*Return empty if the real tokens are empty*/
    //    if(tokenizedSentences.nonEmpty) {
    val embeddings = getModelIfNotSet.calculateEmbeddings(
      tokenizedSentences,
      "token_embeddings",
      $(batchSize),
      100,
      $(dimension),
      $(caseSensitive)
    )
    val res = WordpieceEmbeddingsSentence.pack(embeddings)
    print("debug")
    res
  }

  def getModelIfNotSet: TensorflowAlbert = _model.get.value

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_albert", AlbertEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
  }

  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(getOutputCol, wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

}

trait ReadablePretrainedAlbertModel extends ParamsAndFeaturesReadable[AlbertEmbeddings] with HasPretrained[AlbertEmbeddings] {
  override val defaultModelName: Some[String] = Some("albert")

  /** Java compliant-overrides */
  override def pretrained(): AlbertEmbeddings = super.pretrained()

  override def pretrained(name: String): AlbertEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): AlbertEmbeddings = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): AlbertEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadAlbertTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[AlbertEmbeddings] =>

  override val tfFile: String = "albert_tensorflow"

  def readTensorflow(instance: AlbertEmbeddings, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_albert_tf", initAllTables = true)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): AlbertEmbeddings = {

    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )
    val wrapper = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"), initAllTables = true)

    val albert = new AlbertEmbeddings()
      .setModelIfNotSet(spark, wrapper)
    albert
  }
}


object AlbertEmbeddings extends ReadablePretrainedAlbertModel with ReadAlbertTensorflowModel
