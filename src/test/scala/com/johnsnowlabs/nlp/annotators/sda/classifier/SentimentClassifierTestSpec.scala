package com.johnsnowlabs.nlp.annotators.sda.classifier

import com.johnsnowlabs.nlp.annotator.WordEmbeddings
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.FlatSpec

class SentimentClassifierTestSpec extends FlatSpec {

  "A SentimentClassifier" should "be trained by DataFrame" in {
    import SparkAccessor.spark.implicits._

    val trainingDataDF = Seq(
      ("amazing voice acting", 1.0),
      ("amazing voice acting", 1.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("horrible acting", 0.0),
      ("very bad", 0.0),
      ("very bad", 0.0),
      ("very bad", 0.0),
      ("very bad", 0.0),
      ("very fantastic", 1.0),
      ("very fantastic", 1.0),
      ("incredible!!", 1.0)
    ).toDF("text", "sentiment_label")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val dim = 50
    val embeddings = new WordEmbeddings()
    .setInputCols(Array("document", "token"))
      .setOutputCol("embeddings")
      .setEmbeddingsSource("src/test/resources/embeddings/sentiment_embeddings.csv", dim, "TEXT")

    val embeddingFeaturizer = new EmbeddingFeaturizer()
      .setOutputCol("features")

    val predictor = new LogisticRegression().setLabelCol("sentiment_label")

    val predictionConverter = new PredictionConverter().setOutputCol("sentiment")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        embeddingFeaturizer,
        predictor,
        predictionConverter
      ))

    val model = pipeline.fit(trainingDataDF)

    val testDataDF = Seq(
      "amazing voice acting",
      "horrible staff",
      "very bad",
      "simply fantastic",
      "incredible!!",
      "I think this movie is horrible.",
      "simply put, this is like a bad dream, a horrible one, but in an amazing scenario",
      "amazing staff, really horrible movie",
      "horrible watchout bloody thing"
    ).toDF("text")

    model.transform(testDataDF).select("text", "sentiment").show(truncate=false)
    succeed
  }

  "A SentimentClassifier" should "work in a pipeline" in {

    import SparkAccessor.spark.implicits._

    val trainingDataDF = Seq(
      ("amazing voice acting", 1.0),
      ("horrible staff", 0.0),
      ("very bad", 0.0),
      ("simply fantastic", 1.0),
      ("incredible!!", 1.0)
    ).toDF("text", "sentiment_label")

    /// The glove annotator was breaking when trying to save, so I saved the embeddings instead

    val testDataDF = trainingDataDF

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val spellChecker = new NorvigSweetingApproach()
      .setInputCols(Array("normalized"))
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")

    val dim = 50
    val embeddings = new WordEmbeddings()
      .setInputCols(Array("document", "token"))
      .setOutputCol("embeddings")
      .setEmbeddingsSource("src/test/resources/embeddings/sentiment_embeddings.csv", dim, "TEXT")

    val embeddingFeaturizer = new EmbeddingFeaturizer()
      .setOutputCol("features")

    val predictor = new LogisticRegression().setLabelCol("sentiment_label")

    val predictionConverter = new PredictionConverter().setOutputCol("sentiment")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        normalizer,
        spellChecker,
        embeddings,
        embeddingFeaturizer,
        predictor,
        predictionConverter
      ))

    val model = pipeline.fit(trainingDataDF)

    model.transform(testDataDF).select("sentiment").show(truncate=false)

    val PIPE_PATH = "./tmp_pipeline"
    model.write.overwrite().save(PIPE_PATH)
    val loadedPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedPipeline.transform(testDataDF).show(20)

    succeed
  }
}
