package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._

/**
  * This class is used to calculate ELMO embeddings for For Sequence Batches of TokenizedSentences.
  *
  * https://tfhub.dev/google/elmo/3
  * * pooled_output: pooled output of the entire sequence with shape [batch_size, hidden_size].
  * * sequence_output: representations of every token in the input sequence with shape [batch_size, max_sequence_length, hidden_size].
  *
  * @param tensorflow       Elmo Model wrapper with TensorFlow Wrapper
  * @param batchSize        size of batch
  * @param configProtoBytes Configuration for TensorFlow session
  */

class TensorflowAlbert(val tensorflow: TensorflowWrapper,
                       batchSize: Int,
                       configProtoBytes: Option[Array[Byte]] = None
                      ) extends Serializable {


  // keys representing the input tensors of the XLNET model
  private val tokenIdsKey = "module/input_ids"
  private val maskIdsKey = "module/input_mask"
  private val segmentIdsKey = "module/segment_ids"
  private val outputSequenceKey = "module/seq_out"

  /**
    * Calculate the embeddings for a sequence of Tokens and create WordPieceEmbeddingsSentence objects from them
    *
    * @param sentences    A sequence of Tokenized Sentences for which embeddings will be calculated
    * @param poolingLayer Define which output layer you want from the model word_emb, lstm_outputs1, lstm_outputs2, elmo. See https://tfhub.dev/google/elmo/3 for reference
    * @return A Seq of WordpieceEmbeddingsSentence, one element for each input sentence
    */
  def calculateEmbeddings(sentences: Seq[WordpieceTokenizedSentence],
                          poolingLayer: Int,
                          batchSize: Int,
                          maxSentenceLength: Int,
                          dimension: Int,
                          caseSensitive: Boolean
                         ): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    print("yo")
    //    sentences.zipWithIndex.grouped(batchSize).map{ batch =>
    //      val encoded = batch.map(s => encode(s._1, maxSentenceLength))
    //      val vectors = tag(encoded, extractPoolingLayer(poolingLayer, dimension), maxSentenceLength)
    //
    //    }
    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>

      val encoded = batch.map(s => encode(s._1, maxSentenceLength))

      val vectors = tag(encoded, extractPoolingLayer(poolingLayer, dimension), maxSentenceLength)

      print("debug")
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        sentences.length
        val tokenLength = 23 // quick fix because all tokens are in onesentence._1.tokens.length

        /*All wordpiece embeddings*/
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

        // TODO use SentencePieceEmbeddings HERE
        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap {
          case (token, tokenEmbedding) =>
            val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
            val originalTokensWithEmbeddings = Seq(tokenWithEmbeddings)
            originalTokensWithEmbeddings
        }

        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._2)
      }
    }.toSeq


  }

  /**
    * Tag a seq of TokenizedSentences, will get the embeddings according to key.
    *
    * @param batch         The Tokens for which we calculate embeddings
    * @param embeddingsKey Specification of the output embedding for Elmo
    * @param dimension     Elmo's embeddings dimension: either 512 or 1024
    * @return The Embeddings Vector. For each Seq Element we have a Sentence, and for each sentence we have an Array for each of its words. Each of its words gets a float array to represent its Embeddings
    */
  def tag(batch: Seq[Array[Int]], embeddingsKey: String, maxSentenceLength: Int): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()
    val tensorsSegments = new TensorResources()

    val tokenBuffers = tensors.createIntBuffer(batch.length * maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batch.length * maxSentenceLength)

    val segmentBuffers = tensorsSegments.createIntBuffer(batch.length * maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)
    val sentenceEndTokenId = 2
    batch.map { sentence =>
      if (sentence.length > maxSentenceLength) {
        tokenBuffers.put(sentence.take(maxSentenceLength)) //remove last token, add end token
        maskBuffers.put(sentence.take(maxSentenceLength).map(x => if (x == 0) 0 else 0)) //requried float cast..
        segmentBuffers.put(Array.fill(maxSentenceLength)(0))
      }
      else {
        tokenBuffers.put(sentence)
        maskBuffers.put(sentence.map(x => 1)) // if (x == 0.0.toFloat) 0.0.toFloat else 1.0.toFloat))
        segmentBuffers.put(Array.fill(maxSentenceLength)(1))
      }
    }

    tokenBuffers.flip()
    maskBuffers.flip()
    segmentBuffers.flip()

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensorsSegments.createIntBufferTensor(shape, segmentBuffers)
    // -0.3152 . -0.1753, 0.03, =0,07, -0.27, -0.47503
    runner
      .feed(tokenIdsKey, tokenTensors)
      .feed(tokenIdsKey + "_1", tokenTensors)

      .feed(maskIdsKey, maskTensors)
      .feed(maskIdsKey + "_1", maskTensors)

      .feed(segmentIdsKey, segmentTensors)
      .feed(segmentIdsKey + "_1", segmentTensors)
      .fetch(outputSequenceKey)
    //
    val outs = runner.run().asScala

    val embeddings = TensorResources.extractFloats(outs.head)
    //
    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenBuffers.clear()

    val dim = embeddings.length / (batch.length * maxSentenceLength)
    val shrinkedEmbeddings: Array[Array[Array[Float]]] = embeddings.grouped(dim).toArray.grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(dim)(0f)

    batch.zip(shrinkedEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }

  }

  /**
    * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == 512
    * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === 1024
    * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === 1024
    * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == 1024
    *
    * @param layer Layer specification
    * @return The dimension of chosen layer
    */
  def getDimensions(layer: String): Int = {
    {
      layer match {
        case "word_emb" =>
          512
        case "lstm_outputs1" =>
          1024
        case "lstm_outputs2" =>
          1024
        case "elmo" =>
          1024
      }
    }
  }

  def extractPoolingLayer(layer: Int, dimension: Int): String = {
    val bertLayer = if (dimension == 768) {
      layer match {
        case -1 =>
          "module/bert/encoder/Reshape_13:0"
        case -2 =>
          "module/bert/encoder/Reshape_12:0"
        case 0 =>
          "module/bert/encoder/Reshape_1:0"
      }
    } else {
      layer match {
        case -1 =>
          "module/bert/encoder/Reshape_25:0"
        case -2 =>
          "module/bert/encoder/Reshape_24:0"
        case 0 =>
          "module/bert/encoder/Reshape_1:0"
      }
    }
    bertLayer
  }


  def encode(sentence: WordpieceTokenizedSentence, maxSentenceLength: Int): Array[Int] = {
    val sentenceStartTokenId = 1
    val tokens = sentence.tokens.map(_.pieceId)
    val sentenceEndTokenId = 2

    Array(sentenceStartTokenId) ++
      tokens ++
      Array(sentenceEndTokenId) ++
      Array.fill(maxSentenceLength - tokens.length - 2)(0)
  }

}