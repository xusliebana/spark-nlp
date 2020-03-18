package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._

/**
  * This class is used to calculate ALBERT embeddings for For Sequence Batches of WordpieceTokenizedSentence.
  * Input for this model must be tokenzied with a SentencePieceModel,
  *
  * This Tensorflow model is using the weights provided by https://tfhub.dev/google/albert_xlarge/3
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


  // keys representing the input and output tensors of the ALBERT model

  private val tokenIdsKey = "module/input_ids"
  private val maskIdsKey = "module/input_mask"
  private val segmentIdsKey = "module/segment_ids"
  private val outputSequenceKey = "module/seq_out"
  private val outputPooledKey = "module/pooled_out"

  /**
    * Calculate the embeddings for a sequence of Tokens and create WordPieceEmbeddingsSentence objects from them
    *
    * @param sentences    A sequence of Tokenized Sentences for which embeddings will be calculated
    * @param poolingLayer Define which output layer you want from the model pooled_output or sequence_output. https://tfhub.dev/google/albert_xlarge/3 for reference
    * @return A Seq of WordpieceEmbeddingsSentence, one element for each input sentence
    */
  def calculateEmbeddings(sentences: Seq[WordpieceTokenizedSentence],
                          poolingLayer: String,
                          batchSize: Int,
                          maxSentenceLength: Int,
                          dimension: Int,
                          caseSensitive: Boolean
                         ): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>

      // Encode batches with sentence end and start token id
      val encoded = batch.map(s => encode(s._1, maxSentenceLength))

      // Calculate Embeddings with the Albert model
      val vectors = tag(encoded, extractPoolingLayer(poolingLayer), maxSentenceLength)


      // Wrap our result with internal Spark-NLP WordpieceEmbeddingsSentence Case Class Data structure
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        sentences.length

        val tokenLength = sentence._1.tokens.length


        // We remove first and last because they are sentence end/start Id's
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

        // Create internal Spark-NLP TokenPieceEmbeddings Data structure
        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).map {
          case (token, tokenEmbedding) =>
            TokenPieceEmbeddings(
              token.token,
              token.token,
              -1,
              isWordStart = true,
              isOOV = false,
              tokenEmbedding,
              token.begin,
              token.end
            )

        }

        // Create internal Spark-NLP WordpieceEmbeddingsSentence Data structure
        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._2)
      }
    }.toSeq



  }

  /**
    * Tags a seq of TokenizedSentences, will get the embeddings according to key.
    *
    * @param batch         The Tokens for which we calculate embeddings
    * @param embeddingsKey Specification of the output embedding for Albert
    * @return The Embeddings Vector. For each Seq Element we have a Sentence, and for each sentence we have an Array for each of its words. Each of its words gets a float array to represent its Embeddings
    */
  def tag(batch: Seq[Array[Int]], embeddingsKey: String, maxSentenceLength: Int): Seq[Array[Array[Float]]] = {

    // Create Tensor Resources for every input Tensor
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()
    val tensorsSegments = new TensorResources()


    // Create Buffers for every Tensor Resource
    val tokenBuffers = tensors.createIntBuffer(batch.length * maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batch.length * maxSentenceLength)
    val segmentBuffers = tensorsSegments.createIntBuffer(batch.length * maxSentenceLength)


    // Put the data into the buffers
    val shape = Array(batch.length.toLong, maxSentenceLength)
    batch.map { sentence =>
      if (sentence.length > maxSentenceLength) {
        tokenBuffers.put(sentence.take(maxSentenceLength))
        maskBuffers.put(sentence.take(maxSentenceLength).map(x => if (x == 0) 0 else 0))
        segmentBuffers.put(Array.fill(maxSentenceLength)(0))
      }
      else {
        tokenBuffers.put(sentence)
        maskBuffers.put(sentence.map(x => 1))
        segmentBuffers.put(Array.fill(maxSentenceLength)(1))
      }
    }

    // Flip the buffers
    tokenBuffers.flip()
    maskBuffers.flip()
    segmentBuffers.flip()

    // Get TF Session
    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes).runner

    // Create tensors containing the buffer values
    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensorsSegments.createIntBufferTensor(shape, segmentBuffers)

    // Feed the tensors to the model and fetch the output
    // We are adding the _1 to the key, because they seem overloaded and Tensorflow must have internally added the _1 suffix
    runner
      //      .feed(tokenIdsKey, tokenTensors)
      .feed(tokenIdsKey + "_1", tokenTensors)

      //      .feed(maskIdsKey, maskTensors)
      .feed(maskIdsKey + "_1", maskTensors)

      //      .feed(segmentIdsKey, segmentTensors)
      .feed(segmentIdsKey + "_1", segmentTensors)

      .fetch(embeddingsKey)


    // Get the model outputs
    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    // Deallocate TF Session
    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenBuffers.clear()


    // Prepare Output

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

  /** The dimensionality of the output Embeddings
    * * pooled_output: pooled output of the entire sequence with shape [batch_size, hidden_size].
    * * sequence_output: representations of every token in the input sequence with shape [batch_size, max_sequence_length, hidden_size].
    *
    * @param layer Layer specification
    * @return The dimension of chosen layer
    */
  def getDimensions(layer: String): Int = {
    { // TODO get this right for the different types of models..
      layer match {
        case "pooled_output" =>
          2048
        case "sequence_output" =>
          2048
      }
    }
  }

  /**
    * Handy acess function for getting the name of the pooling layers
    *
    * @param layer The pooling layer for which the name is requested
    * @return The name of the pooling layer
    */
  def extractPoolingLayer(layer: String): String = {
    layer match {
      case "sentence_embeddings" =>
        outputPooledKey
      case "token_embeddings" =>
        outputSequenceKey
    }
  }

  /**
    * Encode a  tokenized sentence containing token-ids  which is represented as WordpieceTokenizedSentence as sentence for the Albert model.
    * This function makes sure every sentence is not longer than maxSentenceLength and also adds the start/end token-ids at the end and beginning of each token for the Albert model.
    *
    * @param sentence          The tokenized sentence containing token-ids which is represented as WordpieceTokenizedSentence
    * @param maxSentenceLength Maximum allowed length for sentences. Sentences longer than  @param maxSentenceLength will be trimmed.
    * @return The input sentences encodes as Array[Int], ready for Albert consumption.
    */
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