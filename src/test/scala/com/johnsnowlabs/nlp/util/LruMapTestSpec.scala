package com.johnsnowlabs.nlp.util

import org.scalatest._

class LruMapTestSpec extends FlatSpec {

  "A LruMap" should "Deque and enqueue correctly" in {

    val lru = new LruMap[String, Double](5)

    val iv = Seq(
      ("a", 234.5),
      ("b", 345.6),
      ("c", 456.7),
      ("d", 567.8),
      ("e", 678.9)
    )

    iv.foreach{case (i, v) => lru.getOrElseUpdate(i, v)}

    assert(lru.getSize == 5, "Wrong initial size")

    lru.getOrElseUpdate("b", 25.1)
    lru.get("b")
    lru.get("a")
    lru.get("a")
    lru.get("d")
    lru.getOrElseUpdate("e", 22.7)
    lru.get("e")
    lru.get("e")
    lru.get("b")

    assert(lru.getSize == 5, "Size not as expected after getting and updated cache")

    lru.getOrElseUpdate("new", 1.11)
    lru.getOrElseUpdate("even newer", 4.13)

    assert(lru.getSize == scala.math.min((5/1.3).toInt + 2, 5), "Size not as expected after adding 2 new values")

    assert(lru.get("new").isDefined, "Recently added key is not in the LRU!")

    assert(lru.get("c").isEmpty, "value 'c' should not be in LRU since it was never used")
    assert(lru.get("d").isEmpty, "value 'd' should not be in LRU since it was rarely queried (once)")

  }

}
