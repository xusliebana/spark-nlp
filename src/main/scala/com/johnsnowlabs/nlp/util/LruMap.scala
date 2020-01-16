package com.johnsnowlabs.nlp.util

import scala.collection.mutable


class LruMap[TKey, TValue](maxCacheSize: Int) {
  private type QueryCount = Int
  private var cache = mutable.Map.empty[TKey, (TValue, QueryCount)]

  private var size = 0

  private def deleteBulk(): Unit = {
    size = (size / 1.3).toInt
    cache = mutable.Map[TKey, (TValue, QueryCount)](cache.toList.sortBy(-_._2._2).take(size): _*)
  }

  def clear(): Unit = {
    cache.clear()
    size = 0
  }

  def getSize: Int = {
    size
  }

  def getOrElseUpdate(key: TKey, value: => TValue): TValue = {
    val isNewKey = !cache.contains(key)
    if (!isNewKey) {
      val content = cache(key)
      cache(key) = (content._1, content._2+1)
      content._1
    } else {
      if (getSize >= maxCacheSize)
        deleteBulk()
      size += 1
      val content = value
      cache(key) = (content, 0)
      content
    }
  }

  def get(key: TKey): Option[TValue] = {
    val current = cache.get(key)
    if (current.isDefined) {
      val priority = current.get._2 + 1
      cache(key) = (current.get._1, priority)
      Some(current.get._1)
    } else None
  }

  private object KeyPriorityOrdering extends Ordering[TKey] {
    override def compare(x: TKey, y: TKey): Int = {
      if (x == y) return 0
      val a = cache.get(x).map(_._2).getOrElse(0)
      val b = cache.get(y).map(_._2).getOrElse(0)
      if (a > b) -1 else 1
    }
  }

}