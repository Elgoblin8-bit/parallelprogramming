/*
  Author     : Gary M. Zoppetti
  Description: A thread-safe queue template.
*/

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

template<typename T>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue () = default;

  // Push an element, notifying any consumers.
  // Move the value into the underlying queue.
  void
  push (T value)
  {
    // Lock mutex, push, notify on destroy.
    std::lock_guard lock (m_mutex);

    m_elements.push (std::move (value));
    m_cv.notify_one ();
  }

  // Wait until an element is available, then pop it
  //   via the reference parameter.
  // Move the element out of the underlying queue.
  void
  waitAndPop (T& value)

  {
    std::unique_lock lock (m_mutex);
    // Wait until its not empty, then pop the front element.
    m_cv.wait (lock, [this] { return !m_elements.empty (); });
    value = std::move (m_elements.front ());
    m_elements.pop ();
  }

  // Return whether the underlying queue is empty.
  bool
  empty () const
  {
    return m_elements.empty ();
  }

private:
  std::queue<T> m_elements;
  mutable std::mutex m_mutex;
  std::condition_variable m_cv;
};