/*
  Filename   : Timer.hpp
  Author     : Gary M. Zoppetti
  Description: A templated timer class for timing algorithms.
               { steady, system, high_resolution }_clock may be used.
*/

/************************************************************/
// Macro to prevent multiple inclusions

#pragma once

/************************************************************/
// System includes

#include <chrono>

/************************************************************/
// Local includes

/************************************************************/
// Using declarations

/************************************************************/

template<typename Clock = std::chrono::steady_clock>
class Timer
{
public:
  Timer () = default;

  void
  start ()
  {
    m_start = Clock::now ();
  }

  void
  stop ()
  {
    m_stop = Clock::now ();
  }

  [[nodiscard]]
  double
  getElapsedMs () const
  {
    auto timeDelta = m_stop - m_start;
    double elapsedMs =
      std::chrono::duration<double, std::milli> (timeDelta).count ();

    return elapsedMs;
  }

private:
  using TimePoint = typename Clock::time_point;
  TimePoint m_start {};
  TimePoint m_stop {};
};

/************************************************************/