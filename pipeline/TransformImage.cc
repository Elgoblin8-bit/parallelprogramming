/*
  Author     : Kenner Jimenez
  Description: A program that reads an image, modifies its pixels, and
writes the result.
*/
/**************************************************/
// System includes.
#include <Magick++.h>
#include <filesystem>
#include <iostream>
#include <print>
#include <string>
#include <thread>
#include <tuple>

/**************************************************/
// Local includes.
#include "ThreadSafeQueue.hpp"

/**************************************************/
// Function prototypes.

void
modifyImage (Magick::Image& image);

std::tuple<Magick::Image, std::filesystem::path>
getInput ();

std::tuple<size_t, size_t, size_t, Magick::Quantum*>
getImageData (Magick::Image& image);

void
finishLine (Magick::Image& image, std::filesystem::path filename);

void
filterGreyscale (Magick::Quantum* pixels,
                 size_t width,
                 size_t height,
                 size_t channels,
                 ThreadSafeQueue<Magick::Quantum*>& queue);

void
filterHorizontalReflect (size_t width,
                         size_t channels,
                         ThreadSafeQueue<Magick::Quantum*>& queue);

/**************************************************/

int
main ()
{

  Magick::InitializeMagick (nullptr);
  auto [image, filename] = getInput ();

  auto [width, height, channels, pixels] = getImageData (image);

  ThreadSafeQueue<Magick::Quantum*> theQueue;
  std::jthread greyThread (filterGreyscale,
                           pixels,
                           image.columns (),
                           image.rows (),
                           image.channels (),
                           std::ref (theQueue));

  std::jthread reflectThread (filterHorizontalReflect,
                              image.columns (),
                              image.channels (),
                              std::ref (theQueue));
  finishLine (image, filename);
}

/**************************************************/
void
finishLine (Magick::Image& image, std::filesystem::path filename)
{
  // Write cache back to image.
  image.syncPixels ();
  // Write result.
  std::string newName = filename.stem ().string () + "Mod.pgm";

  filename.replace_filename (newName);

  image.write (filename);

  std::print ("\nOutput:   {}", filename.filename ().string ());
}
/**************************************************/
// Applies a horizontal reflection filter to the image using the queue to
// process rows in parallel.
void
filterHorizontalReflect (size_t width,
                         size_t channels,
                         ThreadSafeQueue<Magick::Quantum*>& queue)
{
  Magick::Quantum* row = nullptr;
  while (true)
  {
    queue.waitAndPop (row);
    if (!row)
      break;

    // Reflection
    for (size_t x {}; x < width / 2; ++x)
    {
      Magick::Quantum* leftPixel = row + (x * channels);
      Magick::Quantum* rightPixel = row + ((width - 1 - x) * channels);

      // Swap 'channels' number of elements starting at leftPixel
      // with the ones at rightPixel
      std::swap_ranges (leftPixel, leftPixel + channels, rightPixel);
    }
  }
}

/**************************************************/
// Converts the image to greyscale using given formula inside the lambda
// function.
void
filterGreyscale (Magick::Quantum* pixels,
                 size_t width,
                 size_t height,
                 size_t channels,
                 ThreadSafeQueue<Magick::Quantum*>& queue)
{
  auto convertPixel = [] (Magick::Quantum* p)
  {
    uint32_t red = p[0];
    uint32_t green = p[1];
    uint32_t blue = p[2];
    // For each pixel, calculate its brightness using the formula:
    uint32_t brightness = (306 * red + 601 * green + 117 * blue) >> 10;
    brightness = std::clamp (
      brightness, static_cast<uint32_t> (0), static_cast<uint32_t> (65535));

    // R
    p[0] = brightness;
    // G
    p[1] = brightness;
    // B
    p[2] = brightness;
  };

  for (size_t y {}; y < height; ++y)
  {
    Magick::Quantum* rowStart = pixels + (y * width * channels);
    for (size_t x {}; x < width; ++x)
    {
      // We calculate the offset based on the number of channels
      Magick::Quantum* pixel = rowStart + (x * channels);
      convertPixel (pixel);
    }
    queue.push (rowStart);
  }
  // Push a nullptr to signal that we're done.
  queue.push (nullptr);
}

/**************************************************/

std::tuple<size_t, size_t, size_t, Magick::Quantum*>
getImageData (Magick::Image& image)
{
  size_t width = image.columns ();
  size_t height = image.rows ();
  size_t channels = image.channels ();
  Magick::Quantum* pixels = image.getPixels (0, 0, width, height);
  return { width, height, channels, pixels };
}
/**************************************************/
// Gets user input for an image file and returns the loaded image.
std::tuple<Magick::Image, std::filesystem::path>
getInput ()
{
  std::print ("Image ==> ");
  std::filesystem::path filename;
  std::cin >> filename;

  Magick::Image image {};
  try
  {
    image.read (filename);
  }
  catch (Magick::Exception& error)
  {
    std::println ("Caught exception: [{}]", error.what ());
    std::exit (1);
  }
  return { image, filename };
}

/**************************************************/
