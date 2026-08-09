#pragma once
#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <filesystem>

inline std::string insert_suffix_before_extension(const std::string& filepath, const std::string& suffix, const std::string& new_extension = "") {
    std::filesystem::path path(filepath);
    std::string stem = path.stem().string();  
    std::string extension;      // filename without extension
    if (!new_extension.empty()) {
      extension = new_extension;
    } else {
        extension =  path.extension().string(); 
    }
    std::filesystem::path new_path = path.parent_path() / (stem + suffix + extension);
    return new_path.string();
}


template <typename Func, typename... Args>
auto timed_with_progress(const std::string &task_name, Func &&func,
                         Args &&...args)
    -> decltype(func(std::forward<Args>(args)...)) {
  
  std::atomic<bool> done(false);
  auto start = std::chrono::steady_clock::now();
  
  std::thread progress_thread([&]() {
    while (!done) {
      auto elapsed = std::chrono::steady_clock::now() - start;
      auto seconds =
          std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
      std::cout << "\r" << task_name << ": " << seconds << "s" << std::flush;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });
  
  auto result = func(std::forward<Args>(args)...);
  
  done = true;
  progress_thread.join();
  
  auto end = std::chrono::steady_clock::now();
  auto total_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::cout << "\r" << task_name << " completed in: " << total_seconds << "s"
            << std::endl;
  
  return result;
}

#endif // GENERAL_HPP