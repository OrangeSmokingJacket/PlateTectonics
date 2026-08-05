#pragma once

#include "array_utils.hpp"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace utils {

    template <size_t PoolSize>
    class ThreadPool {
    public:
        ThreadPool();
        ~ThreadPool();

        void Enqueue(std::function<void()> task);

        ThreadPool* Get() noexcept;

    private:
        inline static ThreadPool<PoolSize>* s_thread_pool_ = nullptr;

        std::queue<std::function<void()>> task_queue_;
        std::mutex queue_mutex_;
        std::condition_variable_any cv_;
        // threads have to be initialized after mutex, cv and queue
        std::array<std::jthread, PoolSize> workers_;
    };

    using DefaultThreadPool = ThreadPool<6>;

    template <size_t PoolSize>
    ThreadPool<PoolSize>::ThreadPool()
        : workers_(utils::InitializeAll<std::jthread>(std::make_index_sequence<PoolSize>{},
                                                      [this](std::stop_token stop_token) {
                                                          while (!stop_token.stop_requested()) {
                                                              std::function<void()> task;
                                                              {
                                                                  std::unique_lock lock(queue_mutex_);
                                                                  cv_.wait(lock, stop_token, [this]() {
                                                                      return !task_queue_.empty();
                                                                  });
                                                                  if (stop_token.stop_requested())
                                                                      return;
                                                                  task = std::move(task_queue_.front());
                                                                  task_queue_.pop();
                                                              }
                                                              task();
                                                          }
                                                      })) {
        static std::once_flag static_setup_flag;
        std::call_once(static_setup_flag, [this]() { s_thread_pool_ = this; });

        if (std::thread::hardware_concurrency() > 0 && PoolSize >= std::thread::hardware_concurrency()) {
            std::cerr << "Warning: ThreadPool<" << PoolSize << "> may cause oversubscription with hardware limit of "
                      << std::thread::hardware_concurrency() << "threads" << std::endl;
            std::cerr.flush();
        }
    }

    template <size_t PoolSize>
    ThreadPool<PoolSize>::~ThreadPool() {
        s_thread_pool_ = nullptr;
        for (auto& thread : workers_) {
            thread.request_stop();
        }
        cv_.notify_all();
    }

    template <size_t PoolSize>
    void ThreadPool<PoolSize>::Enqueue(std::function<void()> task) {
        {
            std::lock_guard lock(queue_mutex_);
            task_queue_.push(std::move(task));
        }
        cv_.notify_one();
    }

    template <size_t PoolSize>
    auto ThreadPool<PoolSize>::Get() noexcept -> ThreadPool* {
        return s_thread_pool_;
    }
} // namespace utils
