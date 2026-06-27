#pragma once

#include "window.hpp"
#include "layer.hpp"

#include <memory>

namespace Core {

    struct AppConfig {
        std::string name;
        WindowConfig main_window_config;
    };

    class Application {
    public:
        explicit Application(const AppConfig& config);

        Application(const Application&) = delete;
        Application(Application&&) = delete;
        Application& operator=(const Application&) = delete;
        Application& operator=(Application&&) = delete;

        ~Application();

        void RunMainLoop();
        void Stop();

        void AddLayer(std::unique_ptr<BaseLayer>&& layer);

        static Application* Get();
        Window* GetMainWindow() noexcept;

    private:
        Window main_window_;
        std::vector<std::unique_ptr<BaseLayer>> layers_;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_time_point_ = std::chrono::high_resolution_clock::now();
        bool run_main_loop_ = true;

        inline static Application* s_application_ = nullptr;
    };

}
