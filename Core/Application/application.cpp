#include "application.hpp"

#include <GLFW/glfw3.h>

#include <cassert>
#include <iostream>
#include <mutex>

namespace Core {

    Application::Application(const AppConfig& config)
        : main_window_(config.main_window_config) {
        static std::once_flag static_setup_flag;
        std::call_once(static_setup_flag, [this]() { s_application_ = this; });
    }

    Application::~Application() { s_application_ = nullptr; }

    void Application::RunMainLoop() {
        run_main_loop_ = true;

        while (run_main_loop_) {
            glfwPollEvents();

            if (glfwWindowShouldClose(main_window_.GetGLFWWindow())) {
                Stop();
                break;
            }

            auto new_time_point = std::chrono::high_resolution_clock::now();
            auto delta_time =
                static_cast<float>(
                    std::chrono::duration_cast<std::chrono::microseconds>(new_time_point - last_time_point_).count()) /
                1'000'000.0f;
            last_time_point_ = new_time_point;

            // TODO: pass delta_time
            for (const auto& layer : layers_) {
                layer->Update(delta_time);
            }
            for (const auto& layer : layers_) {
                layer->Render();
            }

            assert(glGetError() == GL_NO_ERROR);
            main_window_.Update();
        }
    }

    void Application::Stop() { run_main_loop_ = false; }

    void Application::AddLayer(std::unique_ptr<BaseLayer>&& layer) { layers_.emplace_back(std::move(layer)); }

    Application* Application::Get() { return s_application_; }

    Window* Application::GetMainWindow() noexcept { return &main_window_; }
} // namespace Core
