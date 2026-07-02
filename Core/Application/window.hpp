#pragma once

#include <string>

// order matters here
// clang-format off
#include <ThirdParty/glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

namespace Core {

    struct WindowConfig {
        std::string name;

        int width;
        int height;
        bool is_resizable;
        bool v_sync;
    };

    class Window {
    public:
        explicit Window(const WindowConfig& config);

        Window(const Window&) = delete;
        Window(Window&&) = delete;
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&&) = delete;

        ~Window();

        void Update();

        [[nodiscard]] int Width() const noexcept;
        [[nodiscard]] int Height() const noexcept;
        [[nodiscard]] float AspectRation() const noexcept;

        GLFWwindow* GetGLFWWindow() noexcept;

    private:
        void Setup();

        WindowConfig configuration_;
        GLFWwindow* glfw_window_ = nullptr;
    };

} // namespace Core
