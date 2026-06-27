#include "window.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>

namespace Core {

    void window_resize_callback(GLFWwindow* window, int width, int height) {
    // Tells OpenGL the size of the rendering window
    glViewport(0, 0, width, height);
}

    void PrintOpenGLInfo() {
        std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
        std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
        std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    }

    void Window::Setup() {
        if (!glfwInit()) {
            const char* error;
            glfwGetError(&error);
            std::cerr << "GLFW could is not initialized: " << error << std::endl;
            std::exit(-1);
        }

        glfw_window_ = glfwCreateWindow(configuration_.width, configuration_.height, configuration_.name.c_str(), nullptr, nullptr);
        if (!glfw_window_)
        {
            const char* error;
            glfwGetError(&error);
            std::cerr << "GLFW could not initialize main window: " << error << std::endl;
            glfwTerminate();
            return std::exit(-1);
        }

        glfwMakeContextCurrent(glfw_window_);

        glfwSetFramebufferSizeCallback(glfw_window_, window_resize_callback);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            const char* error;
            glfwGetError(&error);
            std::cerr << "glad was not initialized: " << error << std::endl;
            glfwTerminate();
            return std::exit(-1);
        }

        PrintOpenGLInfo();

        glFrontFace(GL_CCW);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(glfw_window_, true);
        ImGui_ImplOpenGL3_Init("#version 460");
    }

    Window::Window(const WindowConfig& config)
        : configuration_(config) {
        Setup();
    }

    Window::~Window() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwTerminate();
    }

    void Window::Update() {
        glfwSwapBuffers(glfw_window_);
        glfwGetWindowSize(glfw_window_, &configuration_.width, &configuration_.height);
    }

    int Window::Width() const noexcept {
        return configuration_.width;
    }

    int Window::Height() const noexcept {
        return configuration_.height;
    }

    float Window::AspectRation() const noexcept {
        return static_cast<float>(configuration_.width) / static_cast<float>(configuration_.height);
    }

    GLFWwindow* Window::GetGLFWWindow() noexcept {
        return glfw_window_;
    }
}
