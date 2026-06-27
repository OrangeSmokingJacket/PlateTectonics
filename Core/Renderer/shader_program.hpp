#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cassert>
#include <iostream>
#include <Core/Filesystem/filesystem.hpp>

namespace Core {

    static const std::filesystem::path shaders_dir = std::filesystem::path(SHADERS_DIR).make_preferred();
    using ShaderID = GLuint;

    enum class ShaderType : uint8_t {
        Vertex,
        Fragment
    };

    template <ShaderType TYPE>
        requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    GLuint CompileShader(const Filesystem::buffer_t& source);


    template <ShaderType TYPE> requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
        class Shader {
        public:
            explicit Shader(GLuint shader_id);
            explicit Shader(const std::filesystem::path& path);

            Shader(const Shader&) = delete;
            Shader& operator=(const Shader&) = delete;

            Shader(Shader&&) noexcept;
            Shader& operator=(Shader&&) noexcept;
            ~Shader();

            ShaderID ShaderId() const noexcept;

        private:
            ShaderID shader_id_ = GL_INVALID_INDEX;
    };

    // shared ptr is not the best, but it makes them usable in other pipelines, maintaining RAII
    using VertexShader = Shader<ShaderType::Vertex>;
    using FragmentShader = Shader<ShaderType::Fragment>;

    using VertexShaderPtr = std::shared_ptr<VertexShader>;
    using FragmentShaderPtr = std::shared_ptr<FragmentShader>;

    // TODO: there is a need to mix and match vertex and fragment shaders
    class ShaderProgram {
    public:
        explicit ShaderProgram(VertexShaderPtr vertex_shader, FragmentShaderPtr fragment_shader);

        ShaderProgram(const ShaderProgram&) = delete;
        ShaderProgram& operator=(const ShaderProgram&) = delete;

        ShaderProgram(ShaderProgram&&) noexcept;
        ShaderProgram& operator=(ShaderProgram&&) noexcept;
        ~ShaderProgram();

        void Bind();
        static void Unbind();

        GLuint ProgramId() const noexcept;
        ShaderID VertexShaderID() const noexcept;
        ShaderID FragmentShaderID() const noexcept;

    private:
        VertexShaderPtr vertex_shader_;
        FragmentShaderPtr fragment_shader_;
        GLuint shader_program_id_ = GL_INVALID_INDEX;
    };

    template <ShaderType TYPE>
        requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    ShaderID CompileShader(const Filesystem::buffer_t& source) {
        ShaderID shader_id = GL_INVALID_INDEX;

        if constexpr (TYPE == ShaderType::Vertex) {
            shader_id = glCreateShader(GL_VERTEX_SHADER);
        } else {
            shader_id = glCreateShader(GL_FRAGMENT_SHADER);
        }

        assert(shader_id != GL_INVALID_INDEX && "shader was not created in CompileShader()");

        const char* src = source.c_str();
        glShaderSource(shader_id, 1, &src, nullptr);
        glCompileShader(shader_id);

#ifndef NDEBUG
        GLint status;
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE) {
            GLint len = 0;
            glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &len);
            std::string message;
            message.resize(len);
            glGetShaderInfoLog(shader_id, len, NULL, message.data());
            std::cerr << "Shader compilation error: " << message << std::endl;
            std::cerr.flush();
            std::exit(-1);
        }
#endif

        return shader_id;
    }

    template <ShaderType TYPE> requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    Shader<TYPE>::Shader(GLuint shader_id)
        : shader_id_(shader_id) {}

    template <ShaderType TYPE>
        requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    Shader<TYPE>::Shader(const std::filesystem::path& path)
        : shader_id_(CompileShader<TYPE>(Filesystem::ReadFullFile(path))) {}

    // set GL_INVALID_INDEX to moved from shaders, so they won't deallocate them

    template <ShaderType TYPE>
        requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    Shader<TYPE>::Shader(Shader&& other) noexcept
        : shader_id_(other.shader_id_) {
        other.shader_id_ = GL_INVALID_INDEX;
    }

    template<ShaderType TYPE>
        requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    Shader<TYPE>& Shader<TYPE>::operator=(Shader<TYPE>&& other) noexcept {
        if (shader_id_ != GL_INVALID_INDEX) {
            glDeleteShader(shader_id_);
        }
        shader_id_ = other.shader_id_;
        other.shader_id_ = GL_INVALID_INDEX;

        return *this;
    }

    template<ShaderType TYPE>
        requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    Shader<TYPE>::~Shader() {
        if (shader_id_ != GL_INVALID_INDEX) {
            glDeleteShader(shader_id_);
        }
    }

    template<ShaderType TYPE>
        requires (TYPE == ShaderType::Vertex || TYPE == ShaderType::Fragment)
    ShaderID Shader<TYPE>::ShaderId() const noexcept { return shader_id_; }


    // User defined Shader
    class UDS {
    public:
        virtual ~UDS();

        // setup OpenGL state
        virtual void BindData() = 0;
        // Execute shaders
        virtual void Compute() = 0;

    };

    enum class UDSType {
        Bypass,
        Override,
        Compute,
    };

    using UDSPtr = std::unique_ptr<UDS>;

}