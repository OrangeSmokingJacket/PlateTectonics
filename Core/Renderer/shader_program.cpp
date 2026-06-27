#include "shader_program.hpp"

#include <fstream>

namespace Core {

    ShaderProgram::ShaderProgram(VertexShaderPtr vertex_shader, FragmentShaderPtr fragment_shader)
        : vertex_shader_(std::move(vertex_shader))
        , fragment_shader_(std::move(fragment_shader))
        , shader_program_id_(glCreateProgram()) {
        glAttachShader(shader_program_id_, vertex_shader_->ShaderId());
        glAttachShader(shader_program_id_, fragment_shader_->ShaderId());

        glLinkProgram(shader_program_id_);
#ifndef NDEBUG
        GLint status;
        glGetProgramiv(shader_program_id_, GL_LINK_STATUS, &status);
        if (status != GL_TRUE) {
            GLint len = 0;
            glGetProgramiv(shader_program_id_, GL_INFO_LOG_LENGTH, &len);
            std::string message;
            message.resize(len);
            glGetProgramInfoLog(shader_program_id_, len, NULL, message.data());
            std::cerr << "ShaderProgram linkage error: " << message << std::endl;
            std::cerr.flush();
            std::exit(-1);
        }
#endif
        glValidateProgram(shader_program_id_);
    }

    ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
        : vertex_shader_(std::move(other.vertex_shader_))
        , fragment_shader_(std::move(other.fragment_shader_))
        , shader_program_id_(other.shader_program_id_) {
        other.shader_program_id_ = GL_INVALID_INDEX;
    }

    ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept
    {
        if (shader_program_id_ != GL_INVALID_INDEX) {
            glDeleteProgram(shader_program_id_);
        }
        vertex_shader_ = std::move(other.vertex_shader_);
        fragment_shader_ = std::move(other.fragment_shader_);
        shader_program_id_ = other.shader_program_id_;
        other.shader_program_id_ = GL_INVALID_INDEX;

        return *this;
    }

    ShaderProgram::~ShaderProgram() {
        if (shader_program_id_ != GL_INVALID_INDEX) {
            glDeleteProgram(shader_program_id_);
        }
    }

    void ShaderProgram::Bind() {
        glUseProgram(shader_program_id_);
    }

    void ShaderProgram::Unbind() {
        glUseProgram(0);
    }

    GLuint ShaderProgram::ProgramId() const noexcept {
        return shader_program_id_;
    }

    GLuint ShaderProgram::VertexShaderID() const noexcept {
        return vertex_shader_->ShaderId();
    }

    GLuint ShaderProgram::FragmentShaderID() const noexcept {
        return fragment_shader_->ShaderId();
    }

}
