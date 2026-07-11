#include "buffer_objects.hpp"

namespace Core {

    VertexArrayObject::VertexArrayObject() {
        glGenVertexArrays(1, &id_);
        assert(glGetError() == GL_NO_ERROR);
    }

    VertexArrayObject::VertexArrayObject(VertexArrayObject&& other) noexcept { std::swap(id_, other.id_); }

    VertexArrayObject& VertexArrayObject::operator=(VertexArrayObject&& other) noexcept {
        assert(id_ == GL_INVALID_INDEX);
        std::swap(id_, other.id_);
        return *this;
    }

    VertexArrayObject::~VertexArrayObject() {
        if (id_ != GL_INVALID_INDEX) {
            glDeleteVertexArrays(1, &id_);
        }
    }

    void VertexArrayObject::Bind() {
        assert(CurrentBinding() == 0);
        assert(id_ != GL_INVALID_INDEX);
        glBindVertexArray(id_);
    }

    void VertexArrayObject::Unbind() { glBindVertexArray(0); }

    GLuint VertexArrayObject::CurrentBinding() {
        // glGetIntegerv does not have a unsigned version
        GLint result;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &result);
        return result;
    }

} // namespace Core
