#pragma once

// order matters here
// clang-format off
#include <ThirdParty/glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <cassert>
#include <utility>

namespace Core {

    // TODO: other enum values besides Texture2D
    class TextureObject {
    public:
        explicit TextureObject();
        TextureObject(const TextureObject&) = delete;
        TextureObject& operator=(const TextureObject&) = delete;
        TextureObject(TextureObject&&) noexcept;
        TextureObject& operator=(TextureObject&&) noexcept;
        ~TextureObject();

        void Bind();
        static void Unbind();
        static void SetActiveUnit(uint32_t index);

        static GLuint CurrentBinding();

    private:
        GLuint id_ = GL_INVALID_INDEX;
    };

    class VertexArrayObject {
    public:
        explicit VertexArrayObject();
        VertexArrayObject(const VertexArrayObject&) = delete;
        VertexArrayObject& operator=(const VertexArrayObject&) = delete;
        VertexArrayObject(VertexArrayObject&&) noexcept;
        VertexArrayObject& operator=(VertexArrayObject&&) noexcept;
        ~VertexArrayObject();

        void Bind();
        static void Unbind();

        static GLuint CurrentBinding();

    private:
        GLuint id_ = GL_INVALID_INDEX;
    };

    enum class BufferObjectType : uint8_t {
        Vertex,
        Element,
        Uniform,
    };

    template <BufferObjectType T>
    consteval GLenum EnumFromType() {
        switch (T) {
            case BufferObjectType::Vertex:
                return GL_ARRAY_BUFFER;
            case BufferObjectType::Element:
                return GL_ELEMENT_ARRAY_BUFFER;
            case BufferObjectType::Uniform:
                return GL_UNIFORM_BUFFER;
            default:
                return GL_INVALID_ENUM;
        }
    }

    // TODO: multiple sub-buffers
    template <BufferObjectType T>
    class BufferObject {
    public:
        explicit BufferObject(uint32_t capacity);
        BufferObject(const BufferObject&) = delete;
        BufferObject& operator=(const BufferObject&) = delete;
        BufferObject(BufferObject&&) noexcept;
        BufferObject& operator=(BufferObject&&) noexcept;
        ~BufferObject();

        void Bind();
        void BindBase(uint32_t index);
        static void Unbind();
        GLuint Capacity() const noexcept;
        // Assumes that current BufferObject is bounded
        void PopulateData(const void* data, uint32_t size) const;

        static GLuint CurrentBinding();

    private:
        // It is not practical to create 4GB arrays, so uint32 size is used
        uint32_t capacity_;
        GLuint id_ = GL_INVALID_INDEX;
    };

    template <BufferObjectType T>
    BufferObject<T>::BufferObject(uint32_t capacity)
        : capacity_(capacity) {
        glGenBuffers(1, &id_);
        Bind();
        glBufferData(EnumFromType<T>(), capacity_, NULL, GL_STATIC_DRAW);
        Unbind();
    }

    template <BufferObjectType T>
    BufferObject<T>::BufferObject(BufferObject&& other) noexcept {
        capacity_ = other.capacity_;
        std::swap(id_, other.id_);
    }

    template <BufferObjectType T>
    BufferObject<T>& BufferObject<T>::operator=(BufferObject&& other) noexcept {
        assert(id_ == GL_INVALID_INDEX);
        capacity_ = other.capacity_;
        std::swap(id_, other.id_);
        return *this;
    }

    template <BufferObjectType T>
    BufferObject<T>::~BufferObject() {
        if (id_ != GL_INVALID_INDEX) {
            glDeleteBuffers(1, &id_);
        }
    }

    template <BufferObjectType T>
    void BufferObject<T>::Bind() {
        assert(CurrentBinding() != id_);
        glBindBuffer(EnumFromType<T>(), id_);
    }

    template <BufferObjectType T>
    void BufferObject<T>::BindBase(uint32_t index) {
        glBindBufferBase(EnumFromType<T>(), index, id_);
    }

    template <BufferObjectType T>
    void BufferObject<T>::Unbind() {
        glBindBuffer(EnumFromType<T>(), 0);
    }

    template <BufferObjectType T>
    GLuint BufferObject<T>::Capacity() const noexcept {
        return capacity_;
    }

    template <BufferObjectType T>
    void BufferObject<T>::PopulateData(const void* data, uint32_t size) const {
        assert(capacity_ >= size);
        assert(CurrentBinding() == id_);
        glBufferSubData(EnumFromType<T>(), 0, size, data);
    }

    template <BufferObjectType T>
    GLuint BufferObject<T>::CurrentBinding() {
        // glGetIntegerv does not have a unsigned version
        GLint result;
        switch (T) {
            case BufferObjectType::Vertex:
                glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &result);
                break;
            case BufferObjectType::Element:
                glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &result);
                break;
            case BufferObjectType::Uniform:
                glGetIntegerv(GL_UNIFORM_BUFFER_BINDING, &result);
                break;
            default:
                assert(false);
        }
        return result;
    }

    using VertexBufferObject = BufferObject<BufferObjectType::Vertex>;
    using ElementBufferObject = BufferObject<BufferObjectType::Element>;
    using UniformBufferObject = BufferObject<BufferObjectType::Uniform>;

} // namespace Core
