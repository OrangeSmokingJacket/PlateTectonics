#pragma once

// order matters here
// clang-format off
#include <ThirdParty/glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <cassert>
#include <utility>

namespace Core {

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

    // TODO: other OpenGL texture types
    enum class TextureObjectType : uint8_t {
        Texture1D,
        Texture2D,
        Texture3D,
    };

    template <TextureObjectType T>
    class TextureObject {
        static consteval GLenum EnumFromType() {
            switch (T) {
                case TextureObjectType::Texture1D:
                    return GL_TEXTURE_1D;
                case TextureObjectType::Texture2D:
                    return GL_TEXTURE_2D;
                case TextureObjectType::Texture3D:
                    return GL_TEXTURE_3D;
                default:
                    return GL_INVALID_ENUM;
            }
        }

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

    using TextureObject1D = TextureObject<TextureObjectType::Texture1D>;
    using TextureObject2D = TextureObject<TextureObjectType::Texture2D>;
    using TextureObject3D = TextureObject<TextureObjectType::Texture3D>;

    enum class BufferObjectType : uint8_t {
        Vertex,
        Element,
        Uniform,
    };

    // TODO: multiple sub-buffers
    template <BufferObjectType T>
    class BufferObject {
        static consteval GLenum EnumFromType() {
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

    public:
        explicit BufferObject(uint32_t capacity);
        BufferObject(const BufferObject&) = delete;
        BufferObject& operator=(const BufferObject&) = delete;
        BufferObject(BufferObject&&) noexcept;
        BufferObject& operator=(BufferObject&&) noexcept;
        ~BufferObject();

        GLuint Id() const noexcept;
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

    using VertexBufferObject = BufferObject<BufferObjectType::Vertex>;
    using ElementBufferObject = BufferObject<BufferObjectType::Element>;
    using UniformBufferObject = BufferObject<BufferObjectType::Uniform>;

    template <TextureObjectType T>
    TextureObject<T>::TextureObject() {
        glGenTextures(1, &id_);
        assert(glGetError() == GL_NO_ERROR);
    }

    template <TextureObjectType T>
    TextureObject<T>::TextureObject(TextureObject&& other) noexcept {
        std::swap(id_, other.id_);
    }

    template <TextureObjectType T>
    TextureObject<T>& TextureObject<T>::operator=(TextureObject&& other) noexcept {
        assert(id_ == GL_INVALID_INDEX);
        std::swap(id_, other.id_);
        return *this;
    }

    template <TextureObjectType T>
    TextureObject<T>::~TextureObject() {
        if (id_ != GL_INVALID_INDEX) {
            glDeleteTextures(1, &id_);
        }
    }

    template <TextureObjectType T>
    void TextureObject<T>::Bind() {
        assert(CurrentBinding() == 0);
        assert(id_ != GL_INVALID_INDEX);
        glBindTexture(EnumFromType(), id_);
    }

    template <TextureObjectType T>
    void TextureObject<T>::Unbind() {
        glBindTexture(EnumFromType(), 0);
    }

    template <TextureObjectType T>
    void TextureObject<T>::SetActiveUnit(uint32_t index) {
        glActiveTexture(GL_TEXTURE0 + index);
    }

    template <TextureObjectType T>
    GLuint TextureObject<T>::CurrentBinding() {
        GLint result;
        switch (T) {
            case TextureObjectType::Texture1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &result);
                break;
            case TextureObjectType::Texture2D:
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &result);
                break;
            case TextureObjectType::Texture3D:
                glGetIntegerv(GL_TEXTURE_BINDING_3D, &result);
                break;
            default:
                assert(false);
        }
        return result;
    }

    template <BufferObjectType T>
    BufferObject<T>::BufferObject(uint32_t capacity)
        : capacity_(capacity) {
        glGenBuffers(1, &id_);
        Bind();
        glBufferData(EnumFromType(), capacity_, NULL, GL_STATIC_DRAW);
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
    GLuint BufferObject<T>::Id() const noexcept {
        return id_;
    }

    template <BufferObjectType T>
    void BufferObject<T>::Bind() {
        assert(CurrentBinding() != id_);
        glBindBuffer(EnumFromType(), id_);
    }

    template <BufferObjectType T>
    void BufferObject<T>::BindBase(uint32_t index) {
        glBindBufferBase(EnumFromType(), index, id_);
    }

    template <BufferObjectType T>
    void BufferObject<T>::Unbind() {
        glBindBuffer(EnumFromType(), 0);
    }

    template <BufferObjectType T>
    GLuint BufferObject<T>::Capacity() const noexcept {
        return capacity_;
    }

    template <BufferObjectType T>
    void BufferObject<T>::PopulateData(const void* data, uint32_t size) const {
        assert(capacity_ >= size);
        assert(CurrentBinding() == id_);
        glBufferSubData(EnumFromType(), 0, size, data);
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

} // namespace Core
