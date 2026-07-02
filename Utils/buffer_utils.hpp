#pragma once

#include <cereal/archives/binary.hpp>

#include <initializer_list>
#include <memory>
#include <type_traits>

namespace utils {

    template <typename T>
    struct Buffer {
        Buffer() = default;
        explicit Buffer(size_t size);
        explicit Buffer(T* data, size_t size);
        explicit Buffer(std::initializer_list<T> i_list);
        Buffer(const Buffer&) requires(std::is_trivially_copyable_v<T>);
        Buffer& operator=(const Buffer&) requires(std::is_trivially_copyable_v<T>);
        Buffer(Buffer&&) noexcept = default;
        Buffer& operator=(Buffer&&) noexcept = default;
        ~Buffer() = default;

        size_t ByteSize() const noexcept;

        operator bool() const noexcept { return data.get(); }

        T& operator[](size_t index);
        const T& operator[](size_t index) const;

        size_t size = 0;
        std::unique_ptr<T[]> data = nullptr;
    };

    template <typename Archive, typename T>
    requires(std::is_trivially_copyable_v<T>)
    void save(Archive& archive, const Buffer<T>& buffer) {
        archive(buffer.size);
        archive(cereal::binary_data(buffer.data.get(), buffer.size * sizeof(T)));
    }

    template <typename Archive, typename T>
    requires(!std::is_trivially_copyable_v<T>)
    void save(Archive& archive, const Buffer<T>& buffer) {
        archive(buffer.size);
        for (size_t i = 0; i < buffer.size; ++i) {
            archive(buffer[i]);
        }
    }

    template <typename Archive, typename T>
    requires(std::is_trivially_copyable_v<T>)
    void load(Archive& archive, Buffer<T>& buffer) {
        archive(buffer.size);
        if (buffer.size == 0) {
            return;
        }
        buffer.data = std::make_unique<T[]>(buffer.size);
        archive(cereal::binary_data(buffer.data.get(), buffer.size * sizeof(T)));
    }

    template <typename Archive, typename T>
    requires(!std::is_trivially_copyable_v<T>)
    void load(Archive& archive, Buffer<T>& buffer) {
        archive(buffer.size);
        if (buffer.size == 0) {
            return;
        }
        buffer.data = std::make_unique<T[]>(buffer.size);
        for (size_t i = 0; i < buffer.size; ++i) {
            archive(buffer[i]);
        }
    }

    template <typename T>
    Buffer<T>::Buffer(size_t size)
        : size(size)
        , data(std::make_unique<T[]>(size)) {}

    template <typename T>
    Buffer<T>::Buffer(T* data, size_t size)
        : size(size)
        , data(data) {}

    template <typename T>
    Buffer<T>::Buffer(std::initializer_list<T> i_list)
        : size(i_list.size())
        , data(std::make_unique<T[]>(size)) {
        std::copy(i_list.begin(), i_list.end(), data.get());
    }

    template <typename T>
    size_t Buffer<T>::ByteSize() const noexcept {
        return sizeof(T) * size;
    }

    template <typename T>
    T& Buffer<T>::operator[](size_t index) {
        return data[index];
    }

    template <typename T>
    const T& Buffer<T>::operator[](size_t index) const {
        return data[index];
    }

} // namespace utils
