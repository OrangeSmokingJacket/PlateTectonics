#pragma once

#include <Math/math_def.hpp>

#include <cstdint>

namespace Core::Engine::Assets {

	// TODO: conversion to OpenGl GLenum
	// TODO: custom 1 byte or OpenGL style 4 bytes?


	enum class TexelFormat : uint8_t {
		RedChanel,							// 1
		GreenChanel,						// 1
		BlueChanel,							// 1
		AlphaChanel,						// 1
		RGB,								// 3
		BGR,								// 3
		RGBA,								// 4
		BGRA,								// 4
		Luminance,							// 1
		LuminanceAlpha						// 2
	};

	enum class TexelType : uint8_t {
		uint8,							// [0, 255]
		int8,							// [-128, 127]
		uint16,							// [0, 65'535]
		int16,							// [-32'768, 32'767]
		uint32,							// [0, 4'294'967'295]
		int32,							// [0, 2'147'483'647]
		float32,						// [0.0f, 1.0f]
		bitpack_3_3_2,					// [0, 7][0, 7][0, 3]
		bitpack_2_3_3_rev,				// [0, 3][0, 7][0, 7]
		bitpack_5_6_5,					// [0, 31][0, 63][0, 31]
		bitpack_5_6_5_rev,				// [0, 31][0, 63][0, 31]
		bitpack_4_4_4_4,				// [0, 15][0, 15][0, 15][0, 15]
		bitpack_4_4_4_4_rev,			// [0, 15][0, 15][0, 15][0, 15]
		bitpack_5_5_5_1,				// [0, 31][0, 31][0, 31][0, 1]
		bitpack_1_5_5_5_rev,			// [0, 1][0, 31][0, 31][0, 31]
		bitpack_8_8_8_8,				// [0, 255][0, 255][0, 255][0, 255]
		bitpack_8_8_8_8_rev,			// [0, 255][0, 255][0, 255][0, 255]
		bitpack_10_10_10_2,				// [0, 1023][0, 1023][0, 1023][0, 3]
		bitpack_2_10_10_10_rev,			// [0, 3][0, 1023][0, 1023][0, 1023]
	};

	namespace impl {

		// I wish it could be more generic, but specialization has to be defined in the same file
		template<TexelType Type>
		struct TexelTypeMapping;

		template <> struct TexelTypeMapping<TexelType::uint8> {using type = uint8_t; };
		template <> struct TexelTypeMapping<TexelType::int8> { using type = int8_t; };
		template <> struct TexelTypeMapping<TexelType::uint16> { using type = uint16_t; };
		template <> struct TexelTypeMapping<TexelType::int16> { using type = int16_t; };
		template <> struct TexelTypeMapping<TexelType::uint32> { using type = uint32_t; };
		template <> struct TexelTypeMapping<TexelType::int32> { using type = int32_t; };
		template <> struct TexelTypeMapping<TexelType::float32> { using type = float; };
		template <> struct TexelTypeMapping<TexelType::bitpack_3_3_2> {
			using type = struct {
				uint8_t r : 3;
				uint8_t g : 3;
				uint8_t b : 2;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_2_3_3_rev> {
			using type = struct {
				uint8_t b : 2;
				uint8_t g : 3;
				uint8_t r : 3;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_5_6_5> {
			using type = struct {
				uint8_t r : 5;
				uint8_t g : 6;
				uint8_t b : 5;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_5_6_5_rev> {
			using type = struct {
				uint8_t b : 5;
				uint8_t g : 6;
				uint8_t r : 5;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_4_4_4_4> {
			using type = struct {
				uint8_t r : 4;
				uint8_t g : 4;
				uint8_t b : 4;
				uint8_t a : 4;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_4_4_4_4_rev> {
			using type = struct {
				uint8_t a : 4;
				uint8_t b : 4;
				uint8_t g : 4;
				uint8_t r : 4;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_5_5_5_1> {
			using type = struct {
				uint8_t r : 5;
				uint8_t g : 5;
				uint8_t b : 5;
				uint8_t a : 1;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_1_5_5_5_rev> {
			using type = struct {
				uint8_t a : 1;
				uint8_t b : 5;
				uint8_t g : 5;
				uint8_t r : 5;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_8_8_8_8> {
			using type = struct {
				uint8_t r;
				uint8_t g;
				uint8_t b;
				uint8_t a;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_8_8_8_8_rev> {
			using type = struct {
				uint8_t a;
				uint8_t b;
				uint8_t g;
				uint8_t r;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_10_10_10_2> {
			using type = struct {
				uint16_t r : 10;
				uint16_t g : 10;
				uint16_t b : 10;
				uint16_t a : 2;
			};
		};
		template <> struct TexelTypeMapping<TexelType::bitpack_2_10_10_10_rev> {
			using type = struct {
				uint16_t a : 2;
				uint16_t b : 10;
				uint16_t g : 10;
				uint16_t r : 10;
			};
		};


		template<typename T, TexelFormat Format>
		struct TexelFormatMapping;

		template <typename T> struct TexelFormatMapping<T, TexelFormat::RedChanel> { using type = T; };
		template <typename T> struct TexelFormatMapping<T, TexelFormat::GreenChanel> { using type = T; };
		template <typename T> struct TexelFormatMapping<T, TexelFormat::BlueChanel> { using type = T; };
		template <typename T> struct TexelFormatMapping<T, TexelFormat::AlphaChanel> { using type = T; };
		template <typename T> struct TexelFormatMapping<T, TexelFormat::Luminance> { using type = T; };
		template <typename T> requires (!std::is_class_v<T>)
		struct TexelFormatMapping<T, TexelFormat::LuminanceAlpha> { using type = math::vec2<T>; };
		template <typename T> struct TexelFormatMapping<T, TexelFormat::RGB> {
			using type = std::conditional_t<std::is_class_v<T>, T, math::vec3<T>>;
		};
		template <typename T> struct TexelFormatMapping<T, TexelFormat::BGR> {
			using type = std::conditional_t<std::is_class_v<T>, T, math::vec3<T>>;
		};
		template <typename T> struct TexelFormatMapping<T, TexelFormat::RGBA> {
			using type = std::conditional_t<std::is_class_v<T>, T, math::vec4<T>>;
		};
		template <typename T> struct TexelFormatMapping<T, TexelFormat::BGRA> {
			using type = std::conditional_t<std::is_class_v<T>, T, math::vec4<T>>;
		};

		template<TexelType T, TexelFormat F>
		using TexelData = TexelFormatMapping<typename TexelTypeMapping<T>::type, F>::type;

	}
}