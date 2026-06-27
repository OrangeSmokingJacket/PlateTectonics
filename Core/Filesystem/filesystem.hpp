#pragma once

#include <filesystem>

namespace Core::Filesystem {

	using buffer_t = std::string;

	buffer_t ReadFullFile(const std::filesystem::path& path);

}