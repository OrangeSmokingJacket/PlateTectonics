#include "filesystem.hpp"

#include <fstream>
#include <cassert>

namespace Core::Filesystem {

    buffer_t ReadFullFile(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            // TODO: return error
            assert(false);
            return {};
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return std::move(buffer).str();
	}

}