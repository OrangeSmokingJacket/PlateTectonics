set(LIB_NAME asset_ref)

set(HEADERS
    asset_ref.hpp
)

add_library(${LIB_NAME} INTERFACE ${HEADERS})

include_directories(${PROJECT_SOURCE_DIR})

target_link_libraries(${LIB_NAME} INTERFACE
        base_asset
        asset_manager
)
