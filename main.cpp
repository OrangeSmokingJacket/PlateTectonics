#include <App/app_layer.hpp>
#include <Core/Application/application.hpp>

int main() {
    Core::AppConfig config{"PlateTectonics",
                           Core::WindowConfig{.name = "PlateTectonics",
                                              .width = 1080,
                                              .height = 720,
                                              .is_resizable = false,
                                              .v_sync = false}};
    Core::Application app(config);
    app.AddLayer(std::make_unique<MainLayer>());
    app.RunMainLoop();

    return 0;
}
