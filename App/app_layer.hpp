#pragma once

#include <Core/Application/layer.hpp>
#include <Core/Engine/Assets/AssetManager/asset_manager.hpp>
#include <Core/Engine/Scene/scene.hpp>
#include <Core/Engine/Components/Camera/camera.hpp>
#include <Core/Renderer/renderer.hpp>
#include "euler_fluid_sim.hpp"


class MainLayer : public Core::BaseLayer {
public:
    MainLayer();

    void Update(float delta_time) override;
    void Render() override;

private:
    void HandleInputs();
    void HandleImGUI();

    Core::Engine::Assets::AssetManager asset_manager_;
    Core::Engine::Scene scene_;
    Core::RenderPipeline pipeline_;
    EulerFluidSim mantle_sim_;
    // Background Color = #0e1945 -> 14, 25, 69
    Core::Engine::Assets::Color background_color_{14.0f / 255.0f, 25.0f / 255.0f, 69.0f / 255.0f, 1.0f};
    // store for ImGui display
    float last_delta_time_ = 1.0f;
};
