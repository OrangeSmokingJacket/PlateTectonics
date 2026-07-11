#include "app_layer.hpp"

#include <Core/Application/application.hpp>
#include <Core/Engine/Components/EntityManager/basic_entities.hpp>
#include <Core/Engine/Components/Scriptable/scriptable.hpp>
#include <Core/Inputs/inputs.hpp>
#include <Math/matrix_transform.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// TODO: move to separate file
class OrbitCamera : public Core::Engine::Components::Scriptable {
public:
    void Update(float delta_time) override {
        auto* transform_component = entity_->GetComponent<Core::Engine::Components::Transform>();
        auto* window_ptr = Core::Application::Get()->GetMainWindow();

        // Camera movement
        math::vec3<float> input_dir;
        if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::LeftShift)) {
            if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::W)) {
                input_dir[2] -= zoom_speed_ * delta_time;
            }
            if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::S)) {
                input_dir[2] += zoom_speed_ * delta_time;
            }
        } else {
            if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::W)) {
                input_dir[1] -= rotation_speed_ * delta_time;
            }
            if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::S)) {
                input_dir[1] += rotation_speed_ * delta_time;
            }
        }
        if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::D)) {
            input_dir[0] += rotation_speed_ * delta_time;
        }
        if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::A)) {
            input_dir[0] -= rotation_speed_ * delta_time;
        }

        auto& parent_rotation =
            std::get<Core::Engine::Components::EulerAngles>(transform_component->parent_transform->rotation);
        // TODO: set once on start
        parent_rotation.ordering = Core::Engine::Components::EulerRotationOrdering::XYZ;

        parent_rotation.rotation_degrees.x =
            std::clamp(parent_rotation.rotation_degrees.x + input_dir[1], -max_latitude_angle_, max_latitude_angle_);
        // TODO: rotating continuously in one direction will lose precision
        parent_rotation.rotation_degrees.y += input_dir[0];

        transform_component->position.z =
            std::clamp(transform_component->position.z + input_dir[2], min_distance_, max_distance_);
    }

private:
    math::vec2<double> last_mouse_pos_;
    float rotation_speed_ = 50.0f;
    float zoom_speed_ = 5.0f;
    float max_latitude_angle_ = 90.0f;
    float min_distance_ = 1.0f;
    float max_distance_ = 100.0f;
};

constexpr size_t BUFFER_ALLOCATION_SIZE = 32 * 1024 * 1024; // 32 MB

MainLayer::MainLayer()
    : asset_manager_(&thread_pool_)
    , pipeline_(BUFFER_ALLOCATION_SIZE)
    , mantle_sim_(&pipeline_, &thread_pool_) {
    // TODO: this has to be done async
    asset_manager_.LoadAssetsEntries();

    auto camera_parent = scene_.GetEntityManager().AddEntity();
    camera_parent->AddComponent<Core::Engine::Components::Transform>();

    auto camera_obj = scene_.GetEntityManager().AddEntity();
    camera_obj->AddComponent<Core::Engine::Components::Transform>();
    camera_obj->GetComponent<Core::Engine::Components::Transform>()->position.z += 25.0f;
    camera_obj->GetComponent<Core::Engine::Components::Transform>()->parent_transform =
        camera_parent->GetComponent<Core::Engine::Components::Transform>();
    camera_obj->AddComponent<OrbitCamera>();
    camera_obj->AddComponent<Core::Engine::Components::Camera>(math::to_radians(60.0f), 0.01f, 1000.0f);

    // TODO: call Start() on Scriptable entities
}

void MainLayer::Update([[maybe_unused]] float delta_time) {
    last_delta_time_ = delta_time;
    scene_.GetEntityManager().ApplyChanges();
    HandleInputs();

    mantle_sim_.Update(delta_time);

    // TODO: this has to be handled by ECS
    for (auto& entity : scene_.GetEntityManager().Entities()) {
        // Update transform
        entity->GetComponent<Core::Engine::Components::Transform>()->UpdateWorldMatrix();
        // Call each Scriptable

        if (auto cam = entity->GetComponent<OrbitCamera>()) {
            cam->Update(delta_time);
        }
    }
}

void MainLayer::Render() {
    // TODO: render graph should select all passes
    const Core::Window* window_ptr = Core::Application::Get()->GetMainWindow();
    // ImGUI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // TODO: Those OpenGL calls should be abstracted
    glViewport(0, 0, window_ptr->Width(), window_ptr->Height());
    glClearColor(background_color_.r, background_color_.g, background_color_.b, background_color_.a);

    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    // TODO: Culling
    // TODO: transparent sorting
    // TODO: sort by material to minimize program switches
    // TODO: preload assets before rendering

    Core::CameraMatrices matrices{
        .view_matrix = scene_.MainCamera()->ViewMatrix(
            scene_.MainCamera()->GetEntity()->GetComponent<Core::Engine::Components::Transform>()),
        .perspective_matrix = scene_.MainCamera()->PerspectiveMatrix(window_ptr->AspectRation())};
    pipeline_.BindCameraMatrices(matrices);

    // Unlit render pass

    struct RenderParamPack {
        const math::mat4x4<float>& world_matrix;
        const Core::Engine::Components::MeshRenderer* renderer;
    };
    std::vector<RenderParamPack> opaque_render_list;
    std::vector<RenderParamPack> transparent_render_list;

    // TODO: all iteration should be handled by systems
    for (auto& entity : scene_.GetEntityManager().Entities()) {
        const auto* renderer_component = entity->GetComponent<Core::Engine::Components::MeshRenderer>();
        if (!renderer_component) {
            continue;
        }
        const auto* transform_component = entity->GetComponent<Core::Engine::Components::Transform>();

        if (Core::Engine::Assets::IsOpaque(renderer_component->MaterialRef().GetAsset()->GetParameters().blend_mode)) {
            opaque_render_list.emplace_back(transform_component->world_matrix, renderer_component);
        } else {
            transparent_render_list.emplace_back(transform_component->world_matrix, renderer_component);
        }
    }

    auto render_from_list = [this](const std::vector<RenderParamPack>& render_list) {
        for (auto& render_target : render_list) {
            const auto* mesh = render_target.renderer->MeshRef().GetAsset();
            const auto* material = render_target.renderer->MaterialRef().GetAsset();

            pipeline_.Init();

            pipeline_.BindMesh(mesh, material);
            pipeline_.SetModelMatrix(render_target.world_matrix);

            pipeline_.DrawElements(mesh, material);
            pipeline_.Cleanup();
        }
    };

    render_from_list(opaque_render_list);
    render_from_list(transparent_render_list);

    // Slice pass
    math::vec2<float> resolution{static_cast<float>(window_ptr->Width()), static_cast<float>(window_ptr->Height())};

    // Mantle slice pass
    if (mantle_sim_.draw_slice != EulerFluidSim::DrawSlice::None) {
        pipeline_.Init();

        pipeline_.BindSlice(mantle_sim_.RenderTextures(),
                            mantle_sim_.max_temperature_color,
                            // TODO: account for scale
                            math::vec3<float>{mantle_sim_.full_dimensions.x,
                                              mantle_sim_.full_dimensions.y,
                                              mantle_sim_.full_dimensions.z},
                            resolution,
                            0);
        pipeline_.SetModelMatrix(math::mat4x4<float>::identity());
        pipeline_.DrawSlice();

        pipeline_.Cleanup();
    }
    // Debug Draw pass

    mantle_sim_.RenderDebug(&pipeline_);

    pipeline_.DebugDrawAll();

    // UI after rendering
    HandleImGUI();
}

void MainLayer::HandleInputs() {
    auto* window_ptr = Core::Application::Get()->GetMainWindow();

    // Exit
    if (Core::Inputs::IsKeyPressed(window_ptr, Core::Inputs::KeyCode::Escape)) {
        Core::Application::Get()->Stop();
    }
}

void HandleDrawSlice(EulerFluidSim::DrawSlice* target, EulerFluidSim::DrawSlice value) {
    *target = (*target == value) ? EulerFluidSim::DrawSlice::None : value;
}

void MainLayer::HandleImGUI() {
    ImGui::Begin("Settings");
    ImGui::Text("FPS: %.1f", 1.0f / last_delta_time_);
    ImGui::Text("Frame duration: %.1f ms", last_delta_time_ * 1000.0f);
    if (ImGui::CollapsingHeader("Background")) {
        ImGui::Indent(10.0f);

        ImGui::ColorPicker3("Color", background_color_.data(), ImGuiColorEditFlags_PickerHueWheel);

        ImGui::Unindent(10.0f);
    }
    if (ImGui::CollapsingHeader("Mantle Simulation")) {
        ImGui::Indent(10.0f);

        if (ImGui::CollapsingHeader("Physical Constants")) {
            ImGui::Indent(10.0f);

            ImGui::SliderScalar("Thermal Expansion Coefficient",
                                ImGuiDataType_Double,
                                &mantle_sim_.thermal_expansion_coefficient,
                                &mantle_sim_.min_thermal_expansion_coefficient,
                                &mantle_sim_.max_thermal_expansion_coefficient,
                                "{%.8f}",
                                ImGuiSliderFlags_Logarithmic);

            ImGui::Unindent(10.0f);
        }
        if (ImGui::CollapsingHeader("Velocities")) {
            ImGui::Indent(10.0f);

            if (ImGui::Button("Randomize Border Velocities")) {
                mantle_sim_.DebugRandomizeBorderVelocities();
            }
            ImGui::Checkbox("Draw Boundary Velocities", &mantle_sim_.draw_boundary_velocities);
            ImGui::Checkbox("Interpolate Velocities", &mantle_sim_.draw_interpolated_velocities);
            ImGui::SliderInt("Per Cell interpolations",
                             &mantle_sim_.cell_subdivision,
                             1,
                             mantle_sim_.max_cell_subdivision);

            ImGui::Unindent(10.0f);
        }
        if (ImGui::CollapsingHeader("Temperature")) {
            ImGui::Indent(10.0f);
            ImGui::PushID("Temperature_Section");

            if (ImGui::Button("Draw Temperature Slice")) {
                HandleDrawSlice(&mantle_sim_.draw_slice, EulerFluidSim::DrawSlice::Temperature);
            }
            if (ImGui::CollapsingHeader("Colors")) {
                ImGui::ColorPicker3("Coldest",
                                    mantle_sim_.min_temperature_color.data(),
                                    ImGuiColorEditFlags_PickerHueWheel);
                ImGui::ColorPicker3("Hottest",
                                    mantle_sim_.max_temperature_color.data(),
                                    ImGuiColorEditFlags_PickerHueWheel);
            }

            ImGui::PopID();
            ImGui::Unindent(10.0f);
        }
        if (ImGui::CollapsingHeader("Solver")) {
            ImGui::Indent(10.0f);

            ImGui::Text("Last PCG Iterations: %d", mantle_sim_.last_PCG_iteration_count);
            if (ImGui::CollapsingHeader("Pressure")) {
                ImGui::Indent(10.0f);
                ImGui::PushID("Pressure_Section");

                if (ImGui::Button("Draw Pressure Slice")) {
                    HandleDrawSlice(&mantle_sim_.draw_slice, EulerFluidSim::DrawSlice::Pressure);
                }
                if (ImGui::CollapsingHeader("Colors")) {
                    ImGui::ColorPicker3("Min",
                                        mantle_sim_.min_pressure_color.data(),
                                        ImGuiColorEditFlags_PickerHueWheel);
                    ImGui::ColorPicker3("Max",
                                        mantle_sim_.max_pressure_color.data(),
                                        ImGuiColorEditFlags_PickerHueWheel);
                }

                ImGui::PopID();
                ImGui::Unindent(10.0f);
            }
            if (ImGui::CollapsingHeader("Divergence")) {
                ImGui::Indent(10.0f);
                ImGui::PushID("Divergence_Section");

                if (ImGui::Button("Draw Pressure Divergence Slice")) {
                    HandleDrawSlice(&mantle_sim_.draw_slice, EulerFluidSim::DrawSlice::PressureDivergence);
                }
                if (ImGui::CollapsingHeader("Colors")) {
                    ImGui::ColorPicker3("Min",
                                        mantle_sim_.min_divergence_color.data(),
                                        ImGuiColorEditFlags_PickerHueWheel);
                    ImGui::ColorPicker3("Max",
                                        mantle_sim_.max_divergence_color.data(),
                                        ImGuiColorEditFlags_PickerHueWheel);
                }

                ImGui::PopID();
                ImGui::Unindent(10.0f);
            }

            ImGui::Unindent(10.0f);
        }

        ImGui::Unindent(10.0f);
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
