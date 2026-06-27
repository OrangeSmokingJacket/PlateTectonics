#include "inputs.hpp"

#include <Core/Application/window.hpp>
#include <cassert>


namespace Core::Inputs {

    KeyState FromGLFW(int glfw_state) {
        switch (glfw_state) {
        case GLFW_PRESS:
            return KeyState::Pressed;
        case GLFW_REPEAT:
            return KeyState::Held;
        case GLFW_RELEASE:
            return KeyState::Released;
        default:
            assert(false && "Unknown GLFW key state");
        }
    }

    KeyState GetKeyState(Window* window, KeyCode code) {
        GLFWwindow* window_ptr = window->GetGLFWWindow();
        switch (code) {
        case KeyCode::Space:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_SPACE));
        case KeyCode::Comma:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_COMMA));
        case KeyCode::Minus:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_MINUS));
        case KeyCode::Period:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_PERIOD));
        case KeyCode::Slash:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_SLASH));
        case KeyCode::Number0:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_0));
        case KeyCode::Number1:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_1));
        case KeyCode::Number2:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_2));
        case KeyCode::Number3:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_3));
        case KeyCode::Number4:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_4));
        case KeyCode::Number5:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_5));
        case KeyCode::Number6:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_6));
        case KeyCode::Number7:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_7));
        case KeyCode::Number8:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_8));
        case KeyCode::Number9:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_9));
        case KeyCode::Semicolon:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_SEMICOLON));
        case KeyCode::Equal:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_EQUAL));
        case KeyCode::A:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_A));
        case KeyCode::B:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_B));
        case KeyCode::C:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_C));
        case KeyCode::D:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_D));
        case KeyCode::E:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_E));
        case KeyCode::F:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F));
        case KeyCode::G:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_G));
        case KeyCode::H:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_H));
        case KeyCode::I:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_I));
        case KeyCode::J:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_J));
        case KeyCode::K:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_K));
        case KeyCode::L:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_L));
        case KeyCode::M:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_M));
        case KeyCode::N:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_N));
        case KeyCode::O:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_O));
        case KeyCode::P:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_P));
        case KeyCode::Q:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_Q));
        case KeyCode::R:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_R));
        case KeyCode::S:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_S));
        case KeyCode::T:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_T));
        case KeyCode::U:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_U));
        case KeyCode::V:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_V));
        case KeyCode::W:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_W));
        case KeyCode::X:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_X));
        case KeyCode::Y:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_Y));
        case KeyCode::Z:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_Z));
        case KeyCode::LeftBracket:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_LEFT_BRACKET));
        case KeyCode::RightBracket:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_RIGHT_BRACKET));
        case KeyCode::Backslash:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_BACKSLASH));
        case KeyCode::Escape:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_ESCAPE));
        case KeyCode::Enter:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_ENTER));
        case KeyCode::Tab:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_TAB));
        case KeyCode::Backspace:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_BACKSPACE));
        case KeyCode::Insert:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_INSERT));
        case KeyCode::Delete:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_DELETE));
        case KeyCode::RightArrow:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_RIGHT));
        case KeyCode::LeftArrow:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_LEFT));
        case KeyCode::DownArrow:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_DOWN));
        case KeyCode::UpArrow:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_UP));
        case KeyCode::PageUp:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_PAGE_UP));
        case KeyCode::PageDown:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_PAGE_DOWN));
        case KeyCode::Home:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_HOME));
        case KeyCode::End:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_END));
        case KeyCode::CapsLock:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_CAPS_LOCK));
        case KeyCode::ScrollLock:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_SCROLL_LOCK));
        case KeyCode::NumLock:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_NUM_LOCK));
        case KeyCode::PrintScreen:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_PRINT_SCREEN));
        case KeyCode::Pause:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_PAUSE));
        case KeyCode::F1:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F1));
        case KeyCode::F2:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F2));
        case KeyCode::F3:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F3));
        case KeyCode::F4:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F4));
        case KeyCode::F5:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F5));
        case KeyCode::F6:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F6));
        case KeyCode::F7:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F7));
        case KeyCode::F8:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F8));
        case KeyCode::F9:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F9));
        case KeyCode::F10:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F10));
        case KeyCode::F11:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F11));
        case KeyCode::F12:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_F12));
        case KeyCode::NumPad0:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_0));
        case KeyCode::NumPad1:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_1));
        case KeyCode::NumPad2:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_2));
        case KeyCode::NumPad3:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_3));
        case KeyCode::NumPad4:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_4));
        case KeyCode::NumPad5:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_5));
        case KeyCode::NumPad6:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_6));
        case KeyCode::NumPad7:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_7));
        case KeyCode::NumPad8:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_8));
        case KeyCode::NumPad9:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_9));
        case KeyCode::NumPadPeriod:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_DECIMAL));
        case KeyCode::NumPadDivide:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_DIVIDE));
        case KeyCode::NumPadMultiply:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_MULTIPLY));
        case KeyCode::NumPadSubtract:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_SUBTRACT));
        case KeyCode::NumPadAdd:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_ADD));
        case KeyCode::NumPadEnter:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_ENTER));
        case KeyCode::NumPadEqual:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_KP_EQUAL));
        case KeyCode::LeftShift:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_LEFT_SHIFT));
        case KeyCode::LeftControl:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_LEFT_CONTROL));
        case KeyCode::LeftAlt:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_LEFT_ALT));
        case KeyCode::RightShift:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_RIGHT_SHIFT));
        case KeyCode::RightControl:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_RIGHT_CONTROL));
        case KeyCode::RightAlt:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_RIGHT_ALT));
        case KeyCode::Menu:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_KEY_MENU));
        case KeyCode::Mouse1:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_1));
        case KeyCode::Mouse2:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_2));
        case KeyCode::Mouse3:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_3));
        case KeyCode::Mouse4:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_4));
        case KeyCode::Mouse5:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_5));
        case KeyCode::Mouse6:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_6));
        case KeyCode::Mouse7:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_7));
        case KeyCode::Mouse8:
            return FromGLFW(glfwGetKey(window_ptr, GLFW_MOUSE_BUTTON_8));
        default:
            assert(false && "Unknown KeyCode in GetKeyState");
        }
    }

    bool IsKeyPressed(Window* window, KeyCode code) {
        return GetKeyState(window, code) == KeyState::Pressed;
    }

    bool IsKeyHeld(Window* window, KeyCode code) {
        return GetKeyState(window, code) == KeyState::Held;
    }

    bool IsKeyReleased(Window* window, KeyCode code) {
        return GetKeyState(window, code) == KeyState::Released;
    }

    math::vec2<double> GetMousePos(Window* window) {
        math::vec2<double> result;
        glfwGetCursorPos(window->GetGLFWWindow(), &result.x, &result.y);
        result.y *= -1;
        return result;
    }
}
