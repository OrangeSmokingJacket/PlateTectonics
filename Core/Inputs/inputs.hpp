#pragma once

#include <Math/math_def.hpp>

#include <cstdint>

namespace Core {
    class Window;
}

namespace Core::Inputs {

    // TODO: other keys
    enum class KeyCode {
        Node,
        Space,
        Comma,
        Minus,
        Period,
        Slash,
        Number0,
        Number1,
        Number2,
        Number3,
        Number4,
        Number5,
        Number6,
        Number7,
        Number8,
        Number9,
        Semicolon,
        Equal,
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        N,
        O,
        P,
        Q,
        R,
        S,
        T,
        U,
        V,
        W,
        X,
        Y,
        Z,
        LeftBracket,
        RightBracket,
        Backslash,
        Escape,
        Enter,
        Tab,
        Backspace,
        Insert,
        Delete,
        RightArrow,
        LeftArrow,
        DownArrow,
        UpArrow,
        PageUp,
        PageDown,
        Home,
        End,
        CapsLock,
        ScrollLock,
        NumLock,
        PrintScreen,
        Pause,
        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,
        NumPad0,
        NumPad1,
        NumPad2,
        NumPad3,
        NumPad4,
        NumPad5,
        NumPad6,
        NumPad7,
        NumPad8,
        NumPad9,
        NumPadPeriod,
        NumPadDivide,
        NumPadMultiply,
        NumPadSubtract,
        NumPadAdd,
        NumPadEnter,
        NumPadEqual,
        LeftShift,
        LeftControl,
        LeftAlt,
        RightShift,
        RightControl,
        RightAlt,
        Menu,

        Mouse1,
        Mouse2,
        Mouse3,
        Mouse4,
        Mouse5,
        Mouse6,
        Mouse7,
        Mouse8,
    };

    enum class KeyState : uint8_t {
        Pressed,
        Held,
        Released,
    };

    KeyState GetKeyState(Window* window, KeyCode code);
    bool IsKeyPressed(Window* window, KeyCode code);
    bool IsKeyHeld(Window* window, KeyCode code);
    bool IsKeyReleased(Window* window, KeyCode code);

    // [0,0] is bottom left
    math::vec2<double> GetMousePos(Window* window);

    //TODO: scroll wheel input

}
