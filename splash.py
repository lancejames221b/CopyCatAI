import PySimpleGUI as sg
import sys, os

if getattr(sys, "frozen", False):
    # If the code is running in a PyInstaller bundle
    bundle_dir = sys._MEIPASS
else:
    # If the code is running in a normal Python environment
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

logo_path = os.path.join(bundle_dir, "logo.png")


def create_splash_screen():
    # Set up the splash screen layout
    layout = [
        [sg.Image(logo_path)],
    ]

    # Create the window
    window = sg.Window(
        "My Application", layout, no_titlebar=True, grab_anywhere=True, keep_on_top=True
    )

    # Read window events with a timeout of 3 seconds (3000 milliseconds) and then close the window
    event, values = window.read(timeout=3000)
    window.close()

    return window
