
import pyperclip
import PySimpleGUI as sg
from extract import *
from notification import *
import os
from PIL import ImageGrab, Image
from gptplus import *
from prompt_ui import *
from splash import *
import subprocess
import platform
import traceback
import configparser
import webbrowser
from pathlib import Path

home_dir = os.path.expanduser("~")

bundle_dir = os.path.join(home_dir, "Library", "Application Support", "CopyCat")
Path(bundle_dir).mkdir(parents=True, exist_ok=True)

import subprocess
import platform

def is_tesseract_installed():
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ["/opt/homebrew/bin/tesseract", "-v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        elif platform.system() == 'Linux':
            result = subprocess.run(
                ["tesseract", "-v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        elif platform.system() == 'Windows':
            result = subprocess.run(
                ["tesseract", "-v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,  # Added for Windows compatibility
            )
        else:
            return False
        return result.returncode == 0
    except FileNotFoundError:
        return False

def prompt_tesseract_installation():
    if platform.system() == 'Darwin':
        message = (
            "In order for the screenshot image to text feature to work, Tesseract must be installed.\n\n"
            "Tesseract is not installed on your system. Please install it using the following command:\n\n"
            "Open a terminal (command-spacebar then type 'terminal')\n\n"
            "brew install tesseract\n\n"
            "For more detailed installation instructions, visit:\n"
            "https://formulae.brew.sh/formula/tesseract"
        )
    elif platform.system() == 'Linux':
        message = (
            "In order for the screenshot image to text feature to work, Tesseract must be installed.\n\n"
            "Tesseract is not installed on your system. Please install it using the following command:\n\n"
            "Open a terminal and run the following command:\n\n"
            "sudo apt-get install tesseract-ocr\n\n"
            "For more detailed installation instructions, visit:\n"
            "https://github.com/tesseract-ocr/tesseract/wiki"
        )
    elif platform.system() == 'Windows':
        message = (
            "In order for the screenshot image to text feature to work, Tesseract must be installed.\n\n"
            "Tesseract is not installed on your system. Please install it by following these steps:\n\n"
            "1. Download the Tesseract installer from the following link:\n"
            "https://github.com/UB-Mannheim/tesseract/wiki\n\n"
            "2. Run the installer and follow the installation instructions.\n\n"
            "3. After installation, add the Tesseract installation directory to the system's PATH environment variable.\n\n"
            "For more detailed installation instructions, visit:\n"
            "https://github.com/UB-Mannheim/tesseract/wiki/Installation"
        )
    else:
        return
    sg.popup(message, title="Screenshot Feature")

if not is_tesseract_installed():
    prompt_tesseract_installation()

# Define a function that copies a file specified by 'filename'
# and saves it to the specified location. The parameter 'binary'
# is optional with a default value of False.
def copy_files(filename, binary=False):
    # Check if the script is running in a PyInstaller bundle.
    if getattr(sys, "frozen", False):
        # PyInstaller creates a temp folder and stores the path in _MEIPASS
        # Set the app_config_dir to the path stored in _MEIPASS.
        app_config_dir = sys._MEIPASS
    else:
        # Set the app_config_dir to the directory of the current script.
        app_config_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the path to the bundled file by joining the app_config_dir and 'filename'.
    bundled_config_file = os.path.join(app_config_dir, filename)

    # Create the path to the destination file by joining 'filename' to an undefined variable, bundle_dir.
    # This looks like an error in the code.
    destination_file = os.path.join(bundle_dir, filename)

    # If 'binary' is False, open the bundled file as a text file and write the contents to the destination file as a text file.
    if not binary:
        temp_config = open(bundled_config_file, "r").read()
        with open(destination_file, "w") as f:
            f.write(temp_config)

    # If 'binary' is True, open the bundled file as a binary file and write the contents to the destination file as a binary file.
    else:
        temp_config = open(bundled_config_file, "rb").read()
        with open(destination_file, "wb") as f:
            f.write(temp_config)

    # Set the file permission of the destination file to read and write permission for the owner of the file.
    os.chmod(destination_file, 0o600)

    # Print a message to indicate that the file has been created.
    print("created", destination_file)

logo_path = os.path.join(bundle_dir, "logo.png")
config_path = os.path.join(bundle_dir, "config.ini")
memory_path = os.path.join(bundle_dir, "memory.json")
first_time = os.path.join(bundle_dir, "first_time.txt")

if not os.path.exists(config_path):  # If the config file doesn't exist
    copy_files("config.ini")
if not os.path.exists(logo_path):  # If the logo doesn't exist
    copy_files("logo.png", binary=True)

create_splash_screen()

def if_first_time():
    if not os.path.exists(first_time):
        with open(first_time, "w") as f:
            f.write("1")
        return True
    else:
        return False

def make_memory_file(filepath=memory_path):
    with open(filepath, "w") as f:
        json.dump({}, f, indent=4)

if not os.path.exists(memory_path):
    make_memory_file()

def load_config(filepath):
    """
    Loads the configuration file from the specified filepath and returns a dictionary of the configuration options.
    """

    config = configparser.ConfigParser(strict=False, interpolation=None)
    config.read(filepath)
    return config

def save_config(filepath, config_dict):
    """
    Saves the specified dictionary of configuration options to the specified filepath.
    """

    config = configparser.ConfigParser(strict=False, interpolation=None)
    for section, options in config_dict.items():
        config[section] = options
    with open(filepath, "w") as config_file:
        config.write(config_file)

def is_api_key_empty(api_key):
    if len(api_key.strip()) == 0:
        return True
    else:
        return False

def is_notion_token_empty(token, space_id):
    print("is_notion_token_empty")
    print(token, space_id)
    if len(token.strip()) == 0 or len(space_id.strip()) == 0:
        return True
    else:
        return False

CONFIG = load_config(config_path)

def settings_window():
    global CONFIG
    global max_range

    max_tokens = CONFIG.get("OpenAI", "max_tokens")
    if max_tokens == "0":
        max_tokens = None
    else:
        max_tokens = int(max_tokens)

    model = CONFIG.get("OpenAI", "model")
    max_range = 32768 if model == "gpt-4-32k" else 8192 if model == "gpt-4" else 4096 if model == "gpt-3.5-turbo" else 16384 

    layout = [
        [
            sg.Text("OpenAI API Key:"),
            sg.InputText(default_text=CONFIG.get("OpenAI", "api_key"), key="api_key"),
        ],
        [
            sg.Text("NotionAI Token:"),
            sg.InputText(default_text=CONFIG.get("NotionAI", "token_v2"), key="token"),
        ],
        [
            sg.Text("NotionAI SpaceID:"),
            sg.InputText(
                default_text=CONFIG.get("NotionAI", "space_id"), key="spaceid"
            ),
        ],
        [
            sg.Text("OpenAI Temperature (default 0.8):"),
            sg.Slider(
                range=(0, 1),
                default_value=float(CONFIG.get("OpenAI", "temperature")),
                resolution=0.1,
                orientation="h",
                size=(20, 15),
                key="temperature",
            ),
        ],
        [
            sg.Text(
                f"Max Tokens (default None. 32k max for GPT-4-32k, 8k max for GPT-4, 4k max for GPT-3.5-turbo and 16k for GPT_3.5-turbo-16k):"
            ),
            sg.Slider(
                range=(0, max_range),
                default_value=max_tokens,
                resolution=1,
                orientation="h",
                size=(20, 15),
                key="max_tokens",
            ),
        ],
        [sg.Button("Save"), sg.Button("Cancel")],
    ]

    swindow = sg.Window("Settings", layout)
    while True:
        event, values = swindow.read()

        if event == "Save":
            CONFIG.set("OpenAI", "api_key", str(values["api_key"]))
            CONFIG.set("OpenAI", "temperature", str(values["temperature"]))
            CONFIG.set("NotionAI", "token_v2", str(values["token"]))
            CONFIG.set("NotionAI", "space_id", str(values["spaceid"]))
            save_config(config_path, CONFIG)

            sg.popup("Settings saved!", keep_on_top=True)
            swindow.close()
            break
        elif event == "Cancel" or event == sg.WIN_CLOSED:
            swindow.close()
            break

    return

PROMPT = False
SKIP = False
DEBUG = True
TEST = True
window_location = (None, None)
include_urls = CONFIG.getboolean("GUI", "include_urls")
mem_on_off = CONFIG.getboolean("GUI", "mem_on_off")
TOPIC = CONFIG.get("GUI", "topic")
user = CONFIG.get("GUI", "user")
codemode = CONFIG.getboolean("GUI", "codemode")
costs = str(float(CONFIG.get("OpenAI", "costs")))
total_costs = str(float(CONFIG.get("OpenAI", "total_costs")))
total_tokens = str(float(CONFIG.get("OpenAI", "total_tokens")))
model = CONFIG.get("OpenAI", "model")
api_key = CONFIG.get("OpenAI", "api_key")
temperature = CONFIG.get("OpenAI", "temperature")
max_tokens = CONFIG.get("OpenAI", "max_tokens", fallback=None)
if max_tokens == "None":
    max_tokens = None
else:
    max_tokens = int(max_tokens)
openai.api_key = api_key

if if_first_time():
    webbrowser.open_new_tab(
        "https://313372600.notion.site/CopyCat-AI-Instructions-f94df67d0f3e47c89bd93810e38fb272"
    )
    settings_window()

def prompt_user(clip, img=False):
    global CONFIG
    global PROMPT
    global SKIP
    global mem_on_off
    global TOPIC
    global codemode
    global model
    global costs
    global total_costs
    global total_tokens
    global model
    global temperature
    global max_tokens
    global api_key
    global include_urls
    global window_location
    CONFIG = load_config(config_path)
    PROMPT = False
    SKIP = False
    DEBUG = True

    include_urls = CONFIG.getboolean("GUI", "include_urls")
    mem_on_off = CONFIG.getboolean("GUI", "mem_on_off")
    TOPIC = CONFIG.get("GUI", "topic")
    user = CONFIG.get("GUI", "user")
    codemode = CONFIG.getboolean("GUI", "codemode")
    costs = str(float(CONFIG.get("OpenAI", "costs")))
    total_costs = str(float(CONFIG.get("OpenAI", "total_costs")))
    total_tokens = str(float(CONFIG.get("OpenAI", "total_tokens")))
    model = CONFIG.get("OpenAI", "model")
    api_key = CONFIG.get("OpenAI", "api_key")
    temperature = CONFIG.get("OpenAI", "temperature")
    max_tokens = CONFIG.get("OpenAI", "max_tokens", fallback=None)
    if max_tokens == "0":
        max_tokens = None
    else:
        max_tokens = int(max_tokens)
    openai.api_key = api_key
    if is_api_key_empty(api_key) and is_notion_token_empty(
        CONFIG.get("NotionAI", "token_v2"), CONFIG.get("NotionAI", "space_id")
    ):
        settings_window()
        api_key = CONFIG.get("OpenAI", "api_key")
        openai.api_key = api_key

    openai_memory = OpenAIMemory(memory_path, config_path)
    input_text = ""
    PROMPT = True
    if DEBUG:
        print("Loading Prompt")

    try:
        layout = [
            [
                sg.Menu(
                    [["Settings", ["Preferences", "Prompt Manager", "About", "Help"]]],
                    key="-MENU-",
                )
            ],
            [
                sg.Text(
                    "What are your orders? (Press enter to submit or escape to cancel)"
                )
            ],
            [
                sg.Input(
                    key="input", tooltip="Press enter key to submit", do_not_clear=False
                ),
                sg.Button(
                    "OK",
                    tooltip="Press enter key to submit",
                    visible=False,
                    bind_return_key=True,
                    key="-RETURN-",
                ),
                sg.Button(
                    "Cancel",
                    tooltip="Press escape key to cancel",
                    visible=False,
                    bind_return_key=True,
                    key="-ESCAPE-",
                ),
                sg.Text("Select Model:", tooltip="Select Model"),
                sg.Combo(
                    ["gpt-4-32k", "gpt-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo", "NotionAI"],
                    default_value=model,
                    readonly=True,
                    key="-MODEL-",
                    enable_events=True,
                ),
            ],
            [
                sg.Column(
                    [
                        [
                            sg.Checkbox(
                                "Include Url Pages",
                                key="-URLS-",
                                default=include_urls,
                                tooltip="Include Url Pages",
                            ),
                        ],
                        [
                            sg.Checkbox(
                                "Memory On/Off",
                                default=mem_on_off,
                                key="memory_on_off",
                                tooltip="Handy if you're editing docs",
                            ),
                        ],
                        [
                            sg.Checkbox(
                                "Coding Mode",
                                default=codemode,
                                key="code",
                                tooltip="Coding Mode",
                            ),
                        ],
                        [
                            sg.Checkbox(
                                "Preview",
                                default=False,
                                key="-PREVIEW-",
                                tooltip="Preview Mode",
                                enable_events=True,
                            ),
                        ],
                    ],
                    pad=(0, 0),
                ),
                sg.VerticalSeparator(),
                sg.Frame(
                    title="API Info",
                    layout=[
                        [
                            sg.Text("Last Request Costs: "),
                            sg.Text("0", key="lrcosts", size=(10, 1)),
                        ],
                        [
                            sg.Text("Total Requests Costs: "),
                            sg.Text("0", key="total", size=(10, 1)),
                        ],
                        [
                            sg.Text("Temperature: "),
                            sg.Text(temperature, key="temperature", size=(10, 1)),
                        ],
                        [
                            sg.Text("Last Request Tokens: "),
                            sg.Text(max_tokens, key="total_tokens", size=(10, 1)),
                        ],
                    ],
                    relief=sg.RELIEF_SUNKEN,
                    border_width=2,
                ),
                sg.VerticalSeparator(),
                sg.Frame(
                    title="",
                    layout=[
                        [sg.Image(logo_path, size=(100, 100))],
                    ],
                    relief=sg.RELIEF_SUNKEN,
                    border_width=2,
                ),
            ],
            [
                sg.Button(
                    "Clear Memory",
                    key="Clear Memory",
                    tooltip="Clears the memory and starts fresh",
                    enable_events=True,
                ),
            ],
            [
                sg.Text(
                    "New System Prompt: ",
                    tooltip="Topic to use for the prompt. Default is general",
                ),
                sg.Input(
                    key="topic",
                    tooltip="Press enter key to submit",
                    do_not_clear=False,
                    default_text=TOPIC,
                    size=(54, 1),
                ),
            ],
            [
                sg.Text(
                    "Existing System Prompt: ",
                    tooltip="Topic to use for the prompt. Default is general",
                ),
                sg.Combo(
                    openai_memory.get_memory_keys(),
                    default_value=TOPIC,
                    key="-COMBO-",
                    tooltip="Press enter key to submit",
                    readonly=True,
                    enable_events=True,
                    size=(28, 1),
                ),
                sg.Button(
                    "Delete System Prompt",
                    key="Delete Topic",
                    tooltip="Deletes the topic",
                    enable_events=True,
                ),
            ],
            [sg.Multiline(size=(75, 10), key="-PREVIEW-ML-", visible=False)],
        ]

        window = sg.Window(
            "CopyCat AI",
            layout,
            keep_on_top=True,
            auto_close=True,
            auto_close_duration=120,
            location=window_location,
            finalize=True,
        )

        window.bind("<Escape>", "-ESCAPE-")
        window.bind("<Return>", "OK")

        window.bring_to_front()
        if DEBUG:
            print("Prompt Loaded", window)

        window["lrcosts"].update(str(float(costs[:6])))
        window["total"].update(str(float(total_costs[:6])))
        window["total_tokens"].update(str(total_tokens))

        while True:
            if DEBUG:
                print("Got here!")
            event, values = window.read()
            TOPIC = values["-COMBO-"]
            window_location = window.current_location()
            if event == sg.TIMEOUT_KEY:
                window.close()
                return
            if event == sg.WIN_CLOSED or event == None:  # Check for event == WIN_CLOSED
                return

            elif event == "Prompt Manager":
                prompt_manager = PromptManager(memory_path)
                prompt_manager.prompt_manager()
                openai_memory.load_memory()
                window["-COMBO-"].update(openai_memory.get_memory_keys())
                window.refresh()

            elif event == "About":
                sg.popup(
                    "Title: CopyCat AI\n\nCopyright: Nobility AI, 2023\n\nWebsite: https://nobilityai.com\n\nEmail: copycat@nobilityai.com\n\n",
                    keep_on_top=True,
                )

            elif event == "Preferences":
                settings_window()
                temperature = CONFIG.get("OpenAI", "temperature")
                max_tokens = CONFIG.get("OpenAI", "max_tokens")
                if max_tokens == "0":
                    max_tokens = None
                else:
                    max_tokens = int(max_tokens)
                openai.api_key = CONFIG.get("OpenAI", "api_key")
                window["temperature"].update(temperature)
                window["total_tokens"].update(max_tokens)
                window.refresh()
                # api_key = CONFIG.get("OpenAI", "api_key")
                # openai.api_key = api_key
                # temperature = CONFIG.get("OpenAI", "temperature")
                # max_tokens = CONFIG.get("OpenAI", "max_tokens")

            elif event == "Help":
                webbrowser.open(
                    "https://313372600.notion.site/313372600/CopyCat-AI-Instructions-f94df67d0f3e47c89bd93810e38fb272"
                )

            if event == "Clear Memory":
                if DEBUG:
                    print("Clearing memory")

                openai_memory.clear_memory(TOPIC)
                window.refresh()
                # memory(user=user, topic=TOPIC, reset=True)
                TOPIC = values["-COMBO-"]
                if DEBUG:
                    print(TOPIC)
                display_notification(
                    "Topic Memory Cleared",
                    TOPIC + " memory has been cleared",
                    img_success,
                    5000,
                    use_fade_in=False,
                    location=(0, 0),
                )

            if event == "Delete Topic":
                if DEBUG:
                    print("DELETING TOPIC", values["-COMBO-"])
                if (
                    values["topic"] != ""
                    and values["topic"]
                    != "You are an AI assistant helping with knowledge and code."
                ):
                    TOPIC = values[
                        "topic"
                    ]  # use values['topic'] instead of values['-COMBO-']
                    if DEBUG:
                        print("Deleting topic")
                    # memory(user=user, topic=TOPIC, delete=True)
                    openai_memory.clear_memory(TOPIC)
                    CONFIG["GUI"][
                        "topic"
                    ] = "You are an AI assistant helping with knowledge and code."
                    window["-COMBO-"].update(
                        values=openai_memory.get_memory_keys(),
                        value="You are an AI assistant helping with knowledge and code.",
                    )
                    TOPIC = values["topic"]
                    save_config(config_path, CONFIG)
                    # event, values = window.read()
                    window.refresh()
                    display_notification(
                        "Topic Deleted",
                        TOPIC + " has been deleted",
                        img_success,
                        5000,
                        use_fade_in=False,
                        location=(0, 0),
                    )

                else:
                    if DEBUG:
                        print("Cannot delete default topic")
                    window["-COMBO-"].update(
                        values=openai_memory.get_memory_keys(),
                        value="You are an AI assistant helping with knowledge and code.",
                    )
                    TOPIC = values["topic"]
                    # memory(user=user, topic="general", reset=True)
                    openai_memory.clear_memory(
                        "You are an AI assistant helping with knowledge and code."
                    )
                    window.refresh()
                    display_notification(
                        "Default Topic Reset",
                        "Can't Delete Default Topic",
                        img_success,
                        5000,
                        use_fade_in=False,
                        location=(0, 0),
                    )

            if event == "-COMBO-":
                if DEBUG:
                    print(values["-COMBO-"])
                if DEBUG:
                    print("EVENT IS: ", event)
                TOPIC = values["-COMBO-"]
                window["topic"].update(values["-COMBO-"])
                CONFIG["GUI"]["topic"] = values["-COMBO-"]
                save_config(config_path, CONFIG)

            if event == "-MODEL-":
                model = values["-MODEL-"]
                CONFIG["OpenAI"]["model"] = model
                save_config(config_path, CONFIG)

            if event == "-PREVIEW-":
                if values["-PREVIEW-"]:
                    window["-PREVIEW-ML-"].update(visible=values["-PREVIEW-"])
                else:
                    print(values["-PREVIEW-"])

                    window["-PREVIEW-ML-"].update(visible=values["-PREVIEW-"])

            elif event == "OK" or event == "-RETURN-":
                if event == "-ESCAPE-":
                    window.refresh()
                    break
                model = values["-MODEL-"]
                CONFIG["OpenAI"]["model"] = model

                # window["-MODEL-"].update(values["-MODEL-"])
                include_urls = values["-URLS-"]
                mem_on_off = values["memory_on_off"]
                CONFIG["GUI"]["mem_on_off"] = str(mem_on_off)
                CONFIG["GUI"]["include_urls"] = str(include_urls)
                codemode = values["code"]
                CONFIG["GUI"]["codemode"] = str(codemode)
                save_config(config_path, CONFIG)
                input_text = values["input"]
                if len(input_text) > 0:
                    # print("Input text is: ", input_text)
                    if values["topic"] != "":
                        TOPIC = values["topic"]
                if not window["-PREVIEW-ML-"].visible:
                    window.close()
                submit(
                    input_text,
                    clip,
                    img,
                    mem_on_off,
                    TOPIC,
                    codemode,
                    memory_path,
                    config_path,
                    window,
                )

                break

            elif event == "Cancel" or event == "-ESCAPE-" or None:
                if DEBUG:
                    print("Closing window")
                break

        try:
            if window["-PREVIEW-ML-"].visible:
                while True:
                    event, values = window.read()
                    if (
                        event == sg.WIN_CLOSED
                        or event == "-ESCAPE-"
                        or event == "Cancel"
                        or event == "-RETURN-"
                    ):
                        break
        except Exception as e:
            window.close()
            display_notification(
                "Error!",
                "An exception occurred: " + str(e),
                img_error,
                5000,
                use_fade_in=False,
                location=(0, 0),
            )

        window.close()
        return

    except Exception as e:
        if (
            e is not None
            and "NoneType" not in str(type(e))
            and "subscriptable" not in str(e)
        ):
            print(e)
            print(traceback.format_exc())
            display_notification(
                "Error!",
                "An exception occurred: " + str(e),
                img_error,
                5000,
                use_fade_in=False,
                location=(0, 0),
            )
        else:
            window.close()
            return

def code_mode(reply):  # This function is used to format the reply in code mode
    if "```" not in reply:  # This if statement is used to format the reply in code mode
        return reply

    reply = reply.split("\n")
    in_code_block = False
    new_reply = []
    for line in reply:  # This loop is used to format the reply in code mode
        if (
            line.startswith("```")
            or line.startswith("<code>")
            or line.startswith("</code>")
        ):  # This if statement is used to format the reply in code mode
            in_code_block = (
                not in_code_block
            )  # This if statement is used to format the reply in code mode
            continue
        if in_code_block:  # This if statement is used to format the reply in code mode
            new_reply.append(
                line
            )  # This if statement is used to format the reply in code mode
    return "\n".join(new_reply)

def submit(
    input_text, clip, img, mem_on_off, topic, codemode, memory_path, config_path, window
):
    if not window["-PREVIEW-ML-"].visible:
        window.close()
    mem = ""
    reply = ""
    new = False
    if img and is_tesseract_installed():
        if os.path.exists("/tmp/copycat.jpg"):
            os.remove("/tmp/copycat.jpg")
        clip.save("/tmp/copycat.jpg")
        clip = basic_text_extractor("/tmp/copycat.jpg")
    if isLink(clip.strip()) and include_urls:
        url = extracturl(clip.strip())
        if not isTwitterLink(url):
            try:
                link_summary = get_page_from_text(user, clip.strip())
                clip = clip + "\n" + "URL: " + url + "\n" + link_summary
            except Exception as e:
                print(e)
                pass
        else:
            try:
                link_summary = search_twitter(url) + get_page_from_text(
                    user, clip.strip()
                )
                clip = clip + "\n" + "URL: " + url + "\n" + link_summary
            except Exception as e:
                print(e)
                pass
    # helpmewrite(cookies, headers, user, prompt, content, pagetitle):

    if DEBUG:
        print("Mem", mem_on_off)

    if DEBUG:
        print("MODEL", model)
    cost_manager = CostManager(config_path, memory_path)

    try:
        print("topic is: ", topic)
        print("input_text is: ", input_text)
        print("clip is: ", clip)
        print("model is: ", model)
        print("mem_on_off is: ", mem_on_off)
        print("max_tokens is: ", max_tokens)
        print("temperature is: ", temperature)
        print("include_urls is: ", include_urls)
        print("codemode is: ", codemode)
        print("memory_path is: ", memory_path)
        print("config_path is: ", config_path)

        reply = cost_manager.process_request(
            topic,
            f"{input_text}\n\n{clip}",
            model,
            use_memory=mem_on_off,
            tokens=max_tokens,
            temperature=float(temperature),
        )
    except openai.error.AuthenticationError as error:
        print(error)
        display_notification(
            "Go To Settings-Preferences->OpenAI API Key",
            "An exception occurred: " + str(error),
            img_error,
            5000,
            use_fade_in=False,
            location=(0, 0),
        )
        CONFIG = load_config(config_path)
        return

    except APIError as error:
        print(error)
        display_notification(
            "Error!",
            "An exception occurred: " + str(error),
            img_error,
            5000,
            use_fade_in=False,
            location=(0, 0),
        )
        return

    if codemode:
        reply["response"] = code_mode(reply["response"])
        if DEBUG:
            print("Code mode", reply["response"])
    if DEBUG:
        print("Reply", reply["response"])
    if reply["response"].strip():  # If the reply is not empty
        if window["-PREVIEW-ML-"].visible:
            window["-PREVIEW-ML-"].update(reply["response"])
        pyperclip.copy(reply["response"])  # Copy the reply to the clipboard

        if not pyperclip.paste():  # If the clipboard is empty
            pyperclip.copy(reply["response"])  #
        display_notification(  # Display a notification
            "Copied to Clipboard!",  # Title
            "Your AI Request Has Completed Successfully!",  # Message
            img_success,  # Image
            5000,  # Duration
            use_fade_in=False,  # Fade in
            location=(0, 0),  # Location
            keep_on_top=True,  # Keep on top
        )

        return

    else:  # If the reply is empty
        display_notification(  # Display a notification
            "Error!",  # Title
            "Your AI Request Has Failed!",  # Message
            img_error,  # Image
            5000,  # Duration
            use_fade_in=False,  # Fade in
            location=(0, 0),  # Location
            keep_on_top=True,  # Keep on top
        )
        if not pyperclip.paste():  # If the clipboard is empty
            pyperclip.copy("")  # Copy the input text to the clipboard
        return  # Return the input text


def main(PROMPT, SKIP, prompt_user):
    while True:
        try:
            clip = pyperclip.waitForNewPaste()
            if clip == "" or clip == None and not PROMPT and not SKIP:  # IMAGE CHECK
                image = ImageGrab.grabclipboard()
                if isinstance(image, Image.Image):  # If the clipboard contains an image
                    clip = image
                    prompt_user(clip, img=True)
                    PROMPT = False
                    SKIP = False
            elif clip != "" and clip != None and not PROMPT and not SKIP:
                prompt_user(clip.strip())
                PROMPT = False
                SKIP = False
            elif SKIP:
                pyperclip.copy(clip.strip())
                SKIP = False
                PROMPT = False
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    PROMPT = False
    SKIP = False
    try:
        if DEBUG:
            print("****Main was called*****")
        main(PROMPT, SKIP, prompt_user)
    except Exception as e:
        print(e)
        display_notification(
            "Error!",
            "An exception occurred: " + str(e),
            img_error,
            5000,
            use_fade_in=False,
            location=(0, 0),
        )
