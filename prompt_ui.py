import PySimpleGUI as sg
import json


# Class to manage prompts
class PromptManager:
    def __init__(self, memory_file):  # Initialize with a file to store prompts
        self.memory_file = memory_file
        self.load_prompts()

    # Load prompts from memory_file
    def load_prompts(self):
        with open(self.memory_file, "r") as f:
            self.prompts = json.load(f)

    # Save prompts to memory_file
    def save_prompts(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.prompts, f, indent=4)

    # Graphical interface to manage prompts
    def prompt_manager(self, main_window=None):
        if main_window:
            main_window.Hide()
        # Create layout for the window
        layout = [
            [sg.Text("Select a prompt to manage:", font=("Helvetica", 14))],
            [
                sg.Column(
                    [
                        [
                            sg.Listbox(
                                values=list(self.prompts.keys()),
                                size=(60, 20),
                                key="-PROMPT-LIST-",
                            )
                        ]
                    ],
                    element_justification="center",
                )
            ],
            [
                sg.Button("Add"),
                sg.Button("Edit"),
                sg.Button("Delete"),
                sg.Button("Exit"),
            ],
        ]

        # Create the window
        window = sg.Window("Prompt Manager", layout, size=(400, 320), finalize=True)
        window.bring_to_front()

        # Event loop
        while True:
            event, values = window.read()

            # Exit the loop if user closes the window or clicks "Exit" button
            if event in (None, "Exit"):
                break

            # Add a new prompt
            if event == "Add":
                try:
                    new_prompt_key = sg.popup_get_text(
                        "Enter a new system prompt:",
                        keep_on_top=True,
                        size=(50, 10),
                    )
                    if new_prompt_key and new_prompt_key not in self.prompts:
                        new_prompt_text = (
                            ""  # Set the default prompt text to an empty string
                        )
                        self.prompts[new_prompt_key] = new_prompt_text
                        self.save_prompts()
                        sg.popup("Prompt added successfully!", keep_on_top=True)
                        window.Element("-PROMPT-LIST-").Update(
                            values=list(self.prompts.keys())
                        )
                except:
                    sg.popup(
                        "Error: Something went wrong with adding the prompt. Please try again.",
                        keep_on_top=True,
                    )

            # Edit an existing prompt
            if event == "Edit":
                try:
                    prompt_key = values["-PROMPT-LIST-"][0]
                    prompt_text = self.prompts[prompt_key]
                    new_prompt_text = sg.popup_get_text(
                        "Edit Prompt",
                        default_text=prompt_key,  # Set default_text to the selected prompt_key
                        size=(
                            50,
                            10,
                        ),  # Set the input box size to 50 columns and 2 rows
                        keep_on_top=True,
                    )
                    if new_prompt_text:
                        self.prompts[prompt_key] = new_prompt_text
                        self.save_prompts()
                        sg.popup("Prompt updated successfully!", keep_on_top=True)
                        window.Element(
                            "-PROMPT-LIST-"
                        ).Update(  # Refresh the list of prompts
                            values=list(self.prompts.keys())
                        )
                except:
                    sg.popup(
                        "Error: Something went wrong with editing the prompt. Please try again.",
                        keep_on_top=True,
                    )

            # Delete a prompt
            if event == "Delete":
                try:
                    prompt_key = values["-PROMPT-LIST-"][0]
                    confirm = sg.popup_yes_no(
                        f'Are you sure you want to delete prompt "{prompt_key}"?',
                        keep_on_top=True,
                    )
                    if confirm == "Yes":
                        del self.prompts[prompt_key]
                        self.save_prompts()
                        sg.popup("Prompt deleted successfully!", keep_on_top=True)
                        window.Element(
                            "-PROMPT-LIST-"
                        ).Update(  # Refresh the list of prompts
                            values=list(self.prompts.keys())
                        )
                except:
                    sg.popup(
                        "Error: Something went wrong with deleting the prompt. Please try again.",
                        keep_on_top=True,
                    )

        # Close the window
        window.close()
        if main_window:
            main_window.UnHide()
            main_window.bring_to_front()
