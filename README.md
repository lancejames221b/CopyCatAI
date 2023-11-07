# CopyCat AI

CopyCat AI is a Python application that uses OpenAI's models to generate responses based on clipboard contents. It provides a graphical user interface (GUI) for easy interaction and can process text and images from the clipboard.

## Prerequisites

```sh
python3 -m venv ./
source bin/activate
```

## Installation

1. Clone the repository or download the source code.
2. Install the required Python packages using `pip`:
   ```sh
   pip install -r requirements.txt
   ```
3. Obtain your OpenAI API key from [https://openai.com/api/](https://openai.com/api/) and keep it ready for use.

## Usage

To run the application, navigate to the directory containing the downloaded source code and execute:

```sh
nohup python3 copycat.py &
```

Upon running the application, it will monitor the clipboard for changes. When new text or an image is detected, it will prompt the user to generate a response using the GPT-3 model.

### Configuration

The first time you run the application, it will prompt you to enter your OpenAI API key and other configuration settings. These settings are necessary for the application to function correctly.

### Interacting with the GUI

The GUI provides various options to customize the behavior of the AI. You can:

- Set preferences such as the OpenAI model, temperature, and max tokens.
- Manage prompts and memory to tailor responses.
- Adjust settings for including URLs and coding mode.
- Clear memory or reset cost calculations.

### Using the AI

After setting up your preferences and configuration, simply copy text or an image to the clipboard. The application will detect it and provide an option to generate a response. If you're satisfied with the response preview, you can copy it to the clipboard or use it as needed.

## Support and Feedback

If you encounter any issues or have feedback for improvements, please contact the support team at copycat@nobilityai.com or visit the official website at [https://nobilityai.com](https://nobilityai.com).

## License

CopyCat AI is copyrighted by Nobility AI, 2023. Please review the license agreement for using the software before proceeding.

---

Please note that this README provides a general guide for using CopyCat AI. Ensure that you follow any specific instructions or requirements within the source code or documentation provided with the application.
