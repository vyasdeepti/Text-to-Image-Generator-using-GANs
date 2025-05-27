# Text-to-Image Generator using GANs🚀

Text-to-Image-Generator is a Jupyter Notebook-based project that enables users to generate images from textual descriptions using machine learning techniques. The repository showcases how natural language processing (NLP) and generative models can be combined to convert text prompts into corresponding images. It is ideal for learning, experimenting, and demonstrating the capabilities of text-to-image synthesis, making it useful for both educational purposes and as a foundation for further research or application development in the field of AI-powered image generation. 
This project demonstrates text-to-image synthesis in Python using PyTorch and Jupyter Notebook.

**AIM: Generate realistic images from natural language descriptions using Generative Adversarial Networks (GANs).**

## GANs (Generative Adversarial Networks) 🔊🗣
GANs in text-to-image processing are deep learning models used to generate realistic images based on textual descriptions. 
[Read more about GANs](https://insights.daffodilsw.com/blog/a-complete-guide-to-gans)

## How GANs Work in Text-to-Image
- 🧑‍💻 Generator: Takes a text description (like "a yellow bird with black wings") and tries to generate an image that matches the description.
- 🧑‍💻 Discriminator: Judges whether an image matches the given text and whether it is real (from the dataset) or fake (generated).

Both networks are trained together: the generator improves at creating convincing images, while the discriminator gets better at detecting mismatches.

## Typical Workflow:
- 🔶 Text Embedding: The input text is converted into a numerical representation (embedding), often using models like RNNs, LSTMs, or Transformers.
- 🔶 Conditioning: The image generator network is conditioned on the text embedding, so it can produce images relevant to the text.
- 🔶 Adversarial Training: The discriminator receives both the generated image and the embedding to decide if the image matches the text.

## Popular GAN Architectures for Text-to-Image
- 🟢 StackGAN: Generates images in multiple stages, from low to high resolution, based on the text.
- 🟢 AttnGAN: Uses attention mechanisms to focus on relevant words in the text when generating different parts of the image.
- 🟢 DM-GAN: Dynamically refines the image generation process based on feedback from the text.

![image](https://github.com/user-attachments/assets/1552418c-eb98-425f-a9d2-927ca92c1092)

Source: [Image Generation from Text Using StackGAN with Improved Conditional Consistency Regularization](https://www.mdpi.com/1424-8220/23/1/249)


## Table of Contents

- 🚀 [Project Overview](#project-overview)
- 🖼️ [How It Works](#how-it-works)
- 🤖 [Installation](#installation)
- 🧑‍💻 [Configuration & Parameters](#configuration--parameters)
- 🛠️ [Training Details](#training-details)
- 📊 [Results & Examples](#results--examples)
- 🧑‍💻 [Troubleshooting](#troubleshooting)
- 📚 [References](#references)
  
## Project Overview

This repository implements a text-to-image generator using deep learning. By leveraging GANs, the model learns to create images that correspond to textual input, opening up possibilities for creative AI, automatic illustration, and assistive technology.

### Key Features

- **Text Embedding:** Converts text descriptions into vector representations.
- **GAN-based Synthesis:** Uses GANs to generate images from text embeddings.
- **Jupyter Notebook Workflow:** Easy-to-follow, interactive, and visual.
- **Customizable:** Easily extendable to new datasets or architectures.


## How It Works

1. **Text Processing:** The model tokenizes and embeds the input text using NLP techniques.
2. **Generator Network:** The generator receives the embedded text and produces an image.
3. **Discriminator Network:** The discriminator evaluates whether images are real or generated and whether they match the text.
4. **Adversarial Training:** The networks are trained in tandem to improve the realism of generated images and the accuracy of text-image pairing.

## Installation

### Prerequisites

- Python 3.7+
- pip or conda
- Jupyter Notebook or JupyterLab

### Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/vyasdeepti/Text-to-Image-Generator.git
    cd Text-to-Image-Generator
    ```

2. **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib nltk
    ```

## Usage

1. **Open the notebook:**
    - Launch Jupyter and open `gans-text-to-image code.ipynb`.

2. **Configure parameters:**
    - Set paths, epochs, and hyperparameters in the configuration cells.

3. **Run cells sequentially:**
    - Preprocessing → Model definition → Training → Generation → Visualization.

4. **Generate images:**
    - Input your own text prompts in the appropriate cell to generate new images.

---

Here’s an explanation of the code:

```python
import torch
print(torch.__version__)
```

**Explanation:**

- import torch  
  This imports the PyTorch library, which is a popular open-source machine learning framework used for applications such as deep learning, computer vision, and natural language processing.

- print(torch.__version__)  
  This prints the currently installed version of PyTorch to the output.  
  The __version__ attribute is a string that tells you exactly which version of the torch (PyTorch) package is being used in your environment.

**Usage in Notebooks:**  
This pattern is common at the top of machine learning or deep learning notebooks to verify that the correct version of PyTorch is installed and being used. This can help with reproducibility and debugging, especially if certain features or functions require a specific version.


Certainly! Here’s what the command does:

```bash
pip install --upgrade torch
```

**Explanation:**

- pip: This is the Python package installer. It is used to install and manage software packages written in Python.
- install: This tells pip that you want to install a package.
- --upgrade: This flag tells pip to upgrade the package to the latest available version if it is already installed. If the package is not installed, it will just install the newest version.
- torch: This is the name of the package you want to install or upgrade, which refers to **PyTorch**, a popular open-source deep learning library.

**In summary:**  
This command checks if PyTorch (torch) is installed. If it is, it upgrades it to the latest version. If it isn’t installed, it installs the latest version.


---

### 1. `!pip install diffusers --upgrade`

- `!`: The exclamation mark is used in Jupyter notebooks to run shell commands (as if you typed them in a terminal).
- `pip install`: Installs a Python package.
- `diffusers`: This is a library from Hugging Face for running and creating diffusion-based models for image generation, like Stable Diffusion.
- `--upgrade`: Makes sure you get the latest version of the package, even if you already have it installed.

**Summary:**  
Installs or upgrades the Hugging Face Diffusers library, which is essential for working with modern text-to-image models.

---

### 2. 
```python
!pip install invisible_watermark transformers accelerate safetensors

```

- `invisible_watermark`: A library to add invisible watermarks to generated images (sometimes used for watermarking AI-generated images).
- `transformers`: Another Hugging Face library, widely used for natural language processing and also supports text encoders needed for text-to-image models.
- `accelerate`: A Hugging Face utility to make running models on different hardware (CPU, single GPU, multiple GPUs, TPUs) easier and faster.
- `safetensors`: A library for reading and writing tensors (model weights) efficiently and securely, often used as a safer/faster alternative to PyTorch’s default format.

Installs several supporting libraries:
- `invisible_watermark` for watermarking images,
- `transformers` for text processing and model loading,
- `accelerate` for optimizing model speed and hardware use,
- `safetensors` for efficient model weight storage.


**When are these used?**  
These commands are typically run at the start of a notebook to ensure all necessary libraries are available for text-to-image generation and related tasks, especially when using Hugging Face models.

---

### 3. `!pip install transformers==4.43.2`

- The exclamation mark (`!`) is used in Jupyter notebooks to run a shell command.
- `pip install` is the command for installing Python packages.
- `transformers==4.43.2` specifies that you want to install version 4.43.2 of the Hugging Face Transformers library.
    - Transformers is a popular library for natural language processing (NLP) and also supports text encoders for text-to-image models.
    - Specifying a version (`==4.43.2`) ensures that you get a particular, known version of the package, which can help with reproducibility and compatibility.

---

### 4. `import torch`

- This imports the PyTorch library.
- PyTorch is a widely-used library for deep learning and tensor computations.
- It is commonly used for building, training, and running neural networks, including those used in text-to-image generation.

---

### 5. `from diffusers import DiffusionPipeline`

- This imports the `DiffusionPipeline` class from the Hugging Face `diffusers` library.
- The `DiffusionPipeline` is a high-level interface for running diffusion models, such as Stable Diffusion, for generating images from text or other modalities.
- With this class, you can quickly load and use pre-trained diffusion models for generating images, often with just a few lines of code.

These lines ensure you have the correct version of the Transformers library installed, import PyTorch for deep learning operations, and load the class needed to use diffusion models for text-to-image generation.


```bash
pip install -U ipywidgets
```

**Explanation:**

- pip install: This is the command to install a Python package using pip, the Python package installer.
- -U: This flag stands for "upgrade." It tells pip to upgrade the package to the latest version if it’s already installed.
- ipywidgets: This is the name of the package you want to install or upgrade. `ipywidgets` is a library for creating interactive HTML widgets for Jupyter notebooks and JupyterLab.

**What does ipywidgets do?**  
`ipywidgets` allows you to add interactive controls (like sliders, dropdowns, buttons, etc.) to your Jupyter notebooks. This is very useful for data visualization, interactive model demos, and making your notebooks more user-friendly and dynamic.

**Typical use-case:**  
You’ll run this command when you want to add or update interactive widgets in your Jupyter notebooks, ensuring you have the latest features and bug fixes.

---

### 6.  
```python
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
```
**What it does:**
- This line loads a pre-trained diffusion model pipeline from Hugging Face’s diffusers library, specifically the Stable Diffusion XL base model.

**Parameter breakdown:**
- `DiffusionPipeline.from_pretrained(...)`:  
  Loads a pipeline with all model weights and configuration from the Hugging Face Model Hub.
- `"stabilityai/stable-diffusion-xl-base-1.0"`:  
  This is the name of the model repository on Hugging Face, pointing to a powerful text-to-image model.
- `torch_dtype=torch.float16`:  
  Loads the model weights in 16-bit floating point format. This reduces GPU memory usage and can speed up inference, especially on modern GPUs.
- `use_safetensors=True`:  
  Loads model weights using the safer and faster `safetensors` format instead of the traditional PyTorch `.bin` format.
- `variant="fp16"`:  
  Specifies that the FP16 (16-bit floating point) variant of the model should be loaded, ensuring compatibility with the specified `torch_dtype`.

**The result:**  
- `pipe` is now a ready-to-use text-to-image generation pipeline using Stable Diffusion XL.

---

### 7.  
```python
pipe.to("cuda")
```
**What it does:**
- Moves the entire pipeline (including all model weights) to the GPU for faster processing.
- `"cuda"` is PyTorch’s keyword for GPU computation (usually NVIDIA GPUs). If you only use CPU, you’d use `"cpu"` instead.

**Why do this?**
- Running large models like Stable Diffusion on a GPU is much faster than on a CPU, especially for image generation.

These lines download and prepare the Stable Diffusion XL model for efficient image generation, using GPU acceleration and memory-saving optimizations (float16, safetensors).

---

### 8.  
```python
prompt = "This image features a close-up of an eye, showcasing intricate details such as the iris and eyelashes. The artwork emphasizes the use of eye shadow and mascara, highlighting the beauty and artistry involved in eye makeup. Overall, it captures the organ's aesthetic appeal through a creative lens."
```
- This line creates a string variable named `prompt`.
- The string describes—in natural language—the kind of image you want the AI model to generate.
- The description is detailed, focusing on:
  - A close-up of an eye,
  - Details like the iris, eyelashes, eye shadow, and mascara,
  - Artistic and aesthetic aspects of eye makeup.

**Purpose:**  
This text prompt is used to guide the text-to-image model (such as Stable Diffusion, via `DiffusionPipeline`) to produce an image that matches this description.

---

### 9.  
```python
images = pipe(prompt=prompt).images[0]
```
- `pipe(prompt=prompt)` calls the diffusion pipeline (previously loaded, e.g., with Stable Diffusion) to generate an image based on the given prompt.
- `.images` returns a list of images generated by the pipeline (often just one image, but the model can generate multiple images at once).
- `[0]` selects the first image from this list.

**Purpose:**  
This line actually runs the AI model to generate an image that best matches the description in `prompt` and stores the resulting image object in the variable `images`.

- The `prompt` describes exactly what kind of image you want.
- `pipe(prompt=prompt).images[0]` uses a pre-trained AI model to generate that image and stores it in `images`.


## Configuration & Parameters

- **Text Embedding:** (e.g., RNN, LSTM, or pretrained embeddings like GloVe)
- **GAN Architecture:** (e.g., DCGAN, StackGAN, or custom)
- **Hyperparameters:** Learning rate, batch size, number of epochs, etc.—editable in the notebook.
- **Dataset:** Custom or public datasets (e.g., CUB-200, Oxford Flowers).


## Training Details

- **Hardware:** Training on GPU recommended for speed.
- **Epochs:** Training time depends on dataset size and model complexity.
- **Checkpointing:** Intermediate models can be saved and loaded.


## Results & Examples

🔊🗣 Prompt: "This image features a close-up of an eye, showcasing intricate details such as the iris and eyelashes. The artwork emphasizes the use of eye shadow and mascara, highlighting the beauty and artistry involved in eye makeup. Overall, it captures the organ's aesthetic appeal through a creative lens"

<img src="https://github.com/user-attachments/assets/815c66bc-a4ae-4f55-be19-8d9a3b9e52d5" alt="image" width="300"/>  



🔊🗣 Prompt: "The image features a captivating collage of stars and galaxies, showcasing the vastness and beauty of outer space. It includes various astronomical objects such as nebulae, constellations, and spiral galaxies, highlighting elements of the Milky Way. This striking composition emphasizes the wonders of astronomy and the universe we inhabit."

<img src="https://github.com/user-attachments/assets/559f9408-8c6a-44ee-95b9-7bf2b5895e94" alt="output" width="300"/>  

## Troubleshooting

- **CUDA Errors:** Ensure PyTorch is installed with CUDA support if using GPU.
- **Data Format Issues:** Check that your data matches the expected format in the notebook.
- **Memory Issues:** Reduce batch size or image resolution.


## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Generative Adversarial Networks: An Overview](https://arxiv.org/pdf/1710.07035)
- [Training Generative Adversarial Networks with Limited Data](https://proceedings.neurips.cc/paper/2020/file/8d30aa96e72440759f74bd2306c1fa3d-Paper.pdf)


