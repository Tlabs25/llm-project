Building an LLM From Scratch

Project Goal



This project is a learning exercise focused on building a small language model from the ground up using Python and PyTorch.



Rather than starting with a pre-trained model, the goal is to understand the major components involved in a language model by building them step by step.



The project will eventually follow this pipeline:



Training Text

&#x20;     ↓

Tokenizer

&#x20;     ↓

Token IDs

&#x20;     ↓

PyTorch Tensors

&#x20;     ↓

Embeddings

&#x20;     ↓

Positional Information

&#x20;     ↓

Transformer

&#x20;     ↓

Next-Token Prediction

&#x20;     ↓

Loss Calculation

&#x20;     ↓

Backpropagation

&#x20;     ↓

Training

&#x20;     ↓

Generated Text



The final goal is to understand how these components work together and eventually experiment with larger models and cybersecurity-focused language-model applications.



Hardware and Environment



Current development environment:



Operating System: Windows

CPU: AMD Ryzen 5 9600X

GPU: NVIDIA GeForce RTX 5070

GPU VRAM: Approximately 12 GB

Python virtual environment: .venv

Machine learning framework: PyTorch

CUDA version used by PyTorch: 12.8



GPU verification produced:



PyTorch version: 2.11.0+cu128

CUDA available: True

CUDA version: 12.8

GPU: NVIDIA GeForce RTX 5070



This confirms that PyTorch can use the NVIDIA GPU for future model training.



Project Structure



The project is organized as follows:



llm-project/

│

├── .venv/              # Python virtual environment

│

├── data/

│   └── training.txt     # Training text

│

├── src/

│   ├── tokenizer.py

│   ├── prepare\_data.py

│   └── embeddings\_demo.py

│

├── checkpoints/         # Saved model checkpoints

│

└── README.md



The .venv directory contains the isolated Python environment and installed packages.



Project code, datasets, checkpoints, and other project files should remain outside .venv.



Step 1: Creating the Python Environment



A Python virtual environment was created using:



python -m venv .venv



The environment is activated from Command Prompt using:



.venv\\Scripts\\activate.bat



When active, the command prompt displays:



(.venv)

Why Use a Virtual Environment?



The virtual environment keeps packages used by this project separate from the main Python installation.



For example:



Windows

│

├── System Python

│

└── llm-project

&#x20;   └── .venv

&#x20;       └── PyTorch and project dependencies



This helps prevent dependency conflicts between projects and makes the environment easier to reproduce later.



Step 2: Installing PyTorch



PyTorch was installed as the framework used to build and train the neural network.



The first installation resulted in a CPU-only build:



PyTorch version: 2.13.0+cpu

CUDA available: False



This meant the RTX 5070 could not be used.



The CPU-only version was removed and a CUDA-enabled build was installed.



The final installation successfully detected:



PyTorch version: 2.11.0+cu128

CUDA available: True

CUDA version: 12.8

GPU: NVIDIA GeForce RTX 5070

Why PyTorch?



PyTorch provides:



Tensors for storing and processing numerical data

Fast mathematical operations

GPU acceleration through CUDA

Neural network components

Automatic differentiation

Gradient calculation for training

Optimizers and loss functions



These components will be used to build and train the language model.



Step 3: Understanding Tensors



A tensor is a multidimensional container of numbers.



Examples include:



0 Dimensions:

5

1 Dimension:

\[1, 2, 3, 4]

2 Dimensions:

\[

&#x20;\[1, 2, 3],

&#x20;\[4, 5, 6]

]



Language models represent their data and parameters using tensors.



For example, token IDs may be represented as:



\[3, 11, 8, 1, 6, 4, 21]



These tensors can be moved between the CPU and GPU.



A tensor created normally starts on the CPU:



x = torch.tensor(\[1, 2, 3, 4])

print(x.device)



Output:



cpu



It can be moved to the GPU:



x = x.to("cuda")



Output:



cuda:0



This confirmed that PyTorch can successfully move tensors to the RTX 5070.



Step 4: Understanding the LLM Training Process



The basic task of the language model is:



Given previous text, predict what comes next.



For example:



Input:

"The capital of France is"





Target:

"Paris"



Our small model currently works at the character level.



The overall training process will eventually be:



Input Tokens

&#x20;     ↓

Model Prediction

&#x20;     ↓

Compare With Correct Token

&#x20;     ↓

Calculate Loss

&#x20;     ↓

Calculate Gradients

&#x20;     ↓

Update Model Parameters

&#x20;     ↓

Repeat



This process allows the model to gradually improve its predictions.



Step 5: Creating a Character-Level Tokenizer



The first tokenizer created for this project is a character-level tokenizer.



Each unique character in the training text receives its own token ID.



For example:



Text:





Hello



Could become:



H → ID

e → ID

l → ID

l → ID

o → ID



Repeated characters use the same ID.



Spaces and special characters are also tokens.



For example:



'n'  = lowercase letter n

'\\n' = newline character

' '  = space

'.'  = period



These are all separate tokens.



The training dataset produced a vocabulary of 26 unique characters.



Example mappings include:



'\\n' → 0

' '  → 1

'.'  → 2

'T'  → 3

'a'  → 4

...

'n'  → 16

...

't'  → 21



Token IDs are labels rather than numerical meanings.



For example:



'a' → 4

'b' → 5



The fact that the IDs are numerically close does not mean the model considers a and b linguistically similar.



Step 6: Encoding Text



The tokenizer converts text into token IDs.



For example:



The cat



can become:



\[3, 11, 8, 1, 6, 4, 21]



The tokenizer can also decode IDs back into text.



The process is:



Text

&#x20;↓

Tokenizer

&#x20;↓

Token IDs

&#x20;↓

\[3, 11, 8, 1, 6, 4, 21]



And in reverse:



Token IDs

&#x20;↓

Reverse Vocabulary

&#x20;↓

Text



The tokenizer was tested by encoding text and decoding it back successfully.



Step 7: Preparing the Training Data



The entire training text contains 197 characters.



Because this project currently uses character-level tokenization:



197 characters

=

197 tokens



The entire dataset was converted into a PyTorch tensor:



Tensor shape: torch.Size(\[197])



This means the tensor currently contains one dimension with 197 token IDs.



Example:



tensor(\[

&#x20;3, 11, 8, 1, 6, 4, 21, ...

])



The first 20 tokens decode to:



'The cat sat on the m'

Step 8: Creating Input and Target Examples



The language model needs examples in the format:



Previous tokens → Next token



A context length of 8 was selected for the first experiment.



This means the model sees 8 previous tokens and predicts the following token.



Examples generated from the training data include:



Input:  'The cat '

Target: 's'

Input:  'he cat s'

Target: 'a'

Input:  'e cat sa'

Target: 't'

Input:  ' cat sat'

Target: ' '

Input:  'cat sat '

Target: 'o'



These examples are created using a sliding window.



Conceptually:



The cat sat on...

└───────┘ → Predict next token





&#x20;The cat sat on...

&#x20; └───────┘ → Predict next token



The window moves forward one token at a time.



This allows many training examples to be generated automatically from continuous text.



Step 9: Token Embeddings



Token IDs cannot simply be treated as meaningful numerical values.



For example:



'a' → 4

'b' → 5



The difference between 4 and 5 does not represent the linguistic relationship between the letters.



Instead, token IDs are used as indexes into an embedding table.



Conceptually:



Token ID

&#x20;  ↓

Embedding Table

&#x20;  ↓

Vector



Example:



Token 3

&#x20;  ↓

\[-0.2061, 0.9000, -0.6692, 0.8450]



For the current experiment:



Vocabulary size: 26

Embedding size: 4



The embedding layer contains one vector for each possible token.



Conceptually:



Token ID 0  → \[four learned numbers]

Token ID 1  → \[four learned numbers]

Token ID 2  → \[four learned numbers]

...

Token ID 25 → \[four learned numbers]



An input tensor with eight token IDs:



\[3, 11, 8, 1, 6, 4, 21, 1]



has the shape:



torch.Size(\[8])



After passing through the embedding layer, each token becomes a vector containing four numbers:



torch.Size(\[8, 4])



This represents:



8 tokens

×

4 values per token



The same token always retrieves the same embedding vector from the embedding table.



For example, because the space token appeared twice in the input, both positions initially produced the same embedding vector.



The embedding values are initially random and become useful through training.



The future training process will be:



Random Parameters

&#x20;      ↓

Prediction

&#x20;      ↓

Loss

&#x20;      ↓

Backpropagation

&#x20;      ↓

Parameter Updates

&#x20;      ↓

Improved Parameters



PyTorch tracks how tensors are produced so gradients can later flow backward through the model.



This was visible in the embedding output:



grad\_fn=<EmbeddingBackward0>



This indicates that PyTorch can track the embedding operation for gradient calculations during backpropagation.



Current Progress



Completed:



&#x20;Created the project directory

&#x20;Created and activated a Python virtual environment

&#x20;Installed PyTorch

&#x20;Installed a CUDA-enabled PyTorch build

&#x20;Confirmed CUDA is available

&#x20;Confirmed the RTX 5070 is accessible to PyTorch

&#x20;Tested moving tensors from CPU to GPU

&#x20;Learned the basics of PyTorch tensors

&#x20;Created a training text dataset

&#x20;Built a character-level tokenizer

&#x20;Created character-to-ID mappings

&#x20;Created ID-to-character mappings

&#x20;Encoded training text into token IDs

&#x20;Converted the dataset into a PyTorch tensor

&#x20;Created sliding-window input and target examples

&#x20;Created and tested a token embedding layer



Next:



&#x20;Learn positional embeddings

&#x20;Combine token and positional embeddings

&#x20;Understand self-attention

&#x20;Build a self-attention layer

&#x20;Build multi-head attention

&#x20;Build a Transformer block

&#x20;Build the complete language model

&#x20;Add next-token prediction

&#x20;Calculate training loss

&#x20;Implement backpropagation

&#x20;Train the model

&#x20;Generate text

&#x20;Save and load model checkpoints

&#x20;Improve the tokenizer

&#x20;Train on a larger dataset

Current Pipeline



The project has currently implemented:



Training Text

&#x20;     ↓

Character-Level Tokenizer

&#x20;     ↓

Token IDs

&#x20;     ↓

PyTorch Tensor

&#x20;     ↓

Input/Target Sequences

&#x20;     ↓

Token Embeddings



The next component will add positional information, allowing the model to distinguish where tokens occur within a sequence.

