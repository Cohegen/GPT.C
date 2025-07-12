# GPT.C

An implementation of GPT-style neural network components in C.

## Project Structure

- `main.c`: Program entry point and orchestration logic.
- `attention.c` / `attention.h`: Implements attention mechanisms essential to transformer models.
- `block.c` / `block.h`: Defines transformer blocks, likely assembling layers and attention.
- `data.c` / `data.h`: Handles data input/output and possibly preprocessing.
- `feed_forward.c` / `feed_forward.h`: Implements the feed-forward layers of the network.
- `layer_norm.c` / `layer_norm.h`: Layer normalization routines for stabilizing training.
- `linear.c` / `linear.h`: Fully connected layers and their operations.
- `model.c` / `model.h`: Model definition, initialization, and execution.
- `tensor.c` / `tensor.h`: Tensor operations, storage, and manipulation.
- `Makefile`: Build instructions for compiling the project.
- `pride_and_prejudice.txt`: Sample dataset for testing or demonstration.

## Building

```sh
make
```

## Usage

Run the compiled program:

```sh
./gptc
```

*(Replace `gptc` with the actual binary name produced by your Makefile.)*

## Data

The file `pride_and_prejudice.txt` is included as an example dataset. You can replace this with any text corpus for training or inference.



## License

See [LICENSE](LICENSE).

---

**Note:**  
This summary is based on a partial listing of repository files.  
For a complete and up-to-date file listing, (https://github.com/Cohegen/GPT.C/tree/main).
