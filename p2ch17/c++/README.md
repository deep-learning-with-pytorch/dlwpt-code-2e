First ensure you have Cmake:

`brew install cmake`

Generate build files:

```bash
cmake -B build -S . -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
```

Compile the project:

```bash
cmake --build build --config Release
```

Run the application:

```bash
./build/example
```

compile and run to develop:
```bash
cmake --build build --config Release && ./build/example
```