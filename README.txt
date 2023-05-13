# simple-gpt2
A simple implementation of GPT-2 inference without any optimization

## Example
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./examples/main
```

## Tests
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ctest
```

## Notes
- I created this project to learn the inner workings of [ggml](https://github.com/ggerganov/ggml) .
- There's no optimization whatsoever, even matrix calculations are implemented in simple nested loops.
