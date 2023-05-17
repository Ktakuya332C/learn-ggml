# learn-ggml
A tiny ggml implementation that only implements `add` and `dup` .

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
