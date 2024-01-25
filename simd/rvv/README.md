# Introduction

This `rvv` SIMD extension is based on RVV 1.0 version (frozen), implemented by utilizing RVV intrinsics. This implementation is now compatible with the newest rvv-intrinsic version of v0.12.0, which is determined by the compiler used.


## Build

The libjpeg-turbo can be cross-compiled targeting 64-bit RISC-V arch with V extension, so all we need to do is to provide a `toolchain.cmake` file:

```cmake
# borrowed from opencv: https://github.com/opencv/opencv/blob/4.x/platforms/linux/riscv64-clang.toolchain.cmake

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV_CLANG_BUILD_ROOT /path/to/my/llvm/build CACHE PATH "Path to CLANG for RISC-V cross compiler build directory")
set(RISCV_GCC_INSTALL_ROOT /opt/riscv CACHE PATH "Path to GCC for RISC-V cross compiler installation directory")
set(CMAKE_SYSROOT ${RISCV_GCC_INSTALL_ROOT}/sysroot CACHE PATH "RISC-V sysroot")

set(CLANG_TARGET_TRIPLE riscv64-unknown-linux-gnu)

set(CMAKE_C_COMPILER ${RISCV_CLANG_BUILD_ROOT}/bin/clang)
set(CMAKE_C_COMPILER_TARGET ${CLANG_TARGET_TRIPLE})
set(CMAKE_CXX_COMPILER ${RISCV_CLANG_BUILD_ROOT}/bin/clang++)
set(CMAKE_CXX_COMPILER_TARGET ${CLANG_TARGET_TRIPLE})
set(CMAKE_ASM_COMPILER ${RISCV_CLANG_BUILD_ROOT}/bin/clang)
set(CMAKE_ASM_COMPILER_TARGET ${CLANG_TARGET_TRIPLE})
# add crosscompiling emulator here to run test-suite locally
set(CMAKE_CROSSCOMPILING_EMULATOR /path/to/my/qemu-riscv64 "-cpu" "rv64,v=true,vlen=128")

# Don't run the linker on compiler check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_FLAGS "-march=rv64gcv --gcc-toolchain=${RISCV_GCC_INSTALL_ROOT} -w ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-march=rv64gcv --gcc-toolchain=${RISCV_GCC_INSTALL_ROOT} -w ${CXX_FLAGS}")

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

Then following the commands below to build the libjpeg-turbo with `rvv`:
```bash
$ git clone https://github.com/isrc-cas/libjpeg-turbo.git -b riscv-dev
$ cd libjpeg-turbo
$ mkdir build && cd build
$ cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake -DENABLE_SHARED=FALSE ..
$ ninja
```

*(Note: The real machine we have only supports static binary, that is why `ENABLE_SHARED` is disabled here. )*

To verify the correctness of build artifacts, simply run:
```bash
$ ninja test
```

and if the end of test output looks like something below, it means successful build:
```bash
...
100% tests passed, 0 tests failed out of 287

Total Test time (real) = 609.20 sec
```

## Performance Comparison

Thanks for the first device that support rvv 1.0 ([Kendryte K230](https://www.canaan.io/product/k230)), we can make a comparison over the performance between the scale version and the rvv simd version on physical (real) machine.

As there is no build system on K230, we rewrite a test script to record the execution time of each test program based on `libjpeg-turbo/CMakeLists.txt`.

The comparison results are listed below, where we record execution time by `time` command provided by Linux:

*(Note: we remove the execution time of `cp` and `md5cmp`, which are meaningless to our comparison. )*

| test_name | scale | rvv | speedup |
|:---------:|:-----:|:---:|:-------:|
| tjunittest-static | 46.810 | 38.623 | **1.212** |
| tjunittest-static-alloc | 46.756 | 38.423 | **1.217** |
| tjunittest-static-yuv | 19.811 | 22.159 | 0.894 |
| tjunittest-static-yuv-alloc | 19.584 | 22.116 | 0.895 |
| tjunittest-static-lossless | 10.828 | 11.866 | 0.913 |
| tjunittest-static-lossless-alloc | 10.546 | 11.556 | 0.913 |
| tjunittest-static-bmp | 0.153 | 0.153 | 1 |
| tjbench-static-tile | 1.082 | 0.973 | **1.112** |
| tjbench-static-tilem | 1.058 | 0.973 | **1.087** |
|  example-8bit-static-compress | 0.057 | 0.033 | **1.727** |
| example-8bit-static-decompress | 0.032 | 0.031 | **1.032** |

As we only optimize the code for the 8-bit input sample, we do not take 12-bit or 16-bit tests into our consideration for now (but a future work). We only show the comparison of tests that ran enough time, because the performance improvement might be caused by measuring error.

In general, this implementation can be $1ms$ or $2ms$ faster than the scale version in most 8-bit tests, but this cannot prove the effectiveness of our implementation due to error caused by system or measurements.

Following the tips disscused [here](https://github.com/libjpeg-turbo/libjpeg-turbo/issues/620), we futher test our implementation using `tjbench-static`. We test the compression process by running `tjbench-static {image} 95 -rgb -qq -nowrite -warmup 10` command, while leaving the decompression core functions to be as the same as the original ones to eliminate the error. And for testing the decompression process we also disabled all the optimization for the compression core functions. The results are listed bellow just for reference, as there are currently only Kendryte K230 available that support RVV 1.0.

- Results for compression process using accurate FDCT (`tjbench-static {image} 95 -rgb -qq -nowrite -warmup 10`)

<table>
    <tr>
        <th>image</th>
        <th>sample</th>
        <th>metrics</th>
        <th>scale</th>
        <th>rvv</th>
        <th>speedup</th>
    </tr>
    <tr>
        <td rowspan="12" align="center"><i>vgl_5674_0098.ppm</i></td>
        <td rowspan="3" align="center">Gray</td>
        <td align="center">Comp. Perf</td>
        <td align="center">15.59</td>
        <td align="center">26.47</td>
        <td align="center"><b>1.698</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">8.355</td>
        <td align="center">8.355</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">32.69</td>
        <td align="center">32.69</td>
        <td align="center">1.000</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:0</td>
        <td align="center">Comp. Perf</td>
        <td align="center">9.192</td>
        <td align="center">17.63</td>
        <td align="center"><b>1.918</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">7.128</td>
        <td align="center">7.128</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">15.30</td>
        <td align="center">15.32</td>
        <td align="center">1.001</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:2</td>
        <td align="center">Comp. Perf</td>
        <td align="center">7.487</td>
        <td align="center">13.98</td>
        <td align="center"><b>1.867</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">6.175</td>
        <td align="center">6.175</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">14.07</td>
        <td align="center">14.09</td>
        <td align="center">1.001</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:4:4</td>
        <td align="center">Comp. Perf</td>
        <td align="center">5.696</td>
        <td align="center">10.27</td>
        <td align="center"><b>1.803</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">4.796</td>
        <td align="center">4.796</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">11.68</td>
        <td align="center">11.73</td>
        <td align="center">1.004</td>
    </tr>
    <tr>
        <td rowspan="12" align="center"><i>vgl_6434_0018.ppm</i></td>
        <td rowspan="3" align="center">Gray</td>
        <td align="center">Comp. Perf</td>
        <td align="center">17.40</td>
        <td align="center">32.10</td>
        <td align="center"><b>1.845</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">22.28</td>
        <td align="center">22.27</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">49.75</td>
        <td align="center">49.60</td>
        <td align="center">0.997</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:0</td>
        <td align="center">Comp. Perf</td>
        <td align="center">9.874</td>
        <td align="center">20.30</td>
        <td align="center"><b>2.056</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">16.19</td>
        <td align="center">16.19</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">18.99</td>
        <td align="center">19.01</td>
        <td align="center">1.001</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:2</td>
        <td align="center">Comp. Perf</td>
        <td align="center">8.041</td>
        <td align="center">16.04</td>
        <td align="center"><b>1.995</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">13.71</td>
        <td align="center">13.71</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">18.12</td>
        <td align="center">18.13</td>
        <td align="center">1.001</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:4:4</td>
        <td align="center">Comp. Perf</td>
        <td align="center">6.137</td>
        <td align="center">11.80</td>
        <td align="center"><b>1.912</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">10.47</td>
        <td align="center">10.47</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">15.91</td>
        <td align="center">15.97</td>
        <td align="center">1.004</td>
    </tr>
    <tr>
        <td rowspan="12" align="center"><i>vgl_6548_0026.ppm</i></td>
        <td rowspan="3" align="center">Gray</td>
        <td align="center">Comp. Perf</td>
        <td align="center">16.49</td>
        <td align="center">29.02</td>
        <td align="center"><b>1.760</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">11.08</td>
        <td align="center">11.08</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">41.60</td>
        <td align="center">41.34</td>
        <td align="center">0.994</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:0</td>
        <td align="center">Comp. Perf</td>
        <td align="center">9.810</td>
        <td align="center">19.61</td>
        <td align="center"><b>1.999</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">10.87</td>
        <td align="center">10.87</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">18.77</td>
        <td align="center">18.70</td>
        <td align="center">0.996</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:2</td>
        <td align="center">Comp. Perf</td>
        <td align="center">8.056</td>
        <td align="center">15.85</td>
        <td align="center"><b>1.967</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">10.72</td>
        <td align="center">10.72</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">18.63</td>
        <td align="center">18.55</td>
        <td align="center">0.996</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:4:4</td>
        <td align="center">Comp. Perf</td>
        <td align="center">6.224</td>
        <td align="center">11.98</td>
        <td align="center"><b>1.925</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">10.41</td>
        <td align="center">10.41</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">17.35</td>
        <td align="center">17.41</td>
        <td align="center">1.003</td>
    </tr>
    <tr>
        <td rowspan="12" align="center"><i>artificial.ppm</i></td>
        <td rowspan="3" align="center">Gray</td>
        <td align="center">Comp. Perf</td>
        <td align="center">17.53</td>
        <td align="center">32.48</td>
        <td align="center"><b>1.853</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">24.33</td>
        <td align="center">24.33</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">47.20</td>
        <td align="center">47.13</td>
        <td align="center">0.999</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:0</td>
        <td align="center">Comp. Perf</td>
        <td align="center">9.819</td>
        <td align="center">20.36</td>
        <td align="center"><b>2.074</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">18.87</td>
        <td align="center">18.87</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">18.47</td>
        <td align="center">18.48</td>
        <td align="center">1.001</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:2</td>
        <td align="center">Comp. Perf</td>
        <td align="center">8.121</td>
        <td align="center">16.50</td>
        <td align="center"><b>2.032</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">16.80</td>
        <td align="center">16.80</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">17.77</td>
        <td align="center">17.76</td>
        <td align="center">0.999</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:4:4</td>
        <td align="center">Comp. Perf</td>
        <td align="center">6.146</td>
        <td align="center">12.11</td>
        <td align="center"><b>1.970</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">14.37</td>
        <td align="center">14.37</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">15.77</td>
        <td align="center">15.85</td>
        <td align="center">1.005</td>
    </tr>
    <tr>
        <td rowspan="12" align="center"><i>nightshot_iso_100.ppm</i></td>
        <td rowspan="3" align="center">Gray</td>
        <td align="center">Comp. Perf</td>
        <td align="center">17.34</td>
        <td align="center">31.53</td>
        <td align="center"><b>1.818</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">19.32</td>
        <td align="center">19.32</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">36.08</td>
        <td align="center">36.06</td>
        <td align="center">0.999</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:0</td>
        <td align="center">Comp. Perf</td>
        <td align="center">9.691</td>
        <td align="center">19.94</td>
        <td align="center"><b>2.058</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">15.83</td>
        <td align="center">15.83</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">16.01</td>
        <td align="center">16.07</td>
        <td align="center">1.004</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:2:2</td>
        <td align="center">Comp. Perf</td>
        <td align="center">8.027</td>
        <td align="center">16.09</td>
        <td align="center"><b>2.004</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">13.70</td>
        <td align="center">13.70</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">14.76</td>
        <td align="center">14.72</td>
        <td align="center">0.997</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">4:4:4</td>
        <td align="center">Comp. Perf</td>
        <td align="center">6.082</td>
        <td align="center">11.76</td>
        <td align="center"><b>1.934</b></td>
    </tr>
    <tr>
        <td align="center">Comp. Ratio</td>
        <td align="center">10.99</td>
        <td align="center">10.99</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">Decomp. Perf</td>
        <td align="center">12.20</td>
        <td align="center">12.26</td>
        <td align="center">1.005</td>
    </tr>
    <tr>
        <td rowspan="4" align="center"><i>AVERAGE</i></td>
        <td align="center">Gray</td>
        <td align="center">Comp. Perf</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center"><b>1.795</b></td>
    </tr>
    <tr>
        <td align="center">4:2:0</td>
        <td align="center">Comp. Perf</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center"><b>2.021</b></td>
    </tr>
    <tr>
        <td align="center">4:2:2</td>
        <td align="center">Comp. Perf</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center"><b>1.973</b></td>
    </tr>
    <tr>
        <td align="center">4:4:4</td>
        <td align="center">Comp. Perf</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center"><b>1.909</b></td>
    </tr>
</table>
