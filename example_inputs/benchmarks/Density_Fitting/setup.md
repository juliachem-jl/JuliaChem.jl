# Setup for benchmarking runs for Julia Chem Density Fitting Paper

setup sysimg

``` 
julia -q
] 
activate . 
add /home/jackson/source/JuliaChem.jl#your_branch_here
using Pkg; Pkg.add("PackageCompiler")
using PackageCompiler
PackageCompiler.create_sysimage(["JuliaChem"]; sysimage_path="JuliaChem.so")

``` 
run such as 

```
export JULIA_NUM_THREADS=20
export OMP_NUM_THREADS=20

julia -J"../SYSIMG/JuliaChem.so" --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes ./test/density-fitting-test.jl > $outputpath
```