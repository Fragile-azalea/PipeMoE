# PipeMoE
PipeMoE: Accelerating Mixture-of-Experts through Adaptive Pipelining 

## Requirements

To install requirements:
```
pip install scipy==1.7.1
pip install numpy
```

## Using Our Solver in Tutel

### Install Tutel

```
git clone https://github.com/microsoft/tutel --branch main
cd tutel/ && git checkout 2c0cad3a742ecf4c0b0a989d6db629fcc2022bc0
python3 -m pip uninstall tutel -y
python3 install -e .
```

### Apply Patches

```
git apply ../tutel_patch.diff
cp ../solver.py ./tutel/examples/
```