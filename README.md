# Structure-preserving shallow waters FEM discretization

Implements a structure preserving discretization for the shallow waters equations using:

- a rotational formulation derived from a Hamiltonian formulation
- a spatial discretizations using Finite Element Exterior Calculus (FEEC), both on quadrilaterals using mimetic spectral element basis functions and on simplices using standard Raviart-Thomas basis functions
- and geometric integrators (symplectic midpoint integrator, and energy preserving Poisson integrator)

## Generate the python notebook
```bash
jupytext --to ipynb shallow_waters.py
```
