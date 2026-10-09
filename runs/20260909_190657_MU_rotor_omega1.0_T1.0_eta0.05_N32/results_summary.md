# Propagator results — MU_rotor_omega1.0_T1.0_eta0.05_N32

Run: `runs\20260909_190657_MU_rotor_omega1.0_T1.0_eta0.05_N32`  
Generated: 2026-09-09T19:18:50

Model: `x = Q[(s*z) * exp(i theta(z))]  (pure rotation)`, loss = scaled phase + descent, Adam lr=0.001, batch=512, epochs=10000

Exact thimble diagnostic: on the true contour `<Im S> = |z|^2/2 = 15.5`

Complex time `T_c = 1.0 - 0.05i`, `N = 32`, `D = 31`, endpoints `(0.0, 0.0)`


## lambda = 0.0

- N=32 lattice reference: `(0.3121159-0.3022604j)`
- continuum reference:    `(0.312137-0.3022849j)`

- exact \|K\|: `0.43448554`

| seed | K (complex) | \|K\| | ESS/N | phase | descent | <Im S> | Re err % | Im err % | sign |
|---|---|---|---|---|---|---|---|---|---|
| 52 | `-0.3121716282+0.3022028777j` | 0.43448556 | 1.00e+00 | 1.84e-05 | 0.00e+00 | 15.346 | 0.018 | 0.019 | -K |

Seed scatter `|std(K)| = 0.0000e+00` (0.00% of mean |K|)

## lambda = 0.7

- N=32 lattice reference: `(0.2684413-0.2878695j)`
- continuum reference:    `(0.2684401-0.2897428j)`

- exact \|K\|: `0.39361095`

| seed | K (complex) | \|K\| | ESS/N | phase | descent | <Im S> | Re err % | Im err % | sign |
|---|---|---|---|---|---|---|---|---|---|
| 52 | `-0.3189801601+0.2965497085j` | 0.43553424 | 9.99e-01 | 3.08e-06 | 0.00e+00 | 15.344 | 18.827 | 3.015 | -K |

Seed scatter `|std(K)| = 0.0000e+00` (0.00% of mean |K|)

## lambda = 2.4

- N=32 lattice reference: `(-0.110904-0.1529262j)`
- continuum reference:    `(-0.1118026-0.1577383j)`

- exact \|K\|: `0.18890770`

| seed | K (complex) | \|K\| | ESS/N | phase | descent | <Im S> | Re err % | Im err % | sign |
|---|---|---|---|---|---|---|---|---|---|
| 52 | `-0.3323992925+0.2801949834j` | 0.43473960 | 9.96e-01 | 5.16e-06 | 0.00e+00 | 15.344 | 199.718 | 283.222 | +K |

Seed scatter `|std(K)| = 0.0000e+00` (0.00% of mean |K|)
