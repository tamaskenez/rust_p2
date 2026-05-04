# Simple B-rep in Rust

## The project provides:

- lib.rs which contains the core B-rep implementation and the push-pull operation
- primitives.rs with functions to create basic geometric primitives for testing
- the viewer.rs executable which is a simple 3D viewer to load geometric primitives and perform push-pull operations

## Viewer instructions

- Click on "cube" or "ramp" to load a model
- Rotate the model with the mouse
- Select a face and move it by dragging with the mouse

## TODO

- [X] Bug: face can't be pushed in certain cases
- [X] Colinear edges must be merged after pull
- [ ] Refactor/cleanup/missing tests
- [ ] Viewer could remember the last known good state, instead of jumping back to the initial state
- [ ] Don't allow self-intersection

## Future possibilities

- Allow pulling along concave, perpendicular adjacent face
