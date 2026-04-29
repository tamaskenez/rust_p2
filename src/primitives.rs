use nalgebra::{Point3, Unit, Vector3};

use crate::{Edge, EdgeId, Face, FaceId, Mesh, OrientedEdge, OrientedEdgeId, Vertex, VertexId};

/// Creates a unit cube with vertices at (0,0,0)–(1,1,1).
/// Face normals point outward; oriented edges wind counter-clockwise when
/// viewed from outside.
pub fn make_cube() -> Mesh {
    let vertices = vec![
        Vertex {
            position: Point3::new(0.0, 0.0, 0.0),
        }, // 0
        Vertex {
            position: Point3::new(1.0, 0.0, 0.0),
        }, // 1
        Vertex {
            position: Point3::new(1.0, 1.0, 0.0),
        }, // 2
        Vertex {
            position: Point3::new(0.0, 1.0, 0.0),
        }, // 3
        Vertex {
            position: Point3::new(0.0, 0.0, 1.0),
        }, // 4
        Vertex {
            position: Point3::new(1.0, 0.0, 1.0),
        }, // 5
        Vertex {
            position: Point3::new(1.0, 1.0, 1.0),
        }, // 6
        Vertex {
            position: Point3::new(0.0, 1.0, 1.0),
        }, // 7
    ];

    // 12 edges; OE slot 0 = forward (id 2i), slot 1 = reversed (id 2i+1)
    let edge_verts: [[u32; 2]; 12] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0], // bottom ring  (0–3)
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4], // top ring     (4–7)
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7], // verticals    (8–11)
    ];
    let edges: Vec<Edge> = edge_verts
        .into_iter()
        .enumerate()
        .map(|(i, [v0, v1])| Edge {
            vertices: [VertexId(v0), VertexId(v1)],
            oriented_edges: [
                OrientedEdgeId(2 * i as u32),
                OrientedEdgeId(2 * i as u32 + 1),
            ],
        })
        .collect();

    // 24 oriented edges; face back-ref filled below
    let mut oriented_edges: Vec<OrientedEdge> = (0u32..12)
        .flat_map(|i| {
            [
                OrientedEdge {
                    edge: EdgeId(i),
                    forward: true,
                    face: FaceId(u32::MAX),
                },
                OrientedEdge {
                    edge: EdgeId(i),
                    forward: false,
                    face: FaceId(u32::MAX),
                },
            ]
        })
        .collect();

    // oe(edge_idx, forward) → OrientedEdgeId (forward=2i, reversed=2i+1)
    let oe = |e: u32, fwd: bool| OrientedEdgeId(2 * e + if fwd { 0 } else { 1 });

    // Each face: (oriented-edge loop, outward normal).
    // Loops are counter-clockwise when viewed from outside.
    let face_data: [(Vec<OrientedEdgeId>, Vector3<f64>); 6] = [
        // bottom z=0, normal (0,0,-1): 0→3→2→1
        (
            vec![oe(3, false), oe(2, false), oe(1, false), oe(0, false)],
            Vector3::new(0.0, 0.0, -1.0),
        ),
        // top    z=1, normal (0,0,+1): 4→5→6→7
        (
            vec![oe(4, true), oe(5, true), oe(6, true), oe(7, true)],
            Vector3::new(0.0, 0.0, 1.0),
        ),
        // front  y=0, normal (0,-1,0): 0→1→5→4
        (
            vec![oe(0, true), oe(9, true), oe(4, false), oe(8, false)],
            Vector3::new(0.0, -1.0, 0.0),
        ),
        // back   y=1, normal (0,+1,0): 2→3→7→6
        (
            vec![oe(2, true), oe(11, true), oe(6, false), oe(10, false)],
            Vector3::new(0.0, 1.0, 0.0),
        ),
        // left   x=0, normal (-1,0,0): 0→4→7→3
        (
            vec![oe(8, true), oe(7, false), oe(11, false), oe(3, true)],
            Vector3::new(-1.0, 0.0, 0.0),
        ),
        // right  x=1, normal (+1,0,0): 1→2→6→5
        (
            vec![oe(1, true), oe(10, true), oe(5, false), oe(9, false)],
            Vector3::new(1.0, 0.0, 0.0),
        ),
    ];

    let mut faces = Vec::with_capacity(6);
    for (fid, (oe_list, normal_vec)) in face_data.into_iter().enumerate() {
        for &oe_id in &oe_list {
            oriented_edges[oe_id.0 as usize].face = FaceId(fid as u32);
        }
        faces.push(Face {
            oriented_edges: oe_list,
            normal: Some(Unit::new_normalize(normal_vec)),
        });
    }

    Mesh {
        vertices,
        edges,
        oriented_edges,
        faces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate_mesh;

    #[test]
    fn cube_topology() {
        let m = make_cube();

        let errors = validate_mesh(&m);
        assert!(
            errors.is_empty(),
            "mesh validation errors:\n{}",
            errors.join("\n")
        );

        // Cube-specific counts.
        assert_eq!(m.vertices.len(), 8);
        assert_eq!(m.edges.len(), 12);
        assert_eq!(m.oriented_edges.len(), 24);
        assert_eq!(m.faces.len(), 6);

        // All faces are quads and have an outward normal.
        for (i, f) in m.faces.iter().enumerate() {
            assert_eq!(f.oriented_edges.len(), 4, "face {i} is not a quad");
            assert!(f.normal.is_some(), "face {i} has no normal");
        }
    }
}
