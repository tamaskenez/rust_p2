pub mod step;
pub use step::{StepError, load_step};

pub mod primitives;
pub use primitives::make_cube;

use nalgebra::Point3;
use nalgebra::Unit;
use nalgebra::Vector3;

pub const EPS_LENGTH_USER: f64 = 0.01;
pub const EPS_LENGTH_SYSTEM: f64 = 1e-12;
pub const EPS_ANGLE_USER: f64 = 0.05 * (std::f64::consts::PI / 180.0);
pub const EPS_ANGLE_SYSTEM: f64 = 1e-11;
pub const FACE_NORMAL_CHECK_MIN_COS_ANGLE: f64 = 1.0 - 1e-11;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexId(u32);

impl VertexId {
    pub fn new(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(u32);

impl EdgeId {
    pub fn new(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrientedEdgeId(u32);

impl OrientedEdgeId {
    pub const INVALID: Self = Self(u32::MAX);
    pub fn new(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceId(u32);

impl FaceId {
    pub const INVALID: Self = Self(u32::MAX);
    pub fn new(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone)]
pub struct Vertex {
    pub position: Point3<f64>,
}

/// Canonical direction is vertices[0] → vertices[1].
/// oriented_edges[0] is the forward wrapper, oriented_edges[1] is the reversed wrapper.
#[derive(Clone)]
pub struct Edge {
    pub vertices: [VertexId; 2],
    pub oriented_edges: [OrientedEdgeId; 2],
}

#[derive(Clone)]
pub struct OrientedEdge {
    pub edge: EdgeId,
    pub forward: bool,
    pub face: FaceId,
}

#[derive(Clone)]
pub struct Face {
    pub oriented_edges: Vec<OrientedEdgeId>,
    pub normal: Option<Unit<Vector3<f64>>>,
}

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub oriented_edges: Vec<OrientedEdge>,
    pub faces: Vec<Face>,
}

impl std::ops::Index<VertexId> for Vec<Vertex> {
    type Output = Vertex;
    fn index(&self, id: VertexId) -> &Vertex {
        &self[id.index()]
    }
}
impl std::ops::IndexMut<VertexId> for Vec<Vertex> {
    fn index_mut(&mut self, id: VertexId) -> &mut Vertex {
        &mut self[id.index()]
    }
}

impl std::ops::Index<EdgeId> for Vec<Edge> {
    type Output = Edge;
    fn index(&self, id: EdgeId) -> &Edge {
        &self[id.index()]
    }
}
impl std::ops::IndexMut<EdgeId> for Vec<Edge> {
    fn index_mut(&mut self, id: EdgeId) -> &mut Edge {
        &mut self[id.index()]
    }
}

impl std::ops::Index<OrientedEdgeId> for Vec<OrientedEdge> {
    type Output = OrientedEdge;
    fn index(&self, id: OrientedEdgeId) -> &OrientedEdge {
        &self[id.index()]
    }
}
impl std::ops::IndexMut<OrientedEdgeId> for Vec<OrientedEdge> {
    fn index_mut(&mut self, id: OrientedEdgeId) -> &mut OrientedEdge {
        &mut self[id.index()]
    }
}

impl std::ops::Index<FaceId> for Vec<Face> {
    type Output = Face;
    fn index(&self, id: FaceId) -> &Face {
        &self[id.index()]
    }
}
impl std::ops::IndexMut<FaceId> for Vec<Face> {
    fn index_mut(&mut self, id: FaceId) -> &mut Face {
        &mut self[id.index()]
    }
}

/// Validates mesh invariants. Returns `Ok(())` if valid, or `Err` with a list of error descriptions.
pub fn validate_mesh(mesh: &Mesh) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let n_faces = mesh.faces.len();
    let n_oes = mesh.oriented_edges.len();
    let n_edges = mesh.edges.len();
    let n_verts = mesh.vertices.len();

    // Every edge must reference valid vertices and valid OE back-refs.
    for (i, edge) in mesh.edges.iter().enumerate() {
        for (slot, &vid) in edge.vertices.iter().enumerate() {
            if vid.index() >= n_verts {
                errors.push(format!(
                    "edge {i}: vertex slot {slot} ref {} out of range (mesh has {n_verts} vertices)",
                    vid.index(),
                ));
            }
        }
        for (slot, &oe_id) in edge.oriented_edges.iter().enumerate() {
            if oe_id.index() >= n_oes {
                errors.push(format!(
                    "edge {i}: OE slot {slot} ref {} out of range (mesh has {n_oes} oriented edges)",
                    oe_id.index(),
                ));
            } else {
                let oe = &mesh.oriented_edges[oe_id];
                if oe.edge != EdgeId::new(i) {
                    errors.push(format!(
                        "edge {i}: OE slot {slot} (id {}) back-references edge {}, expected {i}",
                        oe_id.index(),
                        oe.edge.index(),
                    ));
                }
                let expected_forward = slot == 0;
                if oe.forward != expected_forward {
                    errors.push(format!(
                        "edge {i}: OE slot {slot} (id {}) has forward={}, expected {expected_forward}",
                        oe_id.index(),
                        oe.forward,
                    ));
                }
            }
        }
    }

    // Every oriented edge must reference a valid edge and face.
    for (i, oe) in mesh.oriented_edges.iter().enumerate() {
        if oe.edge.index() >= n_edges {
            errors.push(format!(
                "oriented_edge {i}: edge ref {} out of range (mesh has {n_edges} edges)",
                oe.edge.index(),
            ));
        }
        if oe.face.index() >= n_faces {
            errors.push(format!(
                "oriented_edge {i}: face back-ref {} out of range (mesh has {n_faces} faces)",
                oe.face.index(),
            ));
        } else if !mesh.faces[oe.face]
            .oriented_edges
            .contains(&OrientedEdgeId::new(i))
        {
            errors.push(format!(
                "oriented_edge {i}: face back-ref {} does not contain it in its loop",
                oe.face.index(),
            ));
        }
    }

    // Each oriented edge must appear in exactly one face loop.
    let mut counts = vec![0u32; n_oes];
    for (fid, face) in mesh.faces.iter().enumerate() {
        for &oe_id in &face.oriented_edges {
            if oe_id.index() < n_oes {
                counts[oe_id.index()] += 1;
            } else {
                errors.push(format!(
                    "face {fid}: OE ref {} out of range (mesh has {n_oes} oriented edges)",
                    oe_id.index(),
                ));
            }
        }
    }
    for (i, &count) in counts.iter().enumerate() {
        match count {
            1 => {}
            0 => errors.push(format!("oriented_edge {i}: not referenced by any face")),
            n => errors.push(format!("oriented_edge {i}: referenced by {n} face loops")),
        }
    }

    // Consecutive oriented edges in each face loop must share a vertex.
    for (fid, face) in mesh.faces.iter().enumerate() {
        let oes = &face.oriented_edges;
        let n = oes.len();
        for j in 0..n {
            let oe_a_id = oes[j];
            let oe_b_id = oes[(j + 1) % n];
            if oe_a_id.index() >= n_oes || oe_b_id.index() >= n_oes {
                continue;
            }
            let oe_a = &mesh.oriented_edges[oe_a_id];
            let oe_b = &mesh.oriented_edges[oe_b_id];
            if oe_a.edge.index() >= n_edges || oe_b.edge.index() >= n_edges {
                continue;
            }
            let edge_a = &mesh.edges[oe_a.edge];
            let edge_b = &mesh.edges[oe_b.edge];
            let end_vid = if oe_a.forward {
                edge_a.vertices[1]
            } else {
                edge_a.vertices[0]
            };
            let start_vid = if oe_b.forward {
                edge_b.vertices[0]
            } else {
                edge_b.vertices[1]
            };
            if end_vid != start_vid {
                errors.push(format!(
                    "face {fid}: OE {j} ends at vertex {} but OE {} starts at vertex {}",
                    end_vid.index(),
                    (j + 1) % n,
                    start_vid.index(),
                ));
            }
        }
    }

    // Closed-manifold: each edge's two oriented edges must belong to different faces.
    for (i, edge) in mesh.edges.iter().enumerate() {
        let [oe0_id, oe1_id] = edge.oriented_edges;
        if oe0_id.index() >= n_oes || oe1_id.index() >= n_oes {
            continue;
        }
        let face0 = mesh.oriented_edges[oe0_id].face;
        let face1 = mesh.oriented_edges[oe1_id].face;
        if face0.index() < n_faces && face1.index() < n_faces && face0 == face1 {
            errors.push(format!(
                "edge {i}: both oriented edges belong to face {} (non-manifold)",
                face0.index(),
            ));
        }
    }

    // If a stored normal is present, it must match the computed one.
    // Only check faces whose geometry is fully in range to avoid panicking.
    for (i, face) in mesh.faces.iter().enumerate() {
        if face.oriented_edges.len() < 3 {
            errors.push(format!(
                "face {i}: has only {} vertices/edges",
                face.oriented_edges.len()
            ));
        }
        if let Some(stored) = &face.normal {
            let geometry_ok = face.oriented_edges.iter().all(|&oe_id| {
                if oe_id.index() >= n_oes {
                    return false;
                }
                let oe = &mesh.oriented_edges[oe_id];
                if oe.edge.index() >= n_edges {
                    return false;
                }
                let edge = &mesh.edges[oe.edge];
                edge.vertices.iter().all(|&vid| vid.index() < n_verts)
            });
            if geometry_ok {
                match compute_face_normal(mesh, FaceId::new(i)) {
                    None => errors.push(format!("face {i}: the normal can't be computed")),
                    Some(computed) => {
                        let cos_angle = computed.dot(stored);
                        if !cos_angle.is_finite() || cos_angle < FACE_NORMAL_CHECK_MIN_COS_ANGLE {
                            errors.push(format!(
                                "face {i}: stored normal {:?} does not match computed normal {:?}",
                                stored.as_ref(),
                                computed.as_ref(),
                            ));
                        }
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Computes the normal of a face using Newell's method.
/// Panics if `face_id` or any referenced id is out of range.
/// Returns None if the normal cannot be computed.
pub fn compute_face_normal(mesh: &Mesh, face_id: FaceId) -> Option<Unit<Vector3<f64>>> {
    let face = &mesh.faces[face_id];

    // Collect the start-vertex position of each oriented edge in loop order.
    let positions: Vec<_> = face
        .oriented_edges
        .iter()
        .map(|&oe_id| {
            let oe = &mesh.oriented_edges[oe_id];
            let edge = &mesh.edges[oe.edge];
            let vid = if oe.forward {
                edge.vertices[0]
            } else {
                edge.vertices[1]
            };
            mesh.vertices[vid].position
        })
        .collect();

    // Newell's method: robust for convex and non-planar polygons.
    let mut normal = Vector3::zeros();
    let n = positions.len();
    for i in 0..n {
        let vi = positions[i];
        let vj = positions[(i + 1) % n];
        normal.x += (vi.y - vj.y) * (vi.z + vj.z);
        normal.y += (vi.z - vj.z) * (vi.x + vj.x);
        normal.z += (vi.x - vj.x) * (vi.y + vj.y);
    }

    Unit::try_new(normal, EPS_LENGTH_SYSTEM)
}
