pub mod primitives;
pub use primitives::{make_cube, make_ramp};

use nalgebra::Point3;
use nalgebra::Unit;
use nalgebra::Vector3;

use std::collections::HashSet;

pub const EPS_LENGTH_USER: f64 = 0.01;
pub const EPS_LENGTH_SYSTEM: f64 = 1e-12;
pub const EPS_ANGLE_USER: f64 = 0.05 * (std::f64::consts::PI / 180.0);
pub const EPS_ANGLE_SYSTEM: f64 = 1e-11;
pub const FACE_NORMAL_CHECK_MIN_COS_ANGLE: f64 = 1.0 - 1e-11;
pub const MAX_ORTHOGONAL_COS_ANGLE: f64 = 1e-11;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct VertexId(u32);

impl VertexId {
    pub fn new(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct EdgeId(u32);

impl EdgeId {
    pub const INVALID: Self = Self(u32::MAX);
    pub fn new(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
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
    pub oriented_edges: [OrientedEdgeId; 2], // [0] is forward, [1] is backward.
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

impl Mesh {
    pub fn face_normal(&mut self, fid: FaceId) -> Option<Unit<Vector3<f64>>> {
        if let Some(n) = self.faces[fid].normal {
            return Some(n);
        }
        let n = compute_face_normal(self, fid)?;
        self.faces[fid].normal = Some(n);
        Some(n)
    }

    pub fn adjacent_faces(&self, fid: FaceId) -> Vec<FaceId> {
        self.faces[fid]
            .oriented_edges
            .iter()
            .map(|&oeid| {
                let oe = &self.oriented_edges[oeid];
                let oes_of_edge_of_oe = &self.edges[oe.edge].oriented_edges;
                // Find the twin oriented edge.
                let opposite_oeid = if oe.forward {
                    oes_of_edge_of_oe[1]
                } else {
                    oes_of_edge_of_oe[0]
                };
                self.oriented_edges[opposite_oeid].face
            })
            .collect()
    }
    pub fn edges_of_face(&self, fid: FaceId) -> Vec<EdgeId> {
        self.faces[fid]
            .oriented_edges
            .iter()
            .map(|&oeid| self.oriented_edges[oeid].edge)
            .collect()
    }
    pub fn vertices_of_face(&self, fid: FaceId) -> Vec<VertexId> {
        self.faces[fid]
            .oriented_edges
            .iter()
            .map(|&oeid| {
                let oe = &self.oriented_edges[oeid];
                self.edges[oe.edge].vertices[!oe.forward as usize] // Use the first vertex.
            })
            .collect()
    }
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
            0 => errors.push(format!("oriented_edge {i}: not referenced by any face")),
            1 => {}
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

struct PushPullWorkspace {
    pub face_normal: Unit<Vector3<f64>>,
    pub adjacent_faces: Vec<FaceId>,
    pub orthogonal_faces_sorted: Vec<FaceId>,
    pub all_faces_orthogonal: bool,
    pub vertices_of_face: Vec<VertexId>,
    pub edges_of_face: Vec<EdgeId>,
}

fn make_push_pull_workspace(mesh: &mut Mesh, fid: FaceId) -> Result<PushPullWorkspace, String> {
    let face_normal = mesh
        .face_normal(fid)
        .ok_or_else(|| format!("Face normal computation failed for face {}", fid.index()))?;

    // Enumerate adjacent faces and collect orthogonal faces.
    let adjacent_faces = mesh.adjacent_faces(fid);
    let mut orthogonal_faces_sorted: Vec<FaceId> = Vec::new();
    let mut all_faces_orthogonal = true;
    for &afid in &adjacent_faces {
        let adjacent_face_normal = mesh
            .face_normal(afid)
            .ok_or_else(|| format!("Face normal computation failed for face {}", afid.index()))?;
        if adjacent_face_normal.dot(face_normal.as_ref()).abs() < MAX_ORTHOGONAL_COS_ANGLE {
            orthogonal_faces_sorted.push(afid);
        } else {
            all_faces_orthogonal = false;
        }
    }

    orthogonal_faces_sorted.sort_unstable();

    Ok(PushPullWorkspace {
        face_normal: face_normal,
        adjacent_faces: adjacent_faces,
        orthogonal_faces_sorted,
        all_faces_orthogonal: all_faces_orthogonal,
        vertices_of_face: mesh.vertices_of_face(fid),
        edges_of_face: mesh.edges_of_face(fid), // Note: retrieving edges twice (also for vertices_of_face)
    })
}

fn assert_validate_mesh(mesh: &Mesh) {
    if cfg!(debug_assertions) {
        return;
    }
    if let Err(errors) = validate_mesh(mesh) {
        for e in &errors {
            println!("{e}");
        }
        panic!("Mesh validation failed");
    }
}

// Moves the face outwards with `offset` which must be positive.
// For now only works if all adjacent faces are orthogonal to the moved face.
fn pull_face(mesh: &mut Mesh, fid: FaceId, offset: f64) -> Result<(), String> {
    assert!(offset > 0.0);

    let wsp = make_push_pull_workspace(mesh, fid)?;

    let vertex_offset = wsp.face_normal.into_inner() * offset;
    if wsp.all_faces_orthogonal {
        // Move the vertices.
        for &vid in &wsp.vertices_of_face {
            mesh.vertices[vid].position += vertex_offset;
        }
    } else {
        assert_validate_mesh(mesh);

        let n = wsp.edges_of_face.len();
        assert_eq!(n, wsp.adjacent_faces.len());

        // Recreate the oriented edges and edges of the pulled face with new vertices.
        let first_pulled_vertex_id = VertexId::new(mesh.vertices.len());
        for i in 0..n {
            let pulled_position = mesh.vertices[wsp.vertices_of_face[i]].position + vertex_offset;
            mesh.vertices.push(Vertex {
                position: pulled_position,
            });
        }
        // Add the new edges of the pulled face.
        let first_pulled_edge_id = EdgeId::new(mesh.edges.len());
        for i in 0..n {
            let v0 = VertexId::new(first_pulled_vertex_id.index() + i);
            let v1 = VertexId::new(first_pulled_vertex_id.index() + (i + 1) % n);
            let oeid = mesh.faces[fid].oriented_edges[i];
            mesh.edges.push(Edge {
                vertices: [v0, v1],
                oriented_edges: [oeid, OrientedEdgeId::INVALID],
            });
            let oriented_edge = &mut mesh.oriented_edges[oeid];
            oriented_edge.edge = EdgeId::new(first_pulled_edge_id.index() + i);
            oriented_edge.forward = true;
        }
        // Add the skirt edges
        let first_skirt_edge_id = EdgeId::new(mesh.edges.len());
        for i in 0..n {
            // Skirt edges go top to bottom.
            mesh.edges.push(Edge {
                vertices: [
                    VertexId::new(first_pulled_vertex_id.index() + i),
                    wsp.vertices_of_face[i],
                ],
                oriented_edges: [OrientedEdgeId::INVALID, OrientedEdgeId::INVALID],
            });
        }
        // Add the skirt faces.
        let first_skirt_face_id = FaceId::new(mesh.faces.len());
        let mut orthogonal_face_and_skirt_face_edge_vec: Vec<(FaceId, FaceId, EdgeId)> = Vec::new();
        for i in 0..n {
            let skirt_face_id = FaceId::new(first_skirt_face_id.index() + i);
            let first_oedge_id = mesh.oriented_edges.len();
            let oriented_edges: Vec<OrientedEdgeId> = (first_oedge_id..first_oedge_id + 4)
                .map(OrientedEdgeId::new)
                .collect();
            // Top edge
            let top_edge_id = EdgeId::new(first_pulled_edge_id.index() + i);
            mesh.oriented_edges.push(OrientedEdge {
                edge: top_edge_id,
                forward: false,
                face: skirt_face_id,
            });
            mesh.edges[top_edge_id].oriented_edges[1] = oriented_edges[0];
            // Skirt edge from top to bottom.
            let downwards_skirt_edge_id = EdgeId::new(first_skirt_edge_id.index() + i);
            mesh.oriented_edges.push(OrientedEdge {
                edge: downwards_skirt_edge_id,
                forward: true,
                face: skirt_face_id,
            });
            mesh.edges[downwards_skirt_edge_id].oriented_edges[0] = oriented_edges[1];
            // Bottom edge, the original moved face edge.
            let bottom_edge_id = wsp.edges_of_face[i];
            let bottom_edge = &mesh.edges[bottom_edge_id];
            let face_oedge_id = mesh.faces[fid].oriented_edges[i];
            // Determine the direction, the orientation of the original oriented face edge.
            let forward = if bottom_edge.oriented_edges[0] == face_oedge_id {
                true
            } else {
                assert_eq!(bottom_edge.oriented_edges[1], face_oedge_id);
                false
            };
            // The new bottom oriented edge will have the same orientation.
            mesh.oriented_edges.push(OrientedEdge {
                edge: bottom_edge_id,
                forward,
                face: skirt_face_id,
            });
            let bottom_edge_oes = &mut mesh.edges[bottom_edge_id].oriented_edges;
            bottom_edge_oes[!forward as usize] = oriented_edges[2];
            let face_across_bottom_edge =
                mesh.oriented_edges[bottom_edge_oes[forward as usize]].face;
            // Store information if this is an orthogonal face which later needs to be merged with the skirt.
            if wsp
                .orthogonal_faces_sorted
                .binary_search(&face_across_bottom_edge)
                .is_ok()
            {
                orthogonal_face_and_skirt_face_edge_vec.push((
                    face_across_bottom_edge,
                    skirt_face_id,
                    bottom_edge_id,
                ));
            }
            // Skirt edge from bottom to top.
            let upwards_skirt_edge_id = EdgeId::new(first_skirt_edge_id.index() + (i + 1) % n);
            mesh.oriented_edges.push(OrientedEdge {
                edge: upwards_skirt_edge_id,
                forward: false,
                face: skirt_face_id,
            });
            mesh.edges[upwards_skirt_edge_id].oriented_edges[1] = oriented_edges[3];
            mesh.faces.push(Face {
                oriented_edges,
                normal: None,
            });
        }

        assert_validate_mesh(mesh);

        // Merge coplanar skirt faces into original orthogonal faces.
        let mut removed_edges: Vec<EdgeId> = Vec::new();
        let mut removed_oedges: Vec<OrientedEdgeId> = Vec::new();
        let mut removed_faces: Vec<FaceId> = Vec::new();
        for (ofid, skirtid, common_eid) in orthogonal_face_and_skirt_face_edge_vec {
            // Find the common edge's oriented edge in the orthogonal face.
            let common_edge_oes = &mesh.edges[common_eid].oriented_edges;
            removed_oedges.extend(common_edge_oes);
            removed_edges.push(common_eid);
            let f0 = mesh.oriented_edges[common_edge_oes[0]].face;
            let f1 = mesh.oriented_edges[common_edge_oes[1]].face;
            // Find which oriented edge id belongs to the orthogonal and skirt faces.
            let face_and_skirt_oe = if f0 == ofid {
                assert_eq!(f1, skirtid);
                (common_edge_oes[0], common_edge_oes[1])
            } else {
                assert_eq!(f0, skirtid);
                assert_eq!(f1, ofid);
                (common_edge_oes[1], common_edge_oes[0])
            };
            // Find face_and_skirt_oe.0 in the orthogonal face.
            let face_ofid = &mut mesh.faces[ofid];
            let pos_in_ofid = face_ofid
                .oriented_edges
                .iter()
                .position(|&oe| oe == face_and_skirt_oe.0)
                .unwrap(); // mesh validity ensures Some.
            // Rotate the oriented edges such that it ends with item at position.
            let num_oes = face_ofid.oriented_edges.len();
            face_ofid
                .oriented_edges
                .rotate_right(num_oes - pos_in_ofid - 1);
            // Remove common edge.
            assert_eq!(
                *face_ofid.oriented_edges.last().unwrap(),
                face_and_skirt_oe.0
            );
            face_ofid.oriented_edges.pop();
            // Continue these oriented edges with the skirt's oriented edges, starting from just
            // after face_and_skirt_oe.1 and stopping before it.
            let face_skirtid = &mut mesh.faces[skirtid];
            let pos_in_skirt = face_skirtid
                .oriented_edges
                .iter()
                .position(|&oe| oe == face_and_skirt_oe.1)
                .unwrap(); // mesh validity ensures Some.
            let num_skirt_edges = face_skirtid.oriented_edges.len();
            let mut oedges_to_copy: Vec<OrientedEdgeId> = Vec::with_capacity(num_skirt_edges - 1);
            for i in 1..num_skirt_edges {
                let oeid = mesh.faces[skirtid].oriented_edges[(pos_in_skirt + i) % num_skirt_edges];
                mesh.oriented_edges[oeid].face = ofid;
                oedges_to_copy.push(oeid);
            }
            mesh.faces[ofid].oriented_edges.extend(oedges_to_copy);
            removed_faces.push(skirtid);
        }

        removed_oedges.sort_unstable_by(|a, b| b.cmp(a));
        for oeid in removed_oedges {
            mesh.oriented_edges.swap_remove(oeid.index());
            // swap_remove reassigns index `mesh.oriented_edges.len() - 1` to oeid
            // It's referenced in its edge and face, update those.
            if oeid.index() != mesh.edges.len() {
                let oe = &mesh.oriented_edges[oeid];

                let swapped_oeid = OrientedEdgeId::new(mesh.oriented_edges.len());

                let edge = &mut mesh.edges[oe.edge];
                if edge.oriented_edges[0].index() == mesh.oriented_edges.len() {
                    edge.oriented_edges[0] = oeid;
                } else {
                    assert_eq!(edge.oriented_edges[1], swapped_oeid);
                    edge.oriented_edges[1] = oeid;
                }

                let face = &mut mesh.faces[oe.face];
                let slot = face
                    .oriented_edges
                    .iter()
                    .position(|&id| id == swapped_oeid)
                    .unwrap();
                face.oriented_edges[slot] = oeid;
            }
        }

        removed_edges.sort_unstable_by(|a, b| b.cmp(a));
        for eid in removed_edges {
            mesh.edges.swap_remove(eid.index());
            // swap_remove reassigns index `mesh.edges.len() - 1` to eid
            // It's referenced in its oriented edges, update those.
            if eid.index() != mesh.edges.len() {
                for i in [0, 1] {
                    mesh.oriented_edges[mesh.edges[eid].oriented_edges[i]].edge = eid;
                }
            }
        }

        removed_faces.sort_unstable_by(|a, b| b.cmp(a));
        for fid in removed_faces {
            mesh.faces.swap_remove(fid.index());
            // swap_remove reassigns index `mesh.faces.len() - 1` to fid
            // It's referenced in its oriented edges, update those.
            if fid.index() != mesh.faces.len() {
                for oeid in &mesh.faces[fid].oriented_edges {
                    mesh.oriented_edges[*oeid].face = fid;
                }
            }
        }
    }

    assert_validate_mesh(mesh);

    Ok(())
}

// Moves the face inwards with `-offset` which must be negative.
// Rejects if the moved faces has non-orthogonal adjacent faces.
fn push_face(mesh: &mut Mesh, fid: FaceId, offset: f64) -> Result<(), String> {
    assert!(offset < 0.0);

    let wsp = make_push_pull_workspace(mesh, fid)?;

    let farthest_offset = if wsp.all_faces_orthogonal {
        // Take all the edges of the adjacent, orthogonal faces. Filter for those whose faces are not both orthogonal, adjacent faces.
        // We achieve this by adding the edge on the first encounter and remove on the second.
        // First add the edges of the moved face.
        let mut farthest_offset_edges: HashSet<EdgeId> = HashSet::new();
        farthest_offset_edges.extend(wsp.edges_of_face);
        // Then add all the edges of the orthogonal faces.
        for ofid in &wsp.adjacent_faces {
            for eid in mesh.edges_of_face(*ofid) {
                if !farthest_offset_edges.insert(eid) {
                    farthest_offset_edges.remove(&eid);
                }
            }
        }
        // Collect vertices.
        let mut farthest_offset_vertices: HashSet<VertexId> = HashSet::new();
        for eid in farthest_offset_edges {
            farthest_offset_vertices.extend(&mesh.edges[eid].vertices);
        }

        // Find the vertex of the moved face that is most in the push direction. Theoretically, all vertices should be equally far.
        let mut lowest_vid = wsp.vertices_of_face[0];
        for vid in &wsp.vertices_of_face {
            if (mesh.vertices[*vid].position - mesh.vertices[lowest_vid].position)
                .dot(&wsp.face_normal)
                < 0.0
            {
                lowest_vid = *vid;
            }
        }

        // Find the vertex nearest to the moved face. That vertex defines the maximum (negative) offset.
        let lowest_position_in_face = mesh.vertices[lowest_vid].position;
        let mut farthest_offset: f64 = -f64::INFINITY;
        for vid in farthest_offset_vertices {
            farthest_offset = farthest_offset
                .max((mesh.vertices[vid].position - lowest_position_in_face).dot(&wsp.face_normal));
        }
        farthest_offset
    } else {
        0.0
    };

    if offset < farthest_offset {
        return Err(format!(
            "Offset {} is too far for face {}",
            offset,
            fid.index()
        ));
    }

    let vertex_offset = wsp.face_normal.into_inner() * offset;
    for &vid in &wsp.vertices_of_face {
        mesh.vertices[vid].position += vertex_offset;
    }

    Ok(())
}

pub fn push_pull_face(mesh: &mut Mesh, fid: FaceId, offset: f64) -> Result<(), String> {
    assert!(fid.index() < mesh.faces.len());
    if offset < 0.0 {
        return push_face(mesh, fid, offset);
    } else if offset > 0.0 {
        return pull_face(mesh, fid, offset);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::make_cube;
    // Rotate the vector such that the minimum element is at [0].
    fn rotate<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        if v.is_empty() {
            return v;
        }
        let min_pos = v.iter().enumerate().min_by_key(|(_, x)| *x).unwrap().0;
        v.rotate_left(min_pos);
        v
    }

    #[test]
    fn adjacent_faces_of_cube() {
        let m = make_cube();
        assert_eq!(
            rotate(m.adjacent_faces(FaceId::new(2))),
            vec![
                FaceId::new(0),
                FaceId::new(5),
                FaceId::new(1),
                FaceId::new(4)
            ]
        );
        assert_eq!(
            rotate(m.adjacent_faces(FaceId::new(3))),
            vec![
                FaceId::new(0),
                FaceId::new(4),
                FaceId::new(1),
                FaceId::new(5)
            ]
        );
        assert_eq!(
            rotate(m.adjacent_faces(FaceId::new(1))),
            vec![
                FaceId::new(2),
                FaceId::new(5),
                FaceId::new(3),
                FaceId::new(4)
            ]
        );
    }
    #[test]
    fn vertices_of_face() {
        let m = make_cube();
        assert_eq!(
            rotate(m.vertices_of_face(FaceId::new(2))),
            vec![
                VertexId::new(0),
                VertexId::new(1),
                VertexId::new(5),
                VertexId::new(4)
            ]
        );
    }
    #[test]
    fn edges_of_face() {
        let m = make_cube();
        assert_eq!(
            rotate(m.edges_of_face(FaceId::new(2))),
            vec![
                EdgeId::new(0),
                EdgeId::new(9),
                EdgeId::new(4),
                EdgeId::new(8)
            ]
        );
    }
}
