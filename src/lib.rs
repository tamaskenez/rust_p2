use nalgebra::Point3;
use nalgebra::Unit;
use nalgebra::Vector3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrientedEdgeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceId(pub u32);

pub struct Vertex {
    pub position: Point3<f64>,
}

/// Canonical direction is vertices[0] → vertices[1].
/// oriented_edges[0] is the forward wrapper, oriented_edges[1] is the reversed wrapper.
pub struct Edge {
    pub vertices: [VertexId; 2],
    pub oriented_edges: [OrientedEdgeId; 2],
}

pub struct OrientedEdge {
    pub edge: EdgeId,
    pub forward: bool,
    pub face: FaceId,
}

pub struct Face {
    pub oriented_edges: Vec<OrientedEdgeId>,
    pub normal: Option<Unit<Vector3<f64>>>,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub oriented_edges: Vec<OrientedEdge>,
    pub faces: Vec<Face>,
}
