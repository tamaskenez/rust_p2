use std::collections::HashMap;
use std::io;
use std::path::Path;

use nalgebra::Point3;

use crate::{Edge, EdgeId, Face, FaceId, Mesh, OrientedEdge, OrientedEdgeId, Vertex, VertexId};

type StepId = u32;
type EntityMap = HashMap<StepId, (String, Vec<Param>)>;

#[derive(Debug)]
pub enum StepError {
    Io(io::Error),
    Parse(String),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::Io(e) => write!(f, "I/O error: {e}"),
            StepError::Parse(s) => write!(f, "parse error: {s}"),
        }
    }
}

impl std::error::Error for StepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StepError::Io(e) => Some(e),
            StepError::Parse(_) => None,
        }
    }
}

impl From<io::Error> for StepError {
    fn from(e: io::Error) -> Self {
        StepError::Io(e)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Str is parsed but intentionally ignored during mesh building
enum Param {
    Omitted,
    Ref(StepId),
    Enum(String),
    Str(String),
    Real(f64),
    List(Vec<Param>),
}

// ── Parser ────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { input, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.input[self.pos..].chars().next()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.advance();
        }
    }

    fn parse_param(&mut self) -> Result<Param, StepError> {
        self.skip_ws();
        match self.peek() {
            Some('*') => {
                self.advance();
                Ok(Param::Omitted)
            }
            Some('#') => {
                self.advance();
                let start = self.pos;
                while matches!(self.peek(), Some('0'..='9')) {
                    self.advance();
                }
                self.input[start..self.pos]
                    .parse::<StepId>()
                    .map(Param::Ref)
                    .map_err(|_| StepError::Parse("bad entity ref".into()))
            }
            Some('.') => {
                self.advance();
                let start = self.pos;
                while !matches!(self.peek(), Some('.') | None) {
                    self.advance();
                }
                let name = self.input[start..self.pos].to_string();
                self.advance(); // closing '.'
                Ok(Param::Enum(name))
            }
            Some('\'') => {
                self.advance();
                let mut s = String::new();
                loop {
                    match self.advance() {
                        None => return Err(StepError::Parse("unterminated string".into())),
                        Some('\'') => {
                            if self.peek() == Some('\'') {
                                self.advance();
                                s.push('\'');
                            } else {
                                break;
                            }
                        }
                        Some(c) => s.push(c),
                    }
                }
                Ok(Param::Str(s))
            }
            Some('(') => {
                self.advance();
                let mut items = Vec::new();
                self.skip_ws();
                if self.peek() != Some(')') {
                    items.push(self.parse_param()?);
                    self.skip_ws();
                    while self.peek() == Some(',') {
                        self.advance();
                        items.push(self.parse_param()?);
                        self.skip_ws();
                    }
                }
                self.skip_ws();
                match self.advance() {
                    Some(')') => Ok(Param::List(items)),
                    _ => Err(StepError::Parse("expected ')'".into())),
                }
            }
            Some(c) if c == '-' || c.is_ascii_digit() => {
                let start = self.pos;
                if c == '-' {
                    self.advance();
                }
                while matches!(self.peek(), Some('0'..='9')) {
                    self.advance();
                }
                if self.peek() == Some('.') {
                    self.advance();
                    while matches!(self.peek(), Some('0'..='9')) {
                        self.advance();
                    }
                }
                if matches!(self.peek(), Some('E' | 'e')) {
                    self.advance();
                    if matches!(self.peek(), Some('+' | '-')) {
                        self.advance();
                    }
                    while matches!(self.peek(), Some('0'..='9')) {
                        self.advance();
                    }
                }
                let tok = &self.input[start..self.pos];
                tok.parse::<f64>()
                    .map(Param::Real)
                    .map_err(|_| StepError::Parse(format!("bad number: {tok}")))
            }
            other => Err(StepError::Parse(format!("unexpected token: {other:?}"))),
        }
    }

    fn parse_param_list(&mut self) -> Result<Vec<Param>, StepError> {
        let mut params = Vec::new();
        self.skip_ws();
        if self.pos >= self.input.len() {
            return Ok(params);
        }
        params.push(self.parse_param()?);
        self.skip_ws();
        while self.peek() == Some(',') {
            self.advance();
            params.push(self.parse_param()?);
            self.skip_ws();
        }
        Ok(params)
    }
}

// ── File-level parsing ────────────────────────────────────────────────────────

fn parse_entity_line(line: &str) -> Result<(StepId, String, Vec<Param>), StepError> {
    let line = line.trim().trim_end_matches(';');
    let eq = line
        .find('=')
        .ok_or_else(|| StepError::Parse(format!("no '=' in '{line}'")))?;
    let id: StepId = line[..eq]
        .trim()
        .trim_start_matches('#')
        .parse()
        .map_err(|_| StepError::Parse("bad entity id".into()))?;
    let rest = line[eq + 1..].trim();
    let paren = rest
        .find('(')
        .ok_or_else(|| StepError::Parse("no '(' in entity".into()))?;
    let entity_type = rest[..paren].trim().to_uppercase();
    let inner = &rest[paren + 1..];

    // find the matching closing parenthesis
    let mut depth = 1usize;
    let mut end = inner.len();
    for (i, c) in inner.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }

    let params = Parser::new(&inner[..end]).parse_param_list()?;
    Ok((id, entity_type, params))
}

fn parse_step_data(content: &str) -> Result<EntityMap, StepError> {
    let mut entities = HashMap::new();
    let mut in_data = false;
    let mut stmt = String::new();

    for raw_line in content.lines() {
        // strip /* ... */ comments (single-line only; sufficient for DATA section)
        let line = match raw_line.find("/*") {
            Some(i) => raw_line[..i].trim(),
            None => raw_line.trim(),
        };
        match line {
            "DATA;" => {
                in_data = true;
                continue;
            }
            "ENDSEC;" => {
                in_data = false;
                continue;
            }
            _ => {}
        }
        if !in_data || line.is_empty() {
            continue;
        }
        stmt.push_str(line);
        if stmt.ends_with(';') {
            if stmt.starts_with('#') {
                let (id, typ, params) = parse_entity_line(&stmt)?;
                entities.insert(id, (typ, params));
            }
            stmt.clear();
        } else {
            stmt.push(' ');
        }
    }
    Ok(entities)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn get_ref(p: Option<&Param>) -> Result<StepId, StepError> {
    match p {
        Some(Param::Ref(id)) => Ok(*id),
        other => Err(StepError::Parse(format!("expected ref, got {other:?}"))),
    }
}

fn get_real(p: Option<&Param>) -> Result<f64, StepError> {
    match p {
        Some(Param::Real(v)) => Ok(*v),
        other => Err(StepError::Parse(format!("expected real, got {other:?}"))),
    }
}

fn get_bool(p: Option<&Param>) -> Result<bool, StepError> {
    match p {
        Some(Param::Enum(s)) if s == "T" => Ok(true),
        Some(Param::Enum(s)) if s == "F" => Ok(false),
        other => Err(StepError::Parse(format!(
            "expected bool enum, got {other:?}"
        ))),
    }
}

fn get_list(p: Option<&Param>) -> Result<&Vec<Param>, StepError> {
    match p {
        Some(Param::List(v)) => Ok(v),
        other => Err(StepError::Parse(format!("expected list, got {other:?}"))),
    }
}

// ── Mesh builder ──────────────────────────────────────────────────────────────

fn build_mesh(entities: &EntityMap) -> Result<Mesh, StepError> {
    let mut mesh = Mesh {
        vertices: Vec::new(),
        edges: Vec::new(),
        oriented_edges: Vec::new(),
        faces: Vec::new(),
    };

    // CARTESIAN_POINT('', (x, y, z))
    let mut cartesian_points: HashMap<StepId, Point3<f64>> = HashMap::new();
    for (&id, (typ, params)) in entities {
        if typ == "CARTESIAN_POINT" {
            let coords = get_list(params.get(1))?;
            cartesian_points.insert(
                id,
                Point3::new(
                    get_real(coords.get(0))?,
                    get_real(coords.get(1))?,
                    get_real(coords.get(2))?,
                ),
            );
        }
    }

    // VERTEX_POINT('', #cartesian_point)
    let mut vertex_map: HashMap<StepId, VertexId> = HashMap::new();
    let mut vp_ids: Vec<StepId> = entities
        .iter()
        .filter(|(_, (t, _))| t == "VERTEX_POINT")
        .map(|(&id, _)| id)
        .collect();
    vp_ids.sort_unstable();
    for step_id in vp_ids {
        let (_, params) = &entities[&step_id];
        let cp_id = get_ref(params.get(1))?;
        let pos = cartesian_points
            .get(&cp_id)
            .copied()
            .ok_or_else(|| StepError::Parse(format!("missing CARTESIAN_POINT #{cp_id}")))?;
        let vid = VertexId::new(mesh.vertices.len());
        mesh.vertices.push(Vertex { position: pos });
        vertex_map.insert(step_id, vid);
    }

    // EDGE_CURVE('', #vertex_start, #vertex_end, #curve, .T.)
    // oriented_edges back-refs filled when ORIENTED_EDGEs are processed
    let mut edge_map: HashMap<StepId, EdgeId> = HashMap::new();
    let mut ec_ids: Vec<StepId> = entities
        .iter()
        .filter(|(_, (t, _))| t == "EDGE_CURVE")
        .map(|(&id, _)| id)
        .collect();
    ec_ids.sort_unstable();
    for step_id in ec_ids {
        let (_, params) = &entities[&step_id];
        let vs = get_ref(params.get(1))?;
        let ve = get_ref(params.get(2))?;
        let v0 = *vertex_map
            .get(&vs)
            .ok_or_else(|| StepError::Parse(format!("missing VERTEX_POINT #{vs}")))?;
        let v1 = *vertex_map
            .get(&ve)
            .ok_or_else(|| StepError::Parse(format!("missing VERTEX_POINT #{ve}")))?;
        let eid = EdgeId::new(mesh.edges.len());
        mesh.edges.push(Edge {
            vertices: [v0, v1],
            oriented_edges: [OrientedEdgeId::INVALID, OrientedEdgeId::INVALID],
        });
        edge_map.insert(step_id, eid);
    }

    // ORIENTED_EDGE('', *, *, #edge_curve, .T./.F.)
    // face back-ref filled when ADVANCED_FACEs are processed
    // slot 0 = forward, slot 1 = reversed
    let mut oe_map: HashMap<StepId, OrientedEdgeId> = HashMap::new();
    let mut oe_ids: Vec<StepId> = entities
        .iter()
        .filter(|(_, (t, _))| t == "ORIENTED_EDGE")
        .map(|(&id, _)| id)
        .collect();
    oe_ids.sort_unstable();
    for step_id in oe_ids {
        let (_, params) = &entities[&step_id];
        let ec_step = get_ref(params.get(3))?;
        let forward = get_bool(params.get(4))?;
        let eid = *edge_map
            .get(&ec_step)
            .ok_or_else(|| StepError::Parse(format!("missing EDGE_CURVE #{ec_step}")))?;
        let oe_id = OrientedEdgeId::new(mesh.oriented_edges.len());
        mesh.oriented_edges.push(OrientedEdge {
            edge: eid,
            forward,
            face: FaceId::INVALID,
        });
        oe_map.insert(step_id, oe_id);
        mesh.edges[eid].oriented_edges[if forward { 0 } else { 1 }] = oe_id;
    }

    // EDGE_LOOP('', (#oe1, #oe2, ...))
    let mut loop_map: HashMap<StepId, Vec<OrientedEdgeId>> = HashMap::new();
    for (&id, (typ, params)) in entities {
        if typ == "EDGE_LOOP" {
            let list = get_list(params.get(1))?;
            let oes: Result<Vec<OrientedEdgeId>, _> =
                list.iter()
                    .map(|p| {
                        let sid = get_ref(Some(p))?;
                        oe_map.get(&sid).copied().ok_or_else(|| {
                            StepError::Parse(format!("missing ORIENTED_EDGE #{sid}"))
                        })
                    })
                    .collect();
            loop_map.insert(id, oes?);
        }
    }

    // FACE_BOUND / FACE_OUTER_BOUND('', #edge_loop, .T./.F.)
    let mut bound_map: HashMap<StepId, StepId> = HashMap::new();
    for (&id, (typ, params)) in entities {
        if typ == "FACE_BOUND" || typ == "FACE_OUTER_BOUND" {
            bound_map.insert(id, get_ref(params.get(1))?);
        }
    }

    // ADVANCED_FACE('', (#bound, ...), #surface, .T./.F.)
    let mut af_ids: Vec<StepId> = entities
        .iter()
        .filter(|(_, (t, _))| t == "ADVANCED_FACE")
        .map(|(&id, _)| id)
        .collect();
    af_ids.sort_unstable();
    for step_id in af_ids {
        // Collect bound refs in a separate block to release the borrow on entities.
        let bound_ids: Vec<StepId> = {
            let (_, params) = &entities[&step_id];
            get_list(params.get(1))?
                .iter()
                .map(|p| get_ref(Some(p)))
                .collect::<Result<_, _>>()?
        };

        // Prefer FACE_OUTER_BOUND; fall back to the first bound.
        let loop_step = bound_ids
            .iter()
            .find(|&&bid| {
                entities
                    .get(&bid)
                    .map_or(false, |(t, _)| t == "FACE_OUTER_BOUND")
            })
            .or_else(|| bound_ids.first())
            .and_then(|bid| bound_map.get(bid).copied())
            .ok_or_else(|| {
                StepError::Parse(format!("ADVANCED_FACE #{step_id} has no usable bound"))
            })?;

        let oe_list = loop_map
            .get(&loop_step)
            .cloned()
            .ok_or_else(|| StepError::Parse(format!("missing EDGE_LOOP #{loop_step}")))?;

        let fid = FaceId::new(mesh.faces.len());
        for &oe_id in &oe_list {
            mesh.oriented_edges[oe_id].face = fid;
        }
        mesh.faces.push(Face {
            oriented_edges: oe_list,
            normal: None,
        });
    }

    // Validate that every oriented edge was claimed by a face.
    for (i, oe) in mesh.oriented_edges.iter().enumerate() {
        if oe.face == FaceId::INVALID {
            return Err(StepError::Parse(format!(
                "oriented_edge {i} was not assigned to any face"
            )));
        }
    }
    // Validate that every edge has both oriented-edge back-refs filled.
    for (i, edge) in mesh.edges.iter().enumerate() {
        if edge.oriented_edges[0] == OrientedEdgeId::INVALID
            || edge.oriented_edges[1] == OrientedEdgeId::INVALID
        {
            return Err(StepError::Parse(format!(
                "edge {i} is missing a forward or reversed oriented_edge back-ref"
            )));
        }
    }

    Ok(mesh)
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn load_step(path: &Path) -> Result<Mesh, StepError> {
    let content = std::fs::read_to_string(path)?;
    let entities = parse_step_data(&content)?;
    build_mesh(&entities)
}
