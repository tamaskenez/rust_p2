use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;

use kiss3d::egui;
use kiss3d::event::{Action, MouseButton, WindowEvent};
use kiss3d::prelude::*;

use brep::{make_cube, Edge, FaceId, Mesh, OrientedEdge};

#[derive(PartialEq, Eq)]
enum Tool {
    Rotate,
    PushPull,
}

struct PushPullOp {
    mesh_before: Mesh,
    offset: f64,
    error: Option<String>,
    scale: f64,
}

const OBJECT_Z: f32 = -3.0;
const DRAG_THRESHOLD: f64 = 5.0;
const FOV_Y: f32 = PI / 4.0;

fn face_gpu_verts(mesh: &Mesh, face_idx: usize) -> Vec<Vec3> {
    let face = &mesh.faces[face_idx];
    face.oriented_edges
        .iter()
        .map(|&oe_id| {
            let oe: &OrientedEdge = &mesh.oriented_edges[oe_id.0 as usize];
            let edge: &Edge = &mesh.edges[oe.edge.0 as usize];
            let vid = if oe.forward { edge.vertices[0] } else { edge.vertices[1] };
            let p = mesh.vertices[vid.0 as usize].position;
            Vec3::new(p.x as f32, p.y as f32, p.z as f32)
        })
        .collect()
}

fn compute_centroid(mesh: &Mesh) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for i in 0..mesh.faces.len() {
        for v in face_gpu_verts(mesh, i) {
            sum += v;
            count += 1;
        }
    }
    if count > 0 { sum / count as f32 } else { Vec3::ZERO }
}

fn face_center_world(mesh: &Mesh, face_idx: usize, centroid: Vec3, rotation: Quat) -> Vec3 {
    let verts = face_gpu_verts(mesh, face_idx);
    let raw_center = verts.iter().copied().fold(Vec3::ZERO, |a, v| a + v) / verts.len() as f32;
    rotation * (raw_center - centroid) + Vec3::new(0.0, 0.0, OBJECT_Z)
}

// Returns the world-space distance that corresponds to 1 screen pixel at the depth of
// `center_world`. Computed by projecting center, shifting 1 pixel right on screen,
// un-projecting back onto the plane z = center_world.z, and measuring the gap.
fn pixels_to_world(center_world: Vec3, win_w: f32, win_h: f32) -> f32 {
    let tan_hfov = (FOV_Y / 2.0).tan();
    let aspect = win_w / win_h;
    let depth = -center_world.z;
    if depth < 1e-7 {
        return 1.0;
    }

    // Project center to screen (pixels)
    let ndcx_c = center_world.x / depth / (aspect * tan_hfov);
    let ndcy_c = center_world.y / depth / tan_hfov;
    let center_on_screen = Vec2::new(
        (ndcx_c + 1.0) / 2.0 * win_w,
        (1.0 - ndcy_c) / 2.0 * win_h,
    );

    // Shift 1 pixel to the right
    let next_to_center_on_screen = center_on_screen + Vec2::X;

    // Un-project next_to_center_on_screen to a world-space ray
    let ndcx_n = 2.0 * next_to_center_on_screen.x / win_w - 1.0;
    let ndcy_n = 1.0 - 2.0 * next_to_center_on_screen.y / win_h;
    let rd = Vec3::new(ndcx_n * aspect * tan_hfov, ndcy_n * tan_hfov, -1.0).normalize();

    // Intersect the ray with the plane z = center_world.z (normal = view axis)
    // Ray: P(t) = eye + t*rd = t*rd (eye at origin); plane: P.z = center_world.z
    let t = center_world.z / rd.z;
    let next_to_center = rd * t;

    // world distance / screen distance (= 1 pixel by construction)
    (center_world - next_to_center).length()
        / (center_on_screen - next_to_center_on_screen).length()
}

fn build_gpu_mesh(mesh: &Mesh, centroid: Vec3, face_mask: impl Fn(usize) -> bool) -> GpuMesh3d {
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    for face_idx in 0..mesh.faces.len() {
        if !face_mask(face_idx) {
            continue;
        }
        let base = vertices.len() as u32;
        let face_verts: Vec<Vec3> = face_gpu_verts(mesh, face_idx)
            .into_iter()
            .map(|v| v - centroid)
            .collect();
        let n = face_verts.len() as u32;
        vertices.extend(face_verts);
        for i in 1..n - 1 {
            indices.push([base, base + i, base + i + 1]);
        }
    }
    GpuMesh3d::new(vertices, indices, None, None, false)
}

fn spawn_node(
    scene: &mut SceneNode3d,
    gpu: GpuMesh3d,
    color: Color,
    rotation: Quat,
) -> SceneNode3d {
    let mut node = scene
        .add_mesh(Rc::new(RefCell::new(gpu)), Vec3::ONE)
        .set_color(color)
        .set_position(Vec3::new(0.0, 0.0, OBJECT_Z))
        .enable_backface_culling(false);
    node.set_rotation(rotation);
    node
}

fn rebuild_scene(
    scene: &mut SceneNode3d,
    main_node: &mut Option<SceneNode3d>,
    sel_node: &mut Option<SceneNode3d>,
    mesh: &Mesh,
    centroid: Vec3,
    selected_face: Option<FaceId>,
    rotation: Quat,
) {
    if let Some(mut n) = main_node.take() {
        n.remove();
    }
    if let Some(mut n) = sel_node.take() {
        n.remove();
    }

    let sel_idx = selected_face.map(|f| f.0 as usize);

    *main_node = Some(spawn_node(
        scene,
        build_gpu_mesh(mesh, centroid, |i| Some(i) != sel_idx),
        Color::new(0.6, 0.8, 1.0, 1.0),
        rotation,
    ));

    if let Some(sel) = selected_face {
        let si = sel.0 as usize;
        *sel_node = Some(spawn_node(
            scene,
            build_gpu_mesh(mesh, centroid, |i| i == si),
            Color::new(1.0, 1.0, 0.0, 1.0),
            rotation,
        ));
    }
}

fn ray_triangle_intersect(ro: Vec3, rd: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = rd.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-7 {
        return None;
    }
    let f = 1.0 / a;
    let s = ro - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * rd.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t > 1e-7 { Some(t) } else { None }
}

fn pick_face(
    mesh: &Mesh,
    centroid: Vec3,
    rotation: Quat,
    cx: f64,
    cy: f64,
    win_w: f32,
    win_h: f32,
) -> Option<FaceId> {
    let nx = (2.0 * cx as f32 / win_w) - 1.0;
    let ny = 1.0 - (2.0 * cy as f32 / win_h);
    let tan_hfov = (FOV_Y / 2.0).tan();
    let aspect = win_w / win_h;
    let rd_world = Vec3::new(nx * aspect * tan_hfov, ny * tan_hfov, -1.0).normalize();

    let inv_rot = rotation.conjugate();
    let ro_local = inv_rot * -Vec3::new(0.0, 0.0, OBJECT_Z);
    let rd_local = inv_rot * rd_world;

    let mut best_t = f32::INFINITY;
    let mut best_face: Option<FaceId> = None;

    for face_idx in 0..mesh.faces.len() {
        let verts: Vec<Vec3> = face_gpu_verts(mesh, face_idx)
            .into_iter()
            .map(|v| v - centroid)
            .collect();
        let n = verts.len();
        for i in 1..n - 1 {
            if let Some(t) =
                ray_triangle_intersect(ro_local, rd_local, verts[0], verts[i], verts[i + 1])
            {
                if t < best_t {
                    best_t = t;
                    best_face = Some(FaceId(face_idx as u32));
                }
            }
        }
    }

    best_face
}

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("B-rep viewer").await;
    let mut camera = FixedView3d::new();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(5.0, 5.0, 5.0));

    let mut current_main: Option<SceneNode3d> = None;
    let mut current_sel: Option<SceneNode3d> = None;
    let mut mesh: Option<Mesh> = None;
    let mut centroid = Vec3::ZERO;
    let mut rotation = Quat::IDENTITY;
    let mut selected_face: Option<FaceId> = None;
    let mut active_tool = Tool::Rotate;
    let mut current_op: Option<PushPullOp> = None;
    let mut last_cursor: Option<(f64, f64)> = None;
    let mut cursor_pos = (0.0f64, 0.0f64);
    let mut press_pos: Option<(f64, f64)> = None;

    while window.render_3d(&mut scene, &mut camera).await {
        let lmb = window.get_mouse_button(MouseButton::Button1) == Action::Press;
        let ui_wants = window.is_egui_capturing_mouse();
        let win_w = window.width() as f32;
        let win_h = window.height() as f32;

        for event in window.events().iter() {
            match event.value {
                WindowEvent::CursorPos(x, y, _) => {
                    cursor_pos = (x, y);
                    if lmb && !ui_wants {
                        match active_tool {
                            Tool::Rotate => {
                                if let Some((lx, ly)) = last_cursor {
                                    let dx = (x - lx) as f32 * 0.01;
                                    let dy = (y - ly) as f32 * 0.01;
                                    let qy = Quat::from_axis_angle(Vec3::Y, dx);
                                    let qx = Quat::from_axis_angle(Vec3::X, dy);
                                    for opt in [&mut current_main, &mut current_sel] {
                                        if let Some(node) = opt {
                                            node.rotate(qy);
                                            node.rotate(qx);
                                        }
                                    }
                                    rotation = qy * rotation;
                                    rotation = qx * rotation;
                                }
                                last_cursor = Some((x, y));
                            }
                            Tool::PushPull => {
                                if let (Some(op), Some(pp)) = (&mut current_op, press_pos) {
                                    op.offset = (pp.1 - y) * op.scale;
                                    println!("push/pull offset: {:.4}", op.offset);
                                }
                            }
                        }
                    } else {
                        last_cursor = None;
                    }
                }
                WindowEvent::MouseButton(MouseButton::Button1, Action::Press, _) => {
                    if !ui_wants {
                        press_pos = Some(cursor_pos);
                        if active_tool == Tool::PushPull {
                            if selected_face.is_none() {
                                if let Some(m) = &mesh {
                                    let picked = pick_face(
                                        m, centroid, rotation,
                                        cursor_pos.0, cursor_pos.1,
                                        win_w, win_h,
                                    );
                                    if picked.is_some() {
                                        selected_face = picked;
                                        rebuild_scene(
                                            &mut scene,
                                            &mut current_main,
                                            &mut current_sel,
                                            m,
                                            centroid,
                                            selected_face,
                                            rotation,
                                        );
                                    }
                                }
                            }
                            if let (Some(m), Some(sel)) = (&mesh, selected_face) {
                                let center = face_center_world(m, sel.0 as usize, centroid, rotation);
                                let scale = pixels_to_world(center, win_w, win_h) as f64;
                                current_op = Some(PushPullOp {
                                    mesh_before: m.clone(),
                                    offset: 0.0,
                                    error: None,
                                    scale,
                                });
                            }
                        }
                    }
                }
                WindowEvent::MouseButton(MouseButton::Button1, Action::Release, _) => {
                    if !ui_wants {
                        match active_tool {
                            Tool::Rotate => {
                                if let (Some(pp), Some(m)) = (press_pos.take(), &mesh) {
                                    let dx = cursor_pos.0 - pp.0;
                                    let dy = cursor_pos.1 - pp.1;
                                    if (dx * dx + dy * dy).sqrt() < DRAG_THRESHOLD {
                                        let new_sel = pick_face(
                                            m, centroid, rotation,
                                            cursor_pos.0, cursor_pos.1,
                                            win_w, win_h,
                                        );
                                        if new_sel != selected_face {
                                            selected_face = new_sel;
                                            rebuild_scene(
                                                &mut scene,
                                                &mut current_main,
                                                &mut current_sel,
                                                m,
                                                centroid,
                                                selected_face,
                                                rotation,
                                            );
                                        }
                                    }
                                }
                            }
                            Tool::PushPull => {
                                let was_click = press_pos.map_or(false, |pp| {
                                    let dx = cursor_pos.0 - pp.0;
                                    let dy = cursor_pos.1 - pp.1;
                                    (dx * dx + dy * dy).sqrt() < DRAG_THRESHOLD
                                });
                                press_pos = None;
                                current_op = None;
                                if was_click {
                                    if let Some(m) = &mesh {
                                        let new_sel = pick_face(
                                            m, centroid, rotation,
                                            cursor_pos.0, cursor_pos.1,
                                            win_w, win_h,
                                        );
                                        if new_sel != selected_face {
                                            selected_face = new_sel;
                                            rebuild_scene(
                                                &mut scene,
                                                &mut current_main,
                                                &mut current_sel,
                                                m,
                                                centroid,
                                                selected_face,
                                                rotation,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Toolbar").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut active_tool, Tool::Rotate, "rotate");
                    ui.selectable_value(&mut active_tool, Tool::PushPull, "push/pull");
                });
            });

            egui::Window::new("Primitives").show(ctx, |ui| {
                if ui.button("cube").clicked() {
                    if let Some(mut n) = current_main.take() {
                        n.remove();
                    }
                    if let Some(mut n) = current_sel.take() {
                        n.remove();
                    }
                    selected_face = None;
                    rotation = Quat::IDENTITY;

                    let cube = make_cube();
                    centroid = compute_centroid(&cube);
                    let gpu = build_gpu_mesh(&cube, centroid, |_| true);
                    current_main = Some(spawn_node(
                        &mut scene,
                        gpu,
                        Color::new(0.6, 0.8, 1.0, 1.0),
                        Quat::IDENTITY,
                    ));
                    mesh = Some(cube);
                }
            });
        });
    }
}
