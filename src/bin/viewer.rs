use std::cell::RefCell;
use std::rc::Rc;

use kiss3d::egui;
use kiss3d::event::{Action, MouseButton, WindowEvent};
use kiss3d::prelude::*;

use brep::{make_cube, Edge, Mesh, OrientedEdge};

fn mesh_to_gpu(mesh: &Mesh) -> GpuMesh3d {
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();

    for face in &mesh.faces {
        let base = vertices.len() as u32;
        let face_verts: Vec<Vec3> = face
            .oriented_edges
            .iter()
            .map(|&oe_id| {
                let oe: &OrientedEdge = &mesh.oriented_edges[oe_id.0 as usize];
                let edge: &Edge = &mesh.edges[oe.edge.0 as usize];
                let vid = if oe.forward { edge.vertices[0] } else { edge.vertices[1] };
                let p = mesh.vertices[vid.0 as usize].position;
                Vec3::new(p.x as f32, p.y as f32, p.z as f32)
            })
            .collect();
        let n = face_verts.len() as u32;
        vertices.extend(face_verts);
        for i in 1..n - 1 {
            indices.push([base, base + i, base + i + 1]);
        }
    }

    // Center vertices so rotation is around the mesh centroid.
    if !vertices.is_empty() {
        let centroid = vertices.iter().copied().fold(Vec3::ZERO, |a, v| a + v)
            / vertices.len() as f32;
        for v in &mut vertices {
            *v -= centroid;
        }
    }

    GpuMesh3d::new(vertices, indices, None, None, false)
}

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("B-rep viewer").await;
    // Fixed camera at origin looking down -Z; object rotation drives the view angle.
    let mut camera = FixedView3d::new();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(5.0, 5.0, 5.0));

    let mut current: Option<SceneNode3d> = None;
    let mut last_cursor: Option<(f64, f64)> = None;

    while window.render_3d(&mut scene, &mut camera).await {
        let lmb = window.get_mouse_button(MouseButton::Button1) == Action::Press;
        let ui_wants_mouse = window.is_egui_capturing_mouse();

        for event in window.events().iter() {
            if let WindowEvent::CursorPos(x, y, _) = event.value {
                if lmb && !ui_wants_mouse {
                    if let Some((lx, ly)) = last_cursor {
                        let dx = (x - lx) as f32 * 0.01;
                        let dy = (y - ly) as f32 * 0.01;
                        if let Some(ref mut node) = current {
                            // rotate() pre-multiplies, so axes stay fixed in world space.
                            node.rotate(Quat::from_axis_angle(Vec3::Y, dx));
                            node.rotate(Quat::from_axis_angle(Vec3::X, dy));
                        }
                    }
                    last_cursor = Some((x, y));
                } else {
                    last_cursor = None;
                }
            }
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Primitives").show(ctx, |ui| {
                if ui.button("cube").clicked() {
                    if let Some(mut node) = current.take() {
                        node.remove();
                    }
                    let gpu_mesh = Rc::new(RefCell::new(mesh_to_gpu(&make_cube())));
                    current = Some(
                        scene
                            .add_mesh(gpu_mesh, Vec3::ONE)
                            .set_color(Color::new(0.6, 0.8, 1.0, 1.0))
                            .set_position(Vec3::new(0.0, 0.0, -3.0))
                            .enable_backface_culling(false),
                    );
                }
            });
        });
    }
}
