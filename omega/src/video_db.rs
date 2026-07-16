use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use image::{Rgb, RgbImage};
use rusttype::{Font, Scale, point};

fn find_font() -> Result<Vec<u8>, String> {
    let font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf"
    ];
    for path in &font_paths {
        if let Ok(bytes) = fs::read(path) {
            return Ok(bytes);
        }
    }
    Err("Could not find any TTF font on system".to_string())
}

fn wrap_text(text: &str, max_width_chars: usize) -> Vec<String> {
    let mut wrapped = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            wrapped.push("".to_string());
        } else {
            let mut current = String::new();
            for word in line.split_whitespace() {
                if current.is_empty() {
                    current.push_str(word);
                } else if current.len() + 1 + word.len() > max_width_chars {
                    wrapped.push(current);
                    current = word.to_string();
                } else {
                    current.push(' ');
                    current.push_str(word);
                }
            }
            if !current.is_empty() {
                wrapped.push(current);
            }
        }
    }
    wrapped
}

fn draw_line_text(
    img: &mut RgbImage,
    font: &Font,
    text: &str,
    x: i32,
    y: i32,
    scale: Scale,
    color: Rgb<u8>,
) {
    let glyphs: Vec<_> = font.layout(text, scale, point(x as f32, y as f32)).collect();
    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|gx, gy, v| {
                let px = bounding_box.min.x + gx as i32;
                let py = bounding_box.min.y + gy as i32;
                if px >= 0 && px < img.width() as i32 && py >= 0 && py < img.height() as i32 {
                    let pixel = img.get_pixel_mut(px as u32, py as u32);
                    let old_r = pixel[0] as f32;
                    let old_g = pixel[1] as f32;
                    let old_b = pixel[2] as f32;

                    let r = (old_r * (1.0 - v) + color[0] as f32 * v) as u8;
                    let g = (old_g * (1.0 - v) + color[1] as f32 * v) as u8;
                    let b = (old_b * (1.0 - v) + color[2] as f32 * v) as u8;
                    *pixel = Rgb([r, g, b]);
                }
            });
        }
    }
}

fn render_master_frame(pages_count: usize, font: &Font, path: &str) -> Result<(), String> {
    let width = 1024;
    let height = 768;
    let mut img = RgbImage::from_fn(width, height, |_, _| Rgb([15, 23, 42])); // dark slate

    // Title
    let scale_title = Scale::uniform(24.0);
    draw_line_text(&mut img, font, "🧠 OMEGA DRIVE - VEKG TOPOLOGY MAP (FRAME 0)", 30, 20, scale_title, Rgb([56, 189, 248]));

    // Divider line
    for px in 30..(width - 30) {
        img.put_pixel(px as u32, 60, Rgb([51, 65, 85]));
        img.put_pixel(px as u32, 61, Rgb([51, 65, 85]));
    }

    // Draw main master box
    let fill_color = Rgb([30, 41, 59]); // slate-800
    let border_color = Rgb([244, 63, 94]); // Rose
    let text_color = Rgb([226, 232, 240]);
    let desc_color = Rgb([148, 163, 184]);

    let scale_text = Scale::uniform(14.0);
    let scale_small = Scale::uniform(11.0);

    // Draw nodes list
    draw_line_text(&mut img, font, "📂 EXPOSED DATA SLAVE FRAMES:", 50, 100, scale_text, Rgb([16, 185, 129]));

    let mut y = 140;
    for i in 1..=pages_count {
        // Draw small preview box
        let box_x = 50;
        let box_y = y;
        let box_w = 400;
        let box_h = 35;

        for px in box_x..=(box_x + box_w) {
            for py in box_y..=(box_y + box_h) {
                let is_border = px == box_x || px == box_x + box_w || py == box_y || py == box_y + box_h;
                img.put_pixel(px as u32, py as u32, if is_border { border_color } else { fill_color });
            }
        }

        let label = format!("[SLAVE FRAME {}]", i);
        draw_line_text(&mut img, font, &label, box_x + 10, box_y + 10, scale_text, text_color);
        
        let link_label = format!("Index: {}s (Extract via VEXTRACT <key> {})", i, i);
        draw_line_text(&mut img, font, &link_label, box_x + 150, box_y + 12, scale_small, desc_color);

        y += 50;
        if y > 700 { break; }
    }

    img.save(path).map_err(|e| format!("Failed to save master image: {}", e))?;
    Ok(())
}

fn render_slave_frame(
    id: u32,
    content: &str,
    font: &Font,
    path: &str,
) -> Result<(), String> {
    let width = 1024;
    let height = 768;
    let mut img = RgbImage::from_fn(width, height, |_, _| Rgb([15, 23, 42]));

    // Header bg
    for px in 0..width {
        for py in 0..45 {
            img.put_pixel(px, py, Rgb([30, 41, 59]));
        }
        img.put_pixel(px, 45, Rgb([16, 185, 129])); // Emerald line
        img.put_pixel(px, 46, Rgb([16, 185, 129]));
    }

    let scale_header = Scale::uniform(16.0);
    let scale_code = Scale::uniform(12.0);

    let header_title = format!("📄 SLAVE FRAME {}", id);
    draw_line_text(&mut img, font, &header_title, 30, 12, scale_header, Rgb([226, 232, 240]));

    // Wrap lines
    let mut wrapped_lines = Vec::new();
    for line in content.lines() {
        if line.len() > 110 {
            wrapped_lines.extend(wrap_text(line, 110));
        } else {
            wrapped_lines.push(line.to_string());
        }
    }

    let mut y = 65;
    for line in wrapped_lines.iter().take(48) {
        draw_line_text(&mut img, font, line, 30, y, scale_code, Rgb([203, 213, 225]));
        y += 14;
    }

    // Footer
    for px in 0..width {
        for py in (height - 30)..height {
            img.put_pixel(px, py, Rgb([30, 41, 59]));
        }
    }
    draw_line_text(
        &mut img,
        font,
        "Topology-driven Video Database (VEKG) - Omega Engine Core",
        30,
        (height - 22) as i32,
        scale_code,
        Rgb([100, 116, 139]),
    );

    img.save(path).map_err(|e| format!("Failed to save slave image: {}", e))?;
    Ok(())
}

pub fn compile_video_from_pages(pages: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    let font_bytes = find_font()?;
    let font = Font::try_from_vec(font_bytes).ok_or("Failed to load font")?;

    // Create a temporary directory for rendering frames
    let temp_dir_name = format!("./temp_compile_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis());
    let temp_path = Path::new(&temp_dir_name);
    fs::create_dir_all(temp_path).map_err(|e| format!("Failed to create temp dir: {}", e))?;

    // 1. Render Master Frame (Frame 0)
    let master_path = format!("{}/frame_0.png", temp_dir_name);
    render_master_frame(pages.len(), &font, &master_path)?;

    // 2. Render Slave Frames (Frames 1..N)
    for (i, page_bytes) in pages.iter().enumerate() {
        let content = String::from_utf8_lossy(page_bytes);
        let slave_path = format!("{}/frame_{}.png", temp_dir_name, i + 1);
        render_slave_frame((i + 1) as u32, &content, &font, &slave_path)?;
    }

    // Duplicate last frame to avoid encoding hiccups on short inputs
    let last_frame_id = pages.len();
    fs::copy(
        format!("{}/frame_{}.png", temp_dir_name, last_frame_id),
        format!("{}/frame_{}.png", temp_dir_name, last_frame_id + 1),
    ).map_err(|e| format!("Failed to duplicate last frame: {}", e))?;

    // 3. Compile H.264 Video using FFmpeg
    let video_file = format!("{}/video.mp4", temp_dir_name);
    let output = Command::new("ffmpeg")
        .args(&[
            "-y",
            "-r", "1",
            "-f", "image2",
            "-i", &format!("{}/frame_%d.png", temp_dir_name),
            "-c:v", "libx264",
            "-g", "1",
            "-preset", "ultrafast",
            "-tune", "fastdecode",
            "-pix_fmt", "yuv420p",
            &video_file,
        ])
        .output()
        .map_err(|e| format!("Failed to spawn FFmpeg: {}", e))?;

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_dir_all(temp_path);
        return Err(format!("FFmpeg compilation failed: {}", err_msg));
    }

    // Read compiled video back
    let video_bytes = fs::read(&video_file).map_err(|e| format!("Failed to read video file: {}", e))?;

    // Clean up temp directory
    let _ = fs::remove_dir_all(temp_path);

    Ok(video_bytes)
}

pub fn extract_frame_from_bytes(video_bytes: &[u8], frame_id: u32) -> Result<Vec<u8>, String> {
    // Spawn FFmpeg to seek and extract a single frame from the in-memory video stream
    // -i pipe:0 (read video from stdin)
    // -ss <frame_id> (seek time in seconds)
    // -vframes 1 (output single frame)
    // -f image2pipe -vcodec png pipe:1 (write PNG to stdout)
    let mut child = Command::new("ffmpeg")
        .args(&[
            "-y",
            "-ss", &frame_id.to_string(),
            "-i", "pipe:0",
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null()) // Mute logs
        .spawn()
        .map_err(|e| format!("Failed to spawn FFmpeg extractor: {}", e))?;

    // Write input bytes to stdin in a separate scope to close stdin afterward
    {
        let mut stdin = child.stdin.take().ok_or("Failed to open stdin for FFmpeg")?;
        stdin.write_all(video_bytes).map_err(|e| format!("Failed to write to FFmpeg stdin: {}", e))?;
    }

    let output = child.wait_with_output().map_err(|e| format!("Failed to await FFmpeg: {}", e))?;
    if !output.status.success() {
        return Err("FFmpeg extraction failed".to_string());
    }

    Ok(output.stdout)
}

use std::path::Path;
