use std::env;
use std::path::{Path, PathBuf};

use image::{GrayImage, Luma};

#[derive(Debug, Clone)]
struct Args {
    width: usize,
    height: usize,
    block_size: usize,
    blocks: usize,
    seed: u64,
    visible_char: char,
    hidden_char: char,
    show_help: bool,
    output_path: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            width: 16,
            height: 16,
            block_size: 4,
            blocks: 4,
            seed: 42,
            visible_char: '.',
            hidden_char: '#',
            show_help: false,
            output_path: None,
        }
    }
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1))?;

    if args.show_help {
        print_help();
        return Ok(());
    }

    let mask = generate_mask(&args)?;

    if let Some(output_path) = &args.output_path {
        save_mask_png(&mask, args.width, args.height, output_path)?;
    } else {
        print_mask(&mask, args.width, args.visible_char, args.hidden_char);
    }

    Ok(())
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = Args::default();
    let mut args = args.into_iter();

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            parsed.show_help = true;
            continue;
        }

        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}\n\nRun with --help to see usage."))?;

        match flag.as_str() {
            "--width" => parsed.width = parse_usize(&flag, &value)?,
            "--height" => parsed.height = parse_usize(&flag, &value)?,
            "--block-size" => parsed.block_size = parse_usize(&flag, &value)?,
            "--blocks" => parsed.blocks = parse_usize(&flag, &value)?,
            "--seed" => parsed.seed = parse_u64(&flag, &value)?,
            "--visible-char" => parsed.visible_char = parse_char(&flag, &value)?,
            "--hidden-char" => parsed.hidden_char = parse_char(&flag, &value)?,
            "--output" => parsed.output_path = Some(resolve_output_path(&value)),
            _ => {
                return Err(format!(
                    "unknown argument: {flag}\n\nRun with --help to see usage."
                ))
            }
        }
    }

    validate_args(&parsed)?;
    Ok(parsed)
}

fn print_help() {
    println!(
        "Usage:
  mask-cli [options]

Options:
  --width <number>          Grid width, default: 16
  --height <number>         Grid height, default: 16
  --block-size <number>     Hidden block size, default: 4
  --blocks <number>         Number of hidden blocks, default: 4
  --seed <number>           Random seed, default: 42
  --visible-char <char>     Character for visible cells, default: .
  --hidden-char <char>      Character for hidden cells, default: #
  --output <path>           Write the mask as a PNG image in repo outputs/
  -h, --help                Show this help message"
    );
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("mask-cli should live under rust/mask-cli")
        .to_path_buf()
}

fn resolve_output_path(value: &str) -> PathBuf {
    let path = PathBuf::from(value);

    if path.is_absolute() {
        path
    } else {
        repo_root().join("outputs").join(path)
    }
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects a positive integer"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects a positive integer"))
}

fn parse_char(flag: &str, value: &str) -> Result<char, String> {
    let mut chars = value.chars();
    let character = chars
        .next()
        .ok_or_else(|| format!("{flag} expects one character"))?;

    if chars.next().is_some() {
        return Err(format!("{flag} expects one character"));
    }

    Ok(character)
}

fn validate_args(args: &Args) -> Result<(), String> {
    if args.width == 0 {
        return Err("--width must be greater than 0".to_string());
    }

    if args.height == 0 {
        return Err("--height must be greater than 0".to_string());
    }

    if args.block_size == 0 {
        return Err("--block-size must be greater than 0".to_string());
    }

    if args.block_size > args.width || args.block_size > args.height {
        return Err("--block-size must fit inside the grid".to_string());
    }

    Ok(())
}

fn generate_mask(args: &Args) -> Result<Vec<bool>, String> {
    let mut mask = vec![false; args.width * args.height];
    let mut seed = args.seed;
    let max_x = args.width - args.block_size + 1;
    let max_y = args.height - args.block_size + 1;

    for _ in 0..args.blocks {
        let x = next_random(&mut seed) as usize % max_x;
        let y = next_random(&mut seed) as usize % max_y;
        fill_block(&mut mask, args.width, x, y, args.block_size);
    }

    Ok(mask)
}

fn save_mask_png(
    mask: &[bool],
    width: usize,
    height: usize,
    output_path: &Path,
) -> Result<(), String> {
    let image_width = u32::try_from(width).map_err(|_| "--width is too large for PNG output")?;
    let image_height = u32::try_from(height).map_err(|_| "--height is too large for PNG output")?;
    let mut image = GrayImage::new(image_width, image_height);

    for y in 0..height {
        for x in 0..width {
            let hidden = mask[y * width + x];
            let value = if hidden { 0 } else { 255 };
            image.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create output directory: {error}"))?;
    }

    image
        .save(output_path)
        .map_err(|error| format!("failed to write PNG: {error}"))
}

fn next_random(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn fill_block(mask: &mut [bool], width: usize, start_x: usize, start_y: usize, block_size: usize) {
    for y in start_y..start_y + block_size {
        for x in start_x..start_x + block_size {
            mask[y * width + x] = true;
        }
    }
}

fn print_mask(mask: &[bool], width: usize, visible_char: char, hidden_char: char) {
    for row in mask.chunks(width) {
        let line: String = row
            .iter()
            .map(|hidden| if *hidden { hidden_char } else { visible_char })
            .collect();
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn generated_mask_has_expected_size() {
        let args = Args {
            width: 8,
            height: 6,
            block_size: 2,
            blocks: 3,
            seed: 1,
            visible_char: '.',
            hidden_char: '#',
            show_help: false,
            output_path: None,
        };

        let mask = generate_mask(&args).unwrap();

        assert_eq!(mask.len(), 48);
    }

    #[test]
    fn same_seed_gives_same_mask() {
        let args = Args {
            width: 8,
            height: 8,
            block_size: 2,
            blocks: 3,
            seed: 7,
            visible_char: '.',
            hidden_char: '#',
            show_help: false,
            output_path: None,
        };

        let first = generate_mask(&args).unwrap();
        let second = generate_mask(&args).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn help_flag_is_supported_without_value() {
        let args = parse_args(test_args(&["--help"])).unwrap();

        assert!(args.show_help);
    }

    #[test]
    fn unknown_argument_returns_error() {
        let error = parse_args(test_args(&["--wat", "12"])).unwrap_err();

        assert!(error.contains("unknown argument"));
    }

    #[test]
    fn missing_value_returns_error() {
        let error = parse_args(test_args(&["--width"])).unwrap_err();

        assert!(error.contains("missing value"));
    }

    #[test]
    fn visible_and_hidden_chars_can_be_changed() {
        let args = parse_args(test_args(&["--visible-char", "_", "--hidden-char", "X"])).unwrap();

        assert_eq!(args.visible_char, '_');
        assert_eq!(args.hidden_char, 'X');
    }

    #[test]
    fn char_options_must_have_one_character() {
        let error = parse_args(test_args(&["--visible-char", "ab"])).unwrap_err();

        assert!(error.contains("expects one character"));
    }

    #[test]
    fn output_path_can_be_set() {
        let args = parse_args(test_args(&["--output", "mask.png"])).unwrap();

        assert_eq!(args.output_path, Some(repo_root().join("outputs/mask.png")));
    }

    #[test]
    fn absolute_output_path_is_kept() {
        let output_path = repo_root().join("outputs/absolute-test.png");
        let args = parse_args(test_args(&["--output", output_path.to_str().unwrap()])).unwrap();

        assert_eq!(args.output_path, Some(output_path));
    }

    #[test]
    fn mask_can_be_saved_as_png() {
        let args = Args {
            width: 4,
            height: 4,
            block_size: 2,
            blocks: 1,
            seed: 1,
            visible_char: '.',
            hidden_char: '#',
            show_help: false,
            output_path: None,
        };
        let mask = generate_mask(&args).unwrap();
        let output_dir = PathBuf::from("target/test-output");
        std::fs::create_dir_all(&output_dir).unwrap();
        let output_path = output_dir.join("mask-cli-test-output.png");

        let _ = std::fs::remove_file(&output_path);
        save_mask_png(&mask, args.width, args.height, &output_path).unwrap();

        assert!(output_path.exists());

        let _ = std::fs::remove_file(output_path);
    }
}
