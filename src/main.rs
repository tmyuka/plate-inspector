use anyhow::{anyhow, Result};
use colored::Colorize;
use getopts::Options;
use glob::glob;
use image::imageops::{resize, FilterType};
use image::{GenericImage, GrayImage, Luma};
use imageproc::contrast::stretch_contrast;
use imageproc::drawing::{draw_line_segment_mut, draw_text_mut};
use imageproc::stats::histogram;
use regex::Regex;
use rusttype::{Font, Scale};
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt::{self, Display};
use std::path::Path;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
const BUILDTIME: &str = env!("BUILD_TIMESTAMP");

// Commandline arguments
#[derive(Debug)]
struct AppConfig {
    // the directory with the images
    plate_dir: String,

    // the name of the output file
    output_file: String,

    // the microscope channel to analyze (1 ..= 5)
    channel: u8,

    // the image configuration
    // (read from file or default)
    image_config: ImageConfig,
}

// These can be read from a config file
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ImageConfig {
    image_regex: String,
    target_size_x: u32,
    target_size_y: u32,
    num_sites_x: u32,
    num_sites_y: u32,
    intensity_cutoff_low: u8,
    intensity_cutoff_high: u8,
    intensity_addition: u8,
    num_rows: u32,
    num_columns: u32
    }

impl Default for ImageConfig {
    fn default() -> Self {
        ImageConfig {
            image_regex: String::from(
                r"_(?P<well>[A-P]\d{2})_(?P<site>s[1-9])_(?P<channel>w[1-5])",
            ),
            target_size_x: 108,
            target_size_y: 108,
            num_sites_x: 3,
            num_sites_y: 3,
            intensity_cutoff_low: 5,
            intensity_cutoff_high: 200,
            intensity_addition: 10,
            num_rows: 16,
            num_columns: 24
        }
    }
}

impl Display for ImageConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ImageConfig(\n    image_regex: r\"{}\",\n    target_size_x: {},\n    target_size_y: {},\n    num_sites_x: {},\n    num_sites_y: {},\n    intensity_cutoff_low: {},\n    intensity_cutoff_high: {},\n    intensity_addition: {}\n   num_rows: {},\n     num_columns: {},\n)",
            self.image_regex,
            self.target_size_x,
            self.target_size_y,
            self.num_sites_x,
            self.num_sites_y,
            self.intensity_cutoff_low,
            self.intensity_cutoff_high,
            self.intensity_addition,
            self.num_rows,
            self.num_columns      
        )
    }
}

struct FileMetadata {
    row: char,
    column: u32,
    site: u32,
}

fn main() {
    let config = parse_args();
    write_overview_image(&config);
}

fn write_overview_image(conf: &AppConfig) {
    // let re = Regex::new(r"_([A-P]\d{2})_(s[1-9])_(w[1-5])").unwrap();
    let ImageConfig {
        ref image_regex,
        target_size_x,
        target_size_y,
        num_sites_x,
        num_sites_y,
        intensity_cutoff_low,
        intensity_cutoff_high,
        intensity_addition,
        num_rows,
        num_columns
    } = conf.image_config;
    let re = Regex::new(image_regex).unwrap_or_else(|_| {
        eprintln!("{} {}", "Could not compile regex:".red(), image_regex.red());
        std::process::exit(1);
    });
    let num_images_expected = num_sites_x * num_sites_y * 96;

    // A 96-er plate has 12 columns and 8 rows.
    let img_width_out: u32 = target_size_x * num_columns * num_sites_x; // 2x2 = 4 sites
    let img_height_out: u32 = target_size_y * num_rows * num_sites_y; // 2x2 = 4 sites

    // Load font for image watermarking
    let font = Vec::from(include_bytes!("../assets/DejaVuSans.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();

    let height = 22.0;
    let scale = Scale {
        x: height, // * 1.0,
        y: height,
    };

    // Initialize an image buffer with the appropriate size.
    let mut imgbuf = image::ImageBuffer::new(img_width_out, img_height_out);

    let mut img_ctr: u32 = 0;
    let mut file_metadata_list: Vec<FileMetadata> = Vec::new();
    let pattern = format!("{}/**/*.tif", conf.plate_dir);
    for entry in glob(&pattern).expect("Invalid glob pattern") {
        if let Err(e) = entry {
            eprintln!("{:?}", e);
            continue;
        }
        let img_path = entry.unwrap();
        let img_path_str = img_path.to_str().unwrap();
        if img_path_str.contains("_thumb") {
            continue;
        }
        let file_metadata = match FileMetadata::from_path(img_path_str, &re, conf.channel) {
            Ok(None) => continue, // Not the correct channel, continue with next file
            Ok(Some(fmd)) => fmd, // Retrieved file metadata
            Err(_) => {
                // Could not parse file metadata
                panic!("Could not parse file metadata for {}", img_path_str);
            }
        };

        let img = match image::open(img_path.clone()) {
            Ok(img) => img.into_luma8(),
            Err(_) => {
                eprintln!("{}: {} {}", "Failed to load image".red(), img_path_str.red(), "replacing with blank image".yellow());
                // Create a blank image with the target size (black or white as needed)
                GrayImage::new(target_size_x, target_size_y)
            }
        };

        let img_resized = resize(&img, target_size_x, target_size_y, FilterType::Nearest);

        let (img_x, img_y) = get_img_pos(&file_metadata, &conf.image_config);
        imgbuf.copy_from(&img_resized, img_x, img_y).unwrap();
        img_ctr += 1;
        if file_metadata.site == 1 {
            // Separate list for the Well_Id labels
            file_metadata_list.push(file_metadata);
        }
    }

    let upper = max_intensity(&imgbuf, (10, intensity_cutoff_high), 0.1) + intensity_addition;
    imgbuf = stretch_contrast(&imgbuf, intensity_cutoff_low, upper);

    // Label the wells
    // This is done so late because the image is stretched to the maximum intensity before labeling
    // Only write the Well_Id in the top left corner (site 1)
    for fmd in file_metadata_list.iter() {
        let well_id = format!("{}{:02}", fmd.row, fmd.column);
        let (img_x, img_y) = get_img_pos(fmd, &conf.image_config);
        draw_text_mut(
            &mut imgbuf,
            Luma([255u8]),
            img_x + 5,
            img_y + 5,
            scale,
            &font,
            &well_id,
        );
    }

    // Draw grid lines between wells
    for row in 1..16 {
        let start_x = 0 as f32;
        let end_x = img_width_out as f32;
        let y = (row * target_size_y * num_sites_y) as f32;
        draw_line_segment_mut(&mut imgbuf, (start_x, y), (end_x, y), Luma([175u8]));
    }
    for col in 1..24 {
        let x = (col * target_size_x * num_sites_x) as f32;
        let start_y = 0 as f32;
        let end_y = img_height_out as f32;
        draw_line_segment_mut(&mut imgbuf, (x, start_y), (x, end_y), Luma([175u8]));
    }

    imgbuf
        .save(&conf.output_file)
        .expect("Failed to write overview image.");
    if img_ctr != num_images_expected {
        eprintln!(
            "Expected {} images, but found {}",
            num_images_expected, img_ctr
        );
    }
}

impl FileMetadata {
    fn from_path(path: &str, re: &Regex, channel: u8) -> anyhow::Result<Option<Self>> {
        let Some(captures) = re.captures(path) else {
            return Err(anyhow::anyhow!("Invalid pattern in path: {}", path));
        };
        let chnel = captures["channel"]
            .chars()
            .nth(1)
            .unwrap()
            .to_digit(10)
            .unwrap();
        if chnel != channel as u32 {
            return Ok(None);
        }
        let well = &captures["well"];
        let row = well.chars().next().unwrap();
        let column = well[1..].parse::<u32>().unwrap();
        let site = captures["site"]
            .chars()
            .nth(1)
            .unwrap()
            .to_digit(10)
            .unwrap();

        Ok(Some(FileMetadata { row, column, site }))
    }
}

// Calculates the intensity with the highest number of pixels in the image
// between the upper and lower bounds and above the given threshold in percent.
fn max_intensity(img: &GrayImage, between: (u8, u8), threshold: f32) -> u8 {
    let hist = histogram(img);
    let mut max_intensity = between.0;
    let num_pixels = (img.width() * img.height()) as f32;
    for (idx, &count) in hist.channels[0]
        .iter()
        .enumerate()
        .skip(between.0 as usize)
        .take((between.1 - between.0) as usize)
    {
        let idx = idx as u8;
        if count as f32 >= (threshold / 100.0 * num_pixels) {
            max_intensity = idx;
        }
    }
    max_intensity
}

// Calculates the the position in the overview image from the file name metadata.
fn get_img_pos(metadata: &FileMetadata, img_conf: &ImageConfig) -> (u32, u32) {
    let row = metadata.row as u32 - 64; // A=1, B=2, ... P=16
    let row_index = row - 1;
    let column = metadata.column;
    let column_index = column - 1;
    let site = metadata.site;
    let (site_offset_x, site_offset_y) = site_offset(site, img_conf);
    let x = (column_index * img_conf.target_size_x * img_conf.num_sites_x) + site_offset_x;
    let y = (row_index * img_conf.target_size_y * img_conf.num_sites_y) + site_offset_y;
    (x, y)
}

// Calculates the offset for the individual sites of the well.
fn site_offset(site: u32, img_conf: &ImageConfig) -> (u32, u32) {
    let site_idx = site - 1;
    let x = (site_idx % img_conf.num_sites_x) * img_conf.target_size_x;
    let y = (site_idx / img_conf.num_sites_y) * img_conf.target_size_y;
    (x, y)
}

fn parse_args() -> AppConfig {
    let args: Vec<String> = std::env::args().collect();
    let program = args[0].as_ref();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    opts.optflag(
        "s",
        "show-config",
        "print the configuration and exit.\nThe output can be used to create a configuration file.",
    );
    opts.optopt(
        "w",
        "",
        "the microscope channel to analyze (REQUIRED, 1-5)",
        "NUMBER[1-5]",
    );
    opts.optopt(
        "o",
        "output",
        "the overview output file to generate (PNG or JPG)",
        "FILE",
    );
    opts.optopt("c", "config", "optional config file to load", "FILE");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(_) => {
            eprintln!("{}\n", "Error parsing commandline arguments.".red());
            print_usage(program, &opts);
            std::process::exit(1);
        }
    };
    if matches.opt_present("h") {
        print_usage(program, &opts);
        std::process::exit(0);
    }
    let img_config = match matches.opt_str("c") {
        Some(path) => {
            let path = Path::new(&path);
            if path.is_file() {
                load_config_from_file(path).unwrap_or_else(|err| {
                    eprintln!("{}", err.to_string().red());
                    std::process::exit(1);
                })
            } else {
                eprintln!("\n{}\n", "Config file does not exist.".red());
                std::process::exit(1);
            }
        }
        None => {
            load_config_from_dir().unwrap_or_else(|_| {
                // eprintln!("{}", err.to_string().red());
                eprintln!("Using default image configuration.");
                ImageConfig::default()
            })
        }
    };
    if matches.opt_present("s") {
        println!("Used configuration:\n\n{}", img_config);
        std::process::exit(0);
    }
    // let channel = parse_channel(matches.opt_str("w"));
    let channel = matches
        .opt_str("w")
        .unwrap_or_else(|| {
            eprintln!("{}\n", "Missing required channel argument: -w".red());
            print_usage(program, &opts);
            std::process::exit(1);
        })
        .parse::<u8>()
        .unwrap_or_else(|x| {
            eprintln!(
                "{}{}",
                "Invalid channel number ".red(),
                format!("{}", x).red()
            );
            eprintln!("Channel has to be a number between 1 and 5.\n");
            print_usage(program, &opts);
            std::process::exit(1);
        });
    if !(1..=5).contains(&channel) {
        eprintln!(
            "{}{}",
            "Channel number out of range: ".red(),
            format!("{}", channel).red()
        );
        eprintln!("Channel number has to be between 1 and 5.\n");
        print_usage(program, &opts);
        std::process::exit(1);
    }
    let output_file = matches.opt_str("o").unwrap_or_else(|| {
        eprintln!("Missing argument: --output. Using default name");
        format!("overview_w{channel}.png")
    });
    let plate_dir = if !matches.free.is_empty() {
        matches.free[0].clone()
    } else {
        eprintln!(
            "{}\n",
            "Missing required positinal argument IMAGE_DIR.".red(),
        );
        print_usage(program, &opts);
        std::process::exit(1);
    };

    AppConfig {
        plate_dir,
        output_file,
        channel,
        image_config: img_config,
    }
}

fn print_usage(program: &str, opts: &Options) {
    let brief = format!(
        "\
Plate inspector, A visual inspector of Cell Painting plates.
Creates a single image overview of all individual images for the given channel.
v {} ({}), by {}. Written in Rust.\n
Configuration files in the `RON` format can be put
at <UserConfigDir>/plate-inspector.ron
or passed at the command line with the `-c` flag.
An example config file can be found at the root of this project.
If no config file is found, a default configuration is used.\n
Usage: {} IMAGE_DIR -w CHANNEL [-c CONFIG_FILE] [-o OUTPUT_FILE]\n
Example: {} queue/C2021-04-10.00-211126-A -w1 -o C2021-04-10.00-211126-A_w1.png",
        &VERSION, &BUILDTIME, &AUTHORS, program, program
    );
    print!("{}", opts.usage(&brief));
}

fn load_config_from_dir() -> Result<ImageConfig> {
    let Some(config_dir) = dirs::config_dir() else {
        return Err(anyhow!("Could not find config directory"))
    };
    let config_file = config_dir.join("plate-inspector.ron");
    if config_file.is_file() {
        load_config_from_file(config_file)
    } else {
        Err(anyhow!("Could not find config file"))
    }
}

fn load_config_from_file<P: AsRef<Path>>(config_file: P) -> Result<ImageConfig> {
    let img_opts_str = std::fs::read_to_string(config_file)?;
    Ok(ron::from_str(&img_opts_str)?)
}
