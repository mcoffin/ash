use generator::{
    write_source_code,
    VK_CONFIG,
};
use std::{
    borrow::Cow,
    error::Error,
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};
use lazy_static::lazy_static;

#[path = "../logging.rs"]
mod logging;

#[derive(Debug, clap::Parser)]
#[clap(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    #[clap(long = "output-directory", short = 'o')]
    out_dir: Option<PathBuf>,
    #[clap(long, env = "ASH_VK_REGISTRY")]
    vk_registry: Option<PathBuf>,
    #[cfg(feature = "openxr")]
    #[clap(long, env = "ASH_XR_REGISTRY")]
    xr_registry: Option<PathBuf>,
}

type SharedError = Arc<dyn Error + Send + Sync + 'static>;

lazy_static! {
    static ref PROJECT_DIR: Result<PathBuf, SharedError> = {
        let mut d = std::env::current_dir()
            .map_err(|e| Arc::new(e) as Arc<dyn Error + Send + Sync + 'static>)?;
        while !d.join("ash/src").exists() {
            if !d.pop() {
                return Err(Arc::new(ProjectDirectoryNotFound) as _);
            }
        }
        Ok(d)
    };
}

#[inline(always)]
fn project_dir() -> Result<&'static Path, SharedError> {
    PROJECT_DIR.as_ref()
        .map_err(Clone::clone)
        .map(AsRef::as_ref)
}

impl Args {
    fn output_dir<'a>(&'a self) -> Result<&'a Path, SharedError> {
        if let Some(p) = self.out_dir.as_deref() {
            return Ok(p);
        }
        project_dir()
    }
    fn vk_registry(&self) -> Result<Cow<'_, Path>, SharedError> {
        if let Some(p) = self.vk_registry.as_deref() {
            return Ok(Cow::Borrowed(p));
        }
        project_dir()
            .map(|d| Cow::Owned(d.join("generator/Vulkan-Headers")))
    }
    #[cfg(feature = "openxr")]
    fn xr_registry(&self) -> &Path {
        self.xr_registry.as_deref()
            .unwrap_or("/usr/share/openxr".as_ref())
    }
}

fn main() {
    use clap::Parser;
    logging::init();
    let args = Args::parse();
    let output_dir = args.output_dir()
        .expect("failed to find output directory");
    let vk_registry_path = args.vk_registry()
        .expect("failed to find registry path");
    write_source_code(&VK_CONFIG, vk_registry_path.as_ref(), output_dir.join("ash/src"));
    #[cfg(feature = "openxr")]
    {
        use generator::openxr::XR_CONFIG;
        write_source_code(&XR_CONFIG, args.xr_registry(), output_dir.join("ash-xr/src"));
    }
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("unable to find project output directory automatically")]
struct ProjectDirectoryNotFound;

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("{0} registry not found")]
#[repr(transparent)]
struct RegistryNotFound<'a>(&'a str);
