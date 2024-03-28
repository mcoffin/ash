use generator::{
    write_source_code,
    VK_CONFIG,
};
use std::{
    borrow::Cow,
    ffi::OsStr,
    path::{
        Path,
        PathBuf,
    },
};

#[inline(always)]
fn env_or<'a, K: AsRef<OsStr>>(key: K, default_value: &'a Path) -> Cow<'a, Path> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .map_or(Cow::Borrowed(default_value), Cow::Owned)
}

fn main() {
    eprintln!("writing {} bindings", VK_CONFIG.desired_api);
    let cwd = std::env::current_dir().unwrap();
    let (vk_path_default, out_path) = if cwd.ends_with("generator") {
        (Path::new("Vulkan-Headers"), "../ash/src")
    } else {
        (Path::new("generator/Vulkan-Headers"), "ash/src")
    };
    let vk_path = env_or("ASH_VK_REGISTRY", vk_path_default);
    write_source_code(&VK_CONFIG, vk_path.as_ref(), out_path);

    #[cfg(feature = "openxr")]
    {
        use generator::openxr as xr;
        eprintln!("writing {} bindings", xr::XR_CONFIG.desired_api);
        let xr_dir = if cwd.ends_with("generator") {
            "../ash-xr/src"
        } else {
            "ash-xr/src"
        };
        let xr_registry_path = env_or("ASH_XR_REGISTRY", "/usr/share/openxr".as_ref());
        write_source_code(&xr::XR_CONFIG, xr_registry_path.as_ref(), xr_dir);
    }
}
