use std::sync::Once;

static INIT_LOGGING: Once = Once::new();

pub fn init() {
    use env_logger::{
        Env, Builder,
    };
    INIT_LOGGING.call_once(|| {
        #[cfg(debug_assertions)]
        const DEFAULT_LOG_FILTER: &'static str = concat!(env!("CARGO_CRATE_NAME"), "=debug,generator=debug");
        #[cfg(not(debug_assertions))]
        const DEFAULT_LOG_FILTER: &'static str = concat!(env!("CARGO_CRATE_NAME"), "=info,generator=info");
        let cfg = Env::default()
            .default_filter_or(DEFAULT_LOG_FILTER);
        Builder::from_env(cfg).init();
    });
}
