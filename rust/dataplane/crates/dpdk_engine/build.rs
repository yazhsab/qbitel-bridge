use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to tell rustc to link the DPDK libraries
    println!("cargo:rustc-link-lib=dpdk");
    println!("cargo:rustc-link-lib=numa");
    
    // Generate bindings for DPDK headers
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate DPDK bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Detect DPDK installation path
    if let Ok(rte_sdk) = env::var("RTE_SDK") {
        println!("cargo:rustc-link-search=native={}/lib", rte_sdk);
        println!("cargo:rustc-link-search=native={}/lib/x86_64-linux-gnu", rte_sdk);
    }
    
    // Look for pkg-config
    if let Ok(lib) = pkg_config::probe("libdpdk") {
        for path in lib.link_paths {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
}