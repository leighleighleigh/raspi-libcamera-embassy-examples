{ nixpkgs ? import <nixpkgs> {}}:
let 
  #rustOverlay = builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz";
  rustOverlay = nixpkgs.fetchFromGitHub {
    owner = "oxalica";
    repo = "rust-overlay";
    rev = "a0b81d4fa349d9af1765b0f0b4a899c13776f706";
    hash = "sha256-IKrk7RL+Q/2NC6+Ql6dwwCNZI6T6JH2grTdJaVWHF0A=";
  };

  pkgs = import <nixpkgs> {
    overlays = [ (import rustOverlay) ];
  };

  rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
in
  pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } rec {
    buildInputs = [ rust ] ++ (with pkgs; [
      #bacon 
      #gcc 
      #rust-analyzer
      #stdenv.cc 
      #systemd
      pkg-config
      glib
      zlib
      libusb1
      just
      
      # cross-rs needs this binary
      #rustup

      # for libcamera-rs
      libcamera
      libclang
    ]);

    LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";

    shellHook = ''
        export PS1="''${debian_chroot:+($debian_chroot)}\[\033[01;39m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\$ "
        export PS1="(nix-rs)$PS1"
        export LD_LIBRARY_PATH="''${LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
    '';
  }
