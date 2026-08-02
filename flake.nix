{
  description = "llama2.zig CUDA development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
          pkg-config
        ];

        PKG_CONFIG_PATH = "${pkgs.cudaPackages.cuda_cudart}/share/pkgconfig";
        LD_LIBRARY_PATH = "${pkgs.cudaPackages.cuda_cudart}/lib:/run/opengl-driver/lib";
      };
    };
}
