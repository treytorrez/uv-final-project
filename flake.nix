{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }: let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [ pkgs.rocmPackages.clr ];
      shellHook = ''
        export LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib:$LD_LIBRARY_PATH
        exec zsh
      '';
    };
  };
}
