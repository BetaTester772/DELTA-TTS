// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "DeltaTTSDemo",
    platforms: [.macOS(.v26)],
    targets: [
        .executableTarget(name: "DeltaTTSDemo", path: "Sources/DeltaTTSDemo"),
    ]
)
